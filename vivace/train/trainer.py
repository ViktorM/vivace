"""Main training orchestrator.

Wires rollout backend, policy_gradient module, eval runner, logging, and
checkpointing into one loop. The loop is what matters; everything else is
delegated. RL hyperparameters live in `cfg.rl` (RLConfig); trainer-side
orchestration fields live directly on TrainerConfig.
"""

from __future__ import annotations

import gc
import os
import random
import socket
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import record_function
from transformers import AutoTokenizer, AutoModelForCausalLM

from vivace.algos.types import RLConfig, SFTConfig, validate_rl_config
from vivace.algos.policy_gradient import (
    compute_advantages, compute_token_logprobs, compute_kl, compute_loss, rl_step,
)
from vivace.envs.base import Env, Example
from vivace.envs.gsm8k import GSM8KEnv
from vivace.rollout.hf_sampler import sample_responses
from vivace.rollout.vllm_worker import VLLMRolloutWorker
from vivace.rewards import extract_answer
from vivace.eval.runner import evaluate_model, compare_metrics, sample_evaluate
from vivace.utils.stats import TrainingStats
from vivace.utils.logging import ConsoleLogger, init_wandb, log_metrics, finish_wandb
from vivace.utils.perf import Timer, throughput
from vivace.utils.profiling import ProfilingConfig, create_profiler, export_and_summarize
from vivace.utils.weight_sync import sender_broadcast_loop
from vivace.utils.distributed import is_main_process, barrier
from vivace.utils.checkpointing import save_checkpoint

from peft import LoraConfig, get_peft_model, TaskType


# --- dtype mapping ---
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,      # FP8 (Hopper/Ada) — forward pass only for now
    "fp8": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,          # FP8 alternate format (wider range, less precision)
}


@dataclass
class TrainerConfig:
    """Trainer orchestration config. RL-math hyperparameters live in the
    nested `rl: RLConfig` — see `vivace/algos/types.py`. This separation
    keeps training infrastructure (model loading, device placement, logging)
    distinct from the algorithm itself, and lets `rl_step` consume `RLConfig`
    directly without the trainer re-translating fields.
    """

    # ----- what to train -----
    model_name: str = "Qwen/Qwen2.5-0.5B"
    env_name: str = "gsm8k"
    algo_name: str = "grpo"

    # ----- execution mode -----
    mode: str = "colocated"          # "colocated" | "disaggregated"
    trainer_gpus: list[int] = field(default_factory=lambda: [0])
    rollout_gpus: list[int] = field(default_factory=lambda: [0])

    # ----- model dtype + compilation -----
    dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    compile_model: bool = False      # torch.compile — can break generate(), off by default

    # ----- LoRA -----
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # ----- training loop orchestration -----
    num_steps: int = 500
    use_vllm: bool = True
    gpu_memory_utilization: float = 0.4  # only used when use_vllm = True
    enforce_eager: bool = False          # disable CUDA graphs in vLLM (safer for weight updates, slower)
    vllm_max_model_len: int | None = None   # cap vLLM's context window. None → use model default (huge for Qwen3.5)
    vllm_max_num_seqs: int | None = None    # cap vLLM's concurrent-decode count. Qwen3.5 needs this to fit Mamba cache.

    # ----- RL hyperparameters (algorithm, optimizer, clipping, generation, adaptive sampling/LR) -----
    # Previously ~25 fields were duplicated here and copied into RLConfig at __init__.
    # Now they live canonically in RLConfig and are accessed via cfg.rl.X.
    rl: RLConfig = field(default_factory=RLConfig)

    # ----- SFT warmup (optional) -----
    sft_warmup: bool = False
    sft_warmup_steps: int = 0

    # ----- eval -----
    eval_n: int = 500               # number of eval examples (-1 = full eval set)
    eval_batch_size: int = 32        # batch size for eval generation
    eval_use_vllm: bool = True       # use vLLM for eval when available (much faster)
    eval_gpus: list[int] = field(default_factory=list)  # separate eval GPUs (empty = use rollout worker)

    # ----- logging -----
    wandb_project: str = "vivace"
    wandb_run_name: str | None = None
    log_interval: int = 1
    eval_interval: int = 50
    checkpoint_interval: int = 100
    run_dir: str = "runs/default"

    # ----- profiling -----
    profiling: dict | None = None    # maps to ProfilingConfig; None = disabled

    # ----- weight sync -----
    # "nccl": direct GPU->GPU broadcast via vLLM collective_rpc (default, fast).
    #         Works for full FT and LoRA (LoRA via merge-broadcast-unmerge).
    # "disk": LoRA adapter save + vLLM LoRARequest reload (simple, slower, kept
    #         as a fallback for debugging and as a reference path).
    weight_sync_method: str = "nccl"
    # Where the disk-sync path writes the adapter. None auto-picks /dev/shm (tmpfs,
    # RAM-backed) on Linux, falling back to run_dir if /dev/shm isn't a directory.
    # tmpfs is ~10× faster than NVMe for the small adapter files; persistence isn't
    # needed (each step overwrites). Override to a regular path if you want the
    # adapter on disk for inspection.
    weight_sync_disk_path: str | None = None

    # ----- misc -----
    seed: int = 42


def _find_free_port() -> int:
    """Bind to port 0 to let the OS pick a free loopback port, then release it.
    Small race window between releasing and another process grabbing it, but
    sufficient for one-shot rendezvous at process startup.
    """
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"unknown dtype {name!r}, expected one of {list(DTYPE_MAP)}")
    return DTYPE_MAP[name]


def _print_system_info() -> None:
    """Print GPU / system info at startup. Useful for confirming you're on the right machine."""
    print(f"\n{'=' * 60}")
    print(f"  vivace — system info")
    print(f"{'=' * 60}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {torch.version.cuda or 'N/A'}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  GPUs:      {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1e9
            print(f"    [{i}] {props.name}  {mem_gb:.1f} GB")
    else:
        print("  GPUs:      none (CPU mode)")
    print(f"{'=' * 60}\n")


def _print_training_info(
    cfg: TrainerConfig,
    model: nn.Module,
    trainer_devices: list[torch.device],
    rollout_devices: list[torch.device],
) -> None:
    """Print model + training config summary after model load."""
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = "LoRA" if cfg.use_lora else "full FT"

    def _fmt_devices(devs: list[torch.device]) -> str:
        return ", ".join(str(d) for d in devs)

    print(f"\n{'=' * 60}")
    print(f"  vivace — training config")
    print(f"{'=' * 60}")
    print(f"  Model:           {cfg.model_name}")
    print(f"  Trainer devices: {_fmt_devices(trainer_devices)}")
    print(f"  Rollout devices: {_fmt_devices(rollout_devices)}")
    print(f"  Dtype:           {cfg.dtype}")
    print(f"  Compile:         {cfg.compile_model}")
    print(f"  Params:          {n_total / 1e6:.1f}M total, {n_trainable / 1e6:.1f}M trainable ({mode})")
    print(f"  Vocab:           {model.config.vocab_size}")
    print(f"  Hidden:          {model.config.hidden_size}")
    print(f"  Layers:          {model.config.num_hidden_layers}")
    if hasattr(model.config, 'layer_types'):
        print(f"  Layer types:     {model.config.layer_types}")
    print(f"  Env:             {cfg.env_name}")
    print(f"  Algo:            {cfg.rl.loss_type} / {cfg.rl.adv_type}")
    print(f"  Batch:           {cfg.rl.batch_size} prompts x {cfg.rl.group_size} group x {cfg.rl.grad_accum_steps} accum")
    print(f"  LR:              {cfg.rl.lr}")
    print(f"  KL coef:         {cfg.rl.kl_coef}")
    print(f"  Steps:           {cfg.num_steps}")
    print(f"  Run dir:         {cfg.run_dir}")
    print(f"{'=' * 60}\n")


class Trainer:
    """Training orchestrator. Owns model, rollout backend, env, RLConfig, and loop.

    Loop invariants:
      - Weight sync runs AFTER optimizer.step(), BEFORE the next rollout.
        Flipping this makes the rollout use stale weights, biasing the importance ratio.
      - Eval and checkpoint run on rank 0 only; other ranks must hit the same `barrier()`.
      - Metric logging all-reduces BEFORE the rank-0 print.
    """

    def __init__(self, cfg: TrainerConfig):
        """Build model, tokenizer, env, rollout backend, optimizer, stats, and
        (optionally) the NCCL weight-sync comm.

        Wrap order for LoRA + DDP: load base -> peft wrap -> .to(device) -> DDP.
        Other orders silently break checkpointing or gradient sync.

        Ref model: LoRA uses `with model.disable_adapter():` for zero extra VRAM.
        Full FT needs a frozen deepcopy (~2× model VRAM).
        """
        self.cfg = cfg
        self.env = GSM8KEnv()  # TODO: dispatch on cfg.env_name for other envs
        self.train_data = self.env.load_split("train")

        self.rollout_worker = None

        _print_system_info()
        torch.backends.cuda.matmul.allow_tf32 = True
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # --- Resolve devices ---
        # Full lists: for DDP/FSDP (training) and vLLM TP (rollout).
        # Primary device: where the model gets loaded initially and where
        # single-GPU ops (optimizer step, non-distributed forward) happen.
        # In DDP, each rank gets its own primary via LOCAL_RANK; for now
        # (single-process) we use [0] from each list.
        if torch.cuda.is_available():
            self.trainer_devices = [torch.device(f"cuda:{g}") for g in cfg.trainer_gpus]
            self.rollout_devices = [torch.device(f"cuda:{g}") for g in cfg.rollout_gpus]
        else:
            self.trainer_devices = [torch.device("cpu")]
            self.rollout_devices = [torch.device("cpu")]
        self.device = self.trainer_devices[0]  # primary — model load target

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # --- Model ---
        model_dtype = _resolve_dtype(cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=model_dtype,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(self.device)
        self.model.config.use_cache = False
        self.model.train()

        # --- LoRA + reference model ---
        self.lora_config = None
        if cfg.use_lora:
            self.lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
            )
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
            # With LoRA the base model IS the reference — use
            # `with model.disable_adapter():` in rollout_phase to forward
            # through the base weights. Zero extra memory.
            self.ref_model = None
        else:
            # Full FT: need a separate frozen copy (~2x model VRAM).
            self.ref_model = deepcopy(self.model).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        # --- Compile (optional) ---
        # Use compiled versions for forward passes (rl_step, log-prob recomputation),
        # raw self.model for generate() — torch.compile and generate() don't mix well.
        # For LoRA: compiled_ref is None (use compiled_model with disable_adapter).
        self.compiled_model = torch.compile(self.model) if cfg.compile_model else self.model
        if self.ref_model is not None:
            self.compiled_ref = torch.compile(self.ref_model) if cfg.compile_model else self.ref_model
        else:
            self.compiled_ref = None  # LoRA: use compiled_model with disable_adapter()

        if cfg.use_vllm:
            # vLLM only needs Punica (enable_lora=True) when we're pushing LoRA
            # adapters to it via LoRARequest (disk path). For NCCL sync with LoRA
            # we merge on the trainer and broadcast merged base weights — vLLM
            # runs pure base generation, and enable_lora=True would re-parent
            # its linear weights under `.base_layer.weight`, breaking name
            # matching in the NCCL broadcast loop.
            vllm_enable_lora = cfg.use_lora and cfg.weight_sync_method == "disk"
            self.rollout_worker = VLLMRolloutWorker(
                model_name=cfg.model_name,
                gpu_ids=cfg.rollout_gpus,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                enable_lora=vllm_enable_lora,
                colocated=(cfg.mode == "colocated"),
                enforce_eager=cfg.enforce_eager,
                max_model_len=cfg.vllm_max_model_len,
                max_num_seqs=cfg.vllm_max_num_seqs,
            )
        else:
            self.rollout_worker = None  # HF sampler path

        # --- RLConfig: now owned directly by TrainerConfig.rl, no re-translation needed ---
        self.rl_cfg = cfg.rl
        # Trainer-side log_interval controls console/wandb cadence and is
        # separate from RLConfig's log_interval (which controls rl_step's own
        # internal prints). Keep the trainer one authoritative by mirroring.
        self.rl_cfg.log_interval = cfg.log_interval
        validate_rl_config(self.rl_cfg)

        # --- Optimizer + scheduler ---
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.rl.lr)
        warmup_steps = 10
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(cfg.num_steps - warmup_steps, 1),
            eta_min=cfg.rl.lr * 0.2)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        # --- Stats + logging ---
        # Label includes key config differences for comparison plots
        label = f"{cfg.rl.loss_type}/{cfg.rl.adv_type}"
        if cfg.use_lora:
            label += f" LoRA r={cfg.lora_rank}"
        if cfg.rl.adaptive_sampling:
            label += f" adapt={cfg.rl.oversample_factor}x"
        label += f" kl={cfg.rl.kl_coef}"
        label += f" ep={cfg.rl.optim_epochs}"
        self.stats = TrainingStats(method=label)
        self.console = ConsoleLogger(label=label)
        init_wandb(cfg, project=cfg.wandb_project, run_name=cfg.wandb_run_name)

        # --- Eval worker ---
        # For now, reuse the rollout worker for eval when available.
        # Future: dedicated eval workers on separate GPUs (cfg.eval_gpus).
        self.eval_worker = None
        if cfg.eval_use_vllm and self.rollout_worker is not None:
            self.eval_worker = self.rollout_worker
        elif cfg.eval_gpus:
            # TODO: build a dedicated VLLMRolloutWorker on eval_gpus
            # self.eval_worker = VLLMRolloutWorker(
            #     model_name=cfg.model_name,
            #     gpu_ids=cfg.eval_gpus,
            #     gpu_memory_utilization=cfg.gpu_memory_utilization,
            #     enable_lora=cfg.use_lora,
            #     colocated=False,
            # )
            pass

        # --- Profiling ---
        self.profiling_cfg = ProfilingConfig(**(cfg.profiling or {}))

        # ----- NCCL weight sync setup (one-time, Pattern A) -----
        # Pattern A = StatelessProcessGroup (TCP rendezvous) + PyNcclCommunicator.
        # See docs/training_theory.md and docs/weight_sync_approaches.md.
        self._nccl_sync_state = None
        if cfg.weight_sync_method == "nccl" and self.rollout_worker is not None:
            # NCCL can't communicate between two ranks on the same GPU device
            # (ncclCommInitRank rejects with NCCL_INVALID_USAGE). For colocated
            # mode, disk sync is the only working option.
            if cfg.mode == "colocated":
                shared = set(cfg.trainer_gpus) & set(cfg.rollout_gpus)
                if shared:
                    raise ValueError(
                        f"weight_sync_method='nccl' requires distinct trainer/rollout GPUs, "
                        f"but mode='colocated' and trainer_gpus={cfg.trainer_gpus} share "
                        f"{sorted(shared)} with rollout_gpus={cfg.rollout_gpus}. "
                        f"Use weight_sync_method='disk' for colocated mode, or set "
                        f"mode='disaggregated' with separate GPUs."
                    )
            from vllm.distributed.utils import StatelessProcessGroup
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vivace.utils.weight_sync import build_param_specs, allocate_fused_buffers

            host, port = "localhost", _find_free_port()
            world_size = 2   # 1 trainer + 1 vLLM worker at TP=1; bump when scaling

            # Worker-side init runs inside the vLLM subprocess via collective_rpc.
            # collective_rpc blocks the caller until the function returns on the
            # worker, so we run it on a background thread while the trainer's own
            # StatelessProcessGroup.create is called concurrently on the main
            # thread. Both sides must be in their create() call at the same time
            # for the TCP rendezvous to complete.
            worker_err: list[BaseException] = []
            def _trigger_worker_init():
                try:
                    self.rollout_worker.init_weight_sync(
                        master_addr=host, master_port=port,
                        trainer_rank=0, worker_rank_offset=1, world_size=world_size,
                    )
                except BaseException as e:
                    worker_err.append(e)
                    print(f"[trainer init] worker-side NCCL init failed: {e!r}", flush=True)

            worker_thread = threading.Thread(target=_trigger_worker_init, daemon=True)
            worker_thread.start()

            # Trainer-side rendezvous. If the worker errored, the trainer's
            # TCPStore will time out — surface the worker error first.
            try:
                pg = StatelessProcessGroup.create(
                    host=host, port=port, rank=0, world_size=world_size,
                    store_timeout=30,
                )
            except BaseException:
                if worker_err:
                    raise RuntimeError(
                        f"NCCL rendezvous failed because the vLLM worker errored: {worker_err[0]!r}"
                    ) from worker_err[0]
                raise
            trainer_comm = PyNcclCommunicator(group=pg, device=self.device)

            worker_thread.join(timeout=60)
            if worker_err:
                raise worker_err[0]

            # Build broadcast specs from the (peft-wrapped) model. For LoRA, the
            # `merge_adapter` call in `_sync_weights_nccl` folds B@A into base before
            # broadcast and `unmerge_adapter` undoes it after, so vLLM (enable_lora=False
            # on this path) receives the full merged weights via these specs. Full FT
            # path ignores merge/unmerge and broadcasts the live trainable params.
            specs, fusion_map = build_param_specs(self.model, fuse=True)
            fused_buffers = allocate_fused_buffers(self.model, specs, self.device)
            self._nccl_sync_state = {
                "specs": specs,
                "fusion_map": fusion_map,
                "fused_buffers": fused_buffers,
                "comm": trainer_comm,
                "pg": pg,
            }
            print(f"[trainer init] NCCL weight sync ready: {len(specs)} params")

        # Resolve disk-sync path once. Default: /dev/shm/vivace_sync_<run_basename>
        # (RAM-backed tmpfs). If /dev/shm isn't a dir (non-Linux) or user supplied
        # a path, use that instead.
        if cfg.weight_sync_disk_path is not None:
            self._resolved_sync_disk_path = cfg.weight_sync_disk_path
        elif os.path.isdir("/dev/shm"):
            run_tag = os.path.basename(cfg.run_dir.rstrip("/")) or "default"
            self._resolved_sync_disk_path = f"/dev/shm/vivace_sync_{run_tag}"
        else:
            self._resolved_sync_disk_path = os.path.join(cfg.run_dir, "adapter_sync")

        _print_training_info(cfg, self.model, self.trainer_devices, self.rollout_devices)
        if cfg.weight_sync_method == "disk":
            print(f"  [disk-sync] adapter path: {self._resolved_sync_disk_path}")

    def _run_eval(self, eval_data: list, label: str = "") -> tuple[dict, list, list]:
        """Run evaluation using the best available backend."""
        n = len(eval_data) if self.cfg.eval_n == -1 else self.cfg.eval_n
        return evaluate_model(
            self.model, self.tokenizer, eval_data, self.env,
            n=n,
            batch_size=self.cfg.eval_batch_size,
            max_new_tokens=self.cfg.rl.max_new_tokens,
            device=str(self.device),
            vllm_worker=self.eval_worker,
        )

    def maybe_sft_warmup(self) -> None:
        """Run a short SFT pass before RL if `cfg.sft_warmup`. Unused by default.

        TODO: implement when SFT warmup is actually needed.
        """
        pass

    def rollout_phase(self) -> list[dict]:
        """Sample prompts -> generate -> score -> compute advantages. Returns
        `grad_accum_steps` micro-batches in the format `rl_step` expects:
        {full_ids, plen, adv, old_logp, ref_logp, mask, token_count, responses, rewards}.

        old_logp is computed under the same model that just sampled (fine in
        our synchronous setup). LoRA ref logp uses `model.disable_adapter()`
        to forward through base weights without a separate ref model.
        """
        micro_batches, alive_rates_step, spread_step = [], [], []
        rl = self.cfg.rl
        B, G = rl.batch_size, rl.group_size

        n_prompts_per_mb = int(rl.oversample_factor * B) if rl.adaptive_sampling else B
        n_prompts = n_prompts_per_mb * rl.grad_accum_steps # generate all prompts at once
        idxs = np.random.choice(len(self.train_data), size=n_prompts, replace=False)

        batch_ex_all = [self.train_data[i] for i in idxs]

        pad_id = self.tokenizer.pad_token_id

        if self.rollout_worker is not None:
            # --- vLLM path ---
            # 1. Build unique_prompts (B, no G-repeat — vLLM handles n=G internally)
            unique_prompts = [self.env.format_prompt(ex) for ex in batch_ex_all]

            # 2. Tokenize with HF tokenizer for consistent token IDs.
            # Left-padded version (used to build full_ids — keeps `plen` a single int
            # downstream so HF forwards process the batch in one shot).
            prompt_enc = self.tokenizer(unique_prompts, return_tensors="pt", padding=True)
            prompt_ids_batch_all = prompt_enc.input_ids.tolist()
            plen = prompt_enc.input_ids.shape[1]

            # 2b. vLLM has no concept of left-padding — it would condition generation
            # on `[pad ... pad | prompt]` and produce different logits than HF (which
            # masks the pads via attention_mask). Strip leading pads per-prompt before
            # handing to vLLM so both sides condition on the actual prompt.
            prompt_ids_for_vllm = []
            for ids in prompt_ids_batch_all:
                first_real = next((j for j, t in enumerate(ids) if t != pad_id), len(ids))
                prompt_ids_for_vllm.append(ids[first_real:])

            # 3. Generate via the worker (handles lora_request, tqdm, etc.)
            with record_function("vllm_generate"):
                vllm_outputs_all, _ = self.rollout_worker.generate(
                    prompt_token_ids=prompt_ids_for_vllm,
                    temperature=rl.temperature, top_p=rl.top_p,
                    max_tokens=rl.max_new_tokens, n=G,
                )

        # {full_ids, plen, adv, old_logp, ref_logp, mask, token_count, responses, rewards}
        for i in range(rl.grad_accum_steps):

            start, end = i * n_prompts_per_mb, (i + 1) * n_prompts_per_mb
            batch_ex = batch_ex_all[start : end]

            if self.rollout_worker is not None:
                vllm_outputs = vllm_outputs_all[start : end]
                prompt_ids_batch = prompt_ids_batch_all[start : end]

                # 4. Build full_ids + responses from vLLM output.
                # Pad all responses to the same length so plen works as a
                # single int (same structure as HF generate output).
                all_resp_ids = []
                responses = []
                for req_output in vllm_outputs:
                    for completion in req_output.outputs:
                        all_resp_ids.append(list(completion.token_ids))
                        responses.append(completion.text)

                # Record true pre-padding lengths so the trainer can build correct
                # attention + loss masks. Without this, right-pads get attended to
                # (pad_id == eos_id) and truncated responses leak one pad into the loss.
                response_lengths = torch.tensor(
                    [len(r) for r in all_resp_ids], device=self.device, dtype=torch.long,
                )
                max_resp_len = max(len(r) for r in all_resp_ids)
                # Right-pad responses to uniform length (prompt is already uniform)
                all_resp_ids = [r + [pad_id] * (max_resp_len - len(r)) for r in all_resp_ids]

                # 5. Build full_ids: [left_pad_prompt | prompt | response | right_pad_response]
                # prompt_ids_batch has shape [B, plen] — repeat each G times
                all_ids = []
                for b_idx, req_output in enumerate(vllm_outputs):
                    p_ids = prompt_ids_batch[b_idx]  # already left-padded to plen
                    for g_idx in range(len(req_output.outputs)):
                        flat_idx = b_idx * len(req_output.outputs) + g_idx
                        all_ids.append(p_ids + all_resp_ids[flat_idx])

                full_ids = torch.tensor(all_ids, device=self.device)
                # plen is correct: all prompts padded to same length, all responses
                # padded to same length, total = plen + max_resp_len for every sequence.

                # 6. Build examples (same as HF path)
                examples = [ex for ex in batch_ex for _ in range(G)]

            else:
                # --- HF path ---

                # Flatten: each prompt repeated G times, each answer repeated G times
                # TODO: investigate vectorizing format_prompt — would need Env.format_prompt_batch(examples)
                # or a batched string template. Current per-example call is ~microseconds so low priority,
                # but at large batch sizes (256+ prompts) the Python loop may show up in profiles.

                # prompts = [self.env.format_prompt(ex) for ex in batch_ex for _ in range(G)]
                prompts = np.repeat([self.env.format_prompt(ex) for ex in batch_ex], G).tolist()
                examples = [ex for ex in batch_ex for _ in range(G)]

                with record_function("hf_generate"):
                    full_ids, responses, plen, response_lengths = sample_responses(
                        self.model, self.tokenizer, prompts,
                        max_new_tokens=rl.max_new_tokens,
                        temperature=rl.temperature, top_p=rl.top_p,
                        device=str(self.device),
                    )
                torch.cuda.empty_cache()

            # Compute rewards for ALL responses (needed for adaptive sampling spread)
            # reward_fn takes (response, Example) — full example available for
            # problem-aware rewards (e.g. bonus for using numbers from the problem)
            with record_function("reward"):
                rewards = torch.tensor(
                    [self.env.reward_fn(resp, ex) for resp, ex in zip(responses, examples)],
                    device=self.device, dtype=torch.float32,
                )

            if rl.adaptive_sampling:
                rg = rewards.view(n_prompts_per_mb, G)
                spread = rg.max(dim=1).values - rg.min(dim=1).values
                top_idx = spread.argsort(descending=True)[:B]

                alive_count = (spread > rl.min_reward_spread).sum().item()
                alive_rates_step.append(alive_count / n_prompts_per_mb)
                spread_step.append((spread.mean().item(), spread.max().item()))

                keep = [j for idx in top_idx for j in range(idx * G, (idx + 1) * G)]
                full_ids = full_ids[keep]
                rewards = rewards[keep]
                responses = [responses[k] for k in keep]
                if response_lengths is not None:
                    response_lengths = response_lengths[keep]

            with record_function("advantages"):
                adv = compute_advantages(rewards.view(B, G), self.rl_cfg)

            # old_logp via recompute (peft separate-matmul forward). For LoRA, this
            # matches policy_logp's forward path → ratio is well-behaved. We tried
            # using vLLM's old_logp directly (skips the recompute, ~700ms/step win)
            # but the peft-vs-vLLM-merged bf16 numerical gap biases importance ratios
            # by an irreducible amount that grows with ‖B@A‖, degrading sample
            # efficiency. Until weight sync supports vLLM enable_lora=True (whose
            # Punica fused-LoRA forward matches peft's separate-matmul numerics),
            # recompute is the principled choice for LoRA.
            with record_function("logprob_recompute"), torch.no_grad():
                with record_function("logprob_policy"):
                    self.compiled_model.eval()
                    old_logp, mask, _ = compute_token_logprobs(
                        self.compiled_model, full_ids, plen, rl.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        response_lengths=response_lengths,
                    )
                    self.compiled_model.train()

                with record_function("logprob_ref"):
                    if self.compiled_ref is not None:
                        ref_logp, _, _ = compute_token_logprobs(
                            self.compiled_ref, full_ids, plen, rl.temperature,
                            pad_token_id=self.tokenizer.pad_token_id,
                            response_lengths=response_lengths,
                        )
                    else:
                        with self.model.disable_adapter():
                            self.compiled_model.eval()
                            ref_logp, _, _ = compute_token_logprobs(
                                self.compiled_model, full_ids, plen, rl.temperature,
                                pad_token_id=self.tokenizer.pad_token_id,
                                response_lengths=response_lengths,
                            )
                            self.compiled_model.train()

            micro_batches.append({
                "full_ids": full_ids, "plen": plen, "adv": adv,
                "old_logp": old_logp, "ref_logp": ref_logp, "mask": mask,
                "token_count": mask.sum(dim=1).clamp(min=1.0),
                "responses": responses, "rewards": rewards,
                "pad_token_id": self.tokenizer.pad_token_id,
                "response_lengths": response_lengths,
            })
        # Store adaptive sampling stats for logging
        self._last_alive_rate = float(np.mean(alive_rates_step)) if alive_rates_step else 0.0
        self._last_spread_mean = float(np.mean([s[0] for s in spread_step])) if spread_step else 0.0
        self._last_spread_max = float(max(s[1] for s in spread_step)) if spread_step else 0.0

        return micro_batches

    def train_phase(self, micro_batches: list[dict]) -> dict:
        """Multi-epoch optimization. Thin wrapper around `rl_step`; lives here
        so Trainer owns the optimizer and `kl_ema` across steps.
        """
        metrics, self.kl_ema = rl_step(
            self.rl_cfg, micro_batches, self.compiled_model, self.ref_model,
            self.optimizer, self.stats, self.step, self.kl_ema,
        )
        return metrics
    
    def sync_weights(self) -> None:
        """Push trainer weights to the rollout worker.

        Dispatches to the backend configured by `cfg.weight_sync_method`:
          - "disk": LoRA adapter save + vLLM LoRARequest reload. Simple, slow,
                   always works. Requires `cfg.use_lora=True`.
          - "nccl": direct GPU->GPU broadcast via vLLM collective_rpc.
                   Fast, supports full FT. See vivace/utils/weight_sync.py.

        No-op only when there's no rollout worker (HF sampler shares weights
        by reference). Colocated mode still needs sync: vLLM runs in a separate
        EngineCore subprocess with its own memory, so trainer updates do NOT
        propagate by reference even on the same GPU.
        """
        # No rollout worker (HF sampler) — model shared by reference, nothing to sync.
        if self.rollout_worker is None:
            return

        # Both colocated and disaggregated need explicit sync — vLLM is a separate
        # subprocess in either case. (Earlier code returned early for colocated based
        # on a false "shared tensors" assumption; that left vLLM permanently on the
        # weights it loaded at __init__ and produced bit-identical pre/post eval.)
        if self.cfg.weight_sync_method == "disk":
            self._sync_weights_disk()
        elif self.cfg.weight_sync_method == "nccl":
            self._sync_weights_nccl()
        else:
            raise ValueError(
                f"unknown weight_sync_method {self.cfg.weight_sync_method!r}, "
                f"expected 'disk' or 'nccl'"
            )

    def _sync_weights_disk(self) -> None:
        """Disk-based LoRA adapter sync. Rank 0 saves; vLLM reloads via update_lora.

        Uses /dev/shm (RAM-backed tmpfs) by default — the adapter is overwritten
        every step and never needs to persist to real disk. Override by setting
        cfg.weight_sync_disk_path to a regular path.
        """
        if not self.cfg.use_lora:
            raise NotImplementedError(
                "full FT weight sync requires weight_sync_method='nccl'"
            )
        adapter_path = self._resolved_sync_disk_path
        os.makedirs(adapter_path, exist_ok=True)
        if is_main_process():
            self.model.save_pretrained(adapter_path)
        barrier()
        self.rollout_worker.update_lora(adapter_path)

    def _sync_weights_nccl(self) -> None:
        """NCCL broadcast from trainer to vLLM worker. Uses state from __init__.

        Receiver runs on a thread so sender can broadcast concurrently — without
        the thread, the RPC dispatch and the broadcast deadlock. See
        docs/weight_sync_approaches.md §"Concurrency model".
        """
        state = self._nccl_sync_state
        assert state is not None, (
            "NCCL sync state not initialized — check Trainer.__init__ for "
            "the weight_sync_method='nccl' setup block"
        )

        # For LoRA: fold A @ B into base so `named_parameters()` yields the full
        # current policy. Must unmerge in `finally` — if broadcast raises and we
        # skip unmerge, the next training step treats merged weights as base
        # and applies the adapter a second time on top.
        if self.cfg.use_lora:
            self.model.merge_adapter()
        try:
            def _trigger_receiver():
                self.rollout_worker.update_weights(state["specs"])

            receiver_thread = threading.Thread(target=_trigger_receiver, daemon=True)
            receiver_thread.start()

            sender_broadcast_loop(
                self.model,
                state["specs"],
                comm=state["comm"],
                src_rank=0,
                fusion_map=state["fusion_map"],
                fused_buffers=state["fused_buffers"],
            )
            receiver_thread.join()
        finally:
            if self.cfg.use_lora:
                self.model.unmerge_adapter()

    def train(self) -> None:
        """Main training loop: optional SFT warmup, baseline eval, N steps of
        (rollout -> train -> sync -> log/eval/ckpt), final eval, cleanup.
        """
        self.maybe_sft_warmup()

        # --- Baseline eval ---
        eval_data = self.env.load_split("eval")
        print("Evaluating baseline...")
        baseline_metrics, baseline_correct, baseline_incorrect = self._run_eval(eval_data)
        backend = baseline_metrics.get("eval_backend", "hf")
        print(f"Baseline accuracy: {baseline_metrics['accuracy_pct']:.1f}%  format: {baseline_metrics['format_rate_pct']:.1f}%  ({backend}, {baseline_metrics.get('eval_time_s', 0):.1f}s)")
        if is_main_process():
            log_metrics({"eval/" + k: v for k, v in baseline_metrics.items()}, step=0)
        self.stats.eval_steps.append(0)
        self.stats.eval_accuracy.append(baseline_metrics["accuracy_pct"])
        self.stats.eval_format_rate.append(baseline_metrics["format_rate_pct"])
        self.stats.eval_reward.append(baseline_metrics["avg_reward"])

        # --- Initial weight sync ---
        # vLLM starts with the base model only. Sync the initial LoRA adapter
        # so vLLM and the trainer use the same policy from step 0.
        self.sync_weights()

        # --- Profiler (optional) ---
        prof = create_profiler(self.profiling_cfg)
        if prof is not None:
            prof.__enter__()
            print(f"Profiler armed: will profile steps {self.profiling_cfg.start_step}-{self.profiling_cfg.end_step}")

        # --- Training loop ---
        self.kl_ema = self.cfg.rl.kl_target
        for step in range(self.cfg.num_steps):
            self.step = step

            with record_function(f"step_{step}"), Timer() as step_t:
                with record_function("rollout"), Timer() as rollout_t:
                    # In colocated mode: wake vLLM before rollout, sleep after.
                    # Skip wake_up on step 0 — vLLM starts awake after construction.
                    if self.rollout_worker and self.rollout_worker.colocated and step > 0:
                        with record_function("vllm_wake_up"):
                            self.rollout_worker.wake_up()
                    micro_batches = self.rollout_phase()
                    if self.rollout_worker and self.rollout_worker.colocated:
                        with record_function("vllm_sleep"):
                            self.rollout_worker.sleep()
                with record_function("train_phase"), Timer() as train_t:
                    metrics = self.train_phase(micro_batches)
                with record_function("weight_sync"):
                    self.sync_weights()

            self.scheduler.step()

            # --- Profiler step ---
            if prof is not None:
                prof.step()
                # Export after profiling window closes
                if step == self.profiling_cfg.end_step:
                    prof.__exit__(None, None, None)
                    export_and_summarize(prof, self.profiling_cfg, self.cfg.run_dir)
                    prof = None  # done profiling, no further overhead

            # --- Log everything in one stats.log() call ---
            with record_function("rollout_token_count"):
                rollout_tokens = int(sum(mb["mask"].sum().item() for mb in micro_batches))
            rollout_samples = int(sum(mb["mask"].shape[0] for mb in micro_batches))
            cur_lr = self.optimizer.param_groups[0]["lr"]

            self.stats.log(
                step=step,
                # Learning dynamics (from rl_step metrics dict)
                losses=metrics["loss"],
                rewards=metrics["reward"],
                kl_values=metrics["kl"],
                clip_fracs=metrics["clip_frac"],
                grad_norms=metrics["grad_norm"],
                entropies=metrics["entropy"],
                lengths=metrics["length_mean"],
                format_rates=metrics["format_rate"],
                # Diagnostics
                reward_std=metrics["reward_std"],
                reward_max=metrics["reward_max"],
                reward_min=metrics["reward_min"],
                length_mean=metrics["length_mean"],
                length_std=metrics["length_std"],
                length_max=metrics["length_max"],
                length_min=metrics["length_min"],
                advantage_std=metrics["advantage_std"],
                # Adaptive sampling (0 if disabled)
                alive_rates=self._last_alive_rate,
                # Performance
                rollout_time=rollout_t.dt,
                train_time=train_t.dt,
                step_time=step_t.dt,
                rollout_tokens=rollout_tokens,
                rollout_samples=rollout_samples,
                tokens_per_sec=throughput(rollout_tokens, rollout_t.dt),
                samples_per_sec=throughput(rollout_samples, rollout_t.dt),
                # LR
                lrs=cur_lr,
            )

            # --- Logging ---
            perf_info = {
                "tokens_per_sec": throughput(rollout_tokens, rollout_t.dt),
                "samples_per_sec": throughput(rollout_samples, rollout_t.dt),
                "step_time": step_t.dt,
                "rollout_time": rollout_t.dt,
                "train_time": train_t.dt,
            }

            if is_main_process():
                if step % self.cfg.log_interval == 0:
                    self.console.log(step, perf=perf_info, **metrics)
                    if self.cfg.rl.adaptive_sampling:
                        print(f"    alive={self._last_alive_rate:.1%} spread_mean={self._last_spread_mean:.3f} spread_max={self._last_spread_max:.3f}")
                    # Core metrics (flat — wandb top level)
                    log_metrics({
                        "loss": metrics["loss"],
                        "reward": metrics["reward"],
                        "kl": metrics["kl"],
                        "clip_frac": metrics["clip_frac"],
                        "grad_norm": metrics["grad_norm"],
                        "entropy": metrics["entropy"],
                        "format_rate": metrics["format_rate"],
                    }, step)
                    # Length distribution (grouped in wandb)
                    log_metrics({
                        "length/mean": metrics["length_mean"],
                        "length/std": metrics["length_std"],
                        "length/max": metrics["length_max"],
                        "length/min": metrics["length_min"],
                    }, step)
                    # Reward distribution
                    log_metrics({
                        "reward_dist/std": metrics["reward_std"],
                        "reward_dist/max": metrics["reward_max"],
                        "reward_dist/min": metrics["reward_min"],
                        "reward_dist/advantage_std": metrics["advantage_std"],
                    }, step)
                    # Adaptive sampling (logged even when disabled — 0 values)
                    if self.cfg.rl.adaptive_sampling:
                        log_metrics({
                            "sampling/alive_rate": self._last_alive_rate,
                            "sampling/spread_mean": self._last_spread_mean,
                            "sampling/spread_max": self._last_spread_max,
                        }, step)
                    # Performance
                    log_metrics({
                        "perf/rollout_time": rollout_t.dt,
                        "perf/train_time": train_t.dt,
                        "perf/step_time": step_t.dt,
                        "perf/tokens_per_sec": perf_info["tokens_per_sec"],
                        "perf/samples_per_sec": perf_info["samples_per_sec"],
                        "perf/lr": cur_lr,
                    }, step)

                # --- Periodic eval ---
                if step > 0 and step % self.cfg.eval_interval == 0:
                    # In colocated mode the rollout phase put vLLM to sleep; wake it
                    # before eval (which calls vllm.generate) and sleep again after so
                    # the next step's rollout starts in the expected (asleep) state.
                    if self.rollout_worker and self.rollout_worker.colocated:
                        self.rollout_worker.wake_up()
                    eval_metrics, _, _ = self._run_eval(eval_data)
                    if self.rollout_worker and self.rollout_worker.colocated:
                        self.rollout_worker.sleep()
                    print(f"  [eval] Step {step:04d} | accuracy={eval_metrics['accuracy_pct']:.1f}%  format={eval_metrics['format_rate_pct']:.1f}%  reward={eval_metrics['avg_reward']:.3f}  ({eval_metrics.get('eval_time_s', 0):.1f}s)")
                    log_metrics({"eval/" + k: v for k, v in eval_metrics.items()}, step)
                    # Store in stats for offline analysis
                    self.stats.eval_steps.append(step)
                    self.stats.eval_accuracy.append(eval_metrics["accuracy_pct"])
                    self.stats.eval_format_rate.append(eval_metrics["format_rate_pct"])
                    self.stats.eval_reward.append(eval_metrics["avg_reward"])

                # --- Periodic checkpoint (stub — save_checkpoint not implemented yet) ---
                # if step > 0 and step % self.cfg.checkpoint_interval == 0:
                #     save_checkpoint(self.model, self.optimizer, self.scheduler,
                #                     step, f"{self.cfg.run_dir}/ckpt-{step}")

        # --- Clean up profiler if training ended before profiling window ---
        if prof is not None:
            prof.__exit__(None, None, None)
            export_and_summarize(prof, self.profiling_cfg, self.cfg.run_dir)
            prof = None

        # --- Final eval ---
        print("\nFinal evaluation...")
        # Same wake/sleep dance as periodic eval (vLLM is asleep after the last step).
        if self.rollout_worker and self.rollout_worker.colocated:
            self.rollout_worker.wake_up()
        final_metrics, final_correct, final_incorrect = self._run_eval(eval_data)
        if self.rollout_worker and self.rollout_worker.colocated:
            self.rollout_worker.sleep()
        print(f"Final accuracy: {final_metrics['accuracy_pct']:.1f}%  format: {final_metrics['format_rate_pct']:.1f}%  ({final_metrics.get('eval_time_s', 0):.1f}s)")
        compare_metrics(baseline_metrics, final_metrics, label="RL training")

        self.stats.eval_steps.append(self.cfg.num_steps)
        self.stats.eval_accuracy.append(final_metrics["accuracy_pct"])
        self.stats.eval_format_rate.append(final_metrics["format_rate_pct"])
        self.stats.eval_reward.append(final_metrics["avg_reward"])

        if is_main_process():
            log_metrics({"eval/" + k: v for k, v in final_metrics.items()},
                        step=self.cfg.num_steps)
            finish_wandb()

        # --- Save stats + plots ---
        from datetime import datetime
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from vivace.utils.stats import plot_stats, plot_perf, plot_health, plot_wallclock

        os.makedirs(self.cfg.run_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = f"{self.cfg.rl.loss_type}_{self.cfg.rl.adv_type}_{self.cfg.num_steps}steps"
        stats_path = os.path.join(self.cfg.run_dir, f"stats_{run_label}_{timestamp}.pt")
        torch.save(self.stats, stats_path)
        print(f"\nStats saved to {stats_path}")

        # Save plots as PNGs (always works, no GUI needed)
        title = f"{self.stats.method} — {self.cfg.num_steps} steps"
        for plot_fn, name in [(plot_stats, "stats"), (plot_perf, "perf"), (plot_health, "health"), (plot_wallclock, "wallclock")]:
            plot_fn(self.stats, title=title)
            plot_path = os.path.join(self.cfg.run_dir, f"plot_{name}_{run_label}_{timestamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
        print(f"Plots saved to {self.cfg.run_dir}/plot_*_{timestamp}.png")

        print("To view stats interactively:")
        print(f"  stats = torch.load('{stats_path}', weights_only=False)")
        print(f"  from vivace.utils.stats import plot_stats, plot_perf, plot_health")
        print(f"  plot_stats(stats)")

        # --- Cleanup ---
        if self.rollout_worker is not None:
            del self.rollout_worker.llm  # shut down vLLM engine + subprocess
            self.rollout_worker = None
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
