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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

from vivace.algos.types import RLConfig, SFTConfig, validate_rl_config
from vivace.algos.policy_gradient import (
    compute_advantages, compute_token_logprobs, compute_kl, compute_loss, rl_step,
)
from vivace.envs.base import Env, Example
from vivace.envs import Env, make_env
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
from vivace.utils.distributed import is_main_process, barrier, init_distributed, reduce_metrics
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
    # Train env(s). Single string (existing configs) or list (mix corpora —
    # train splits concatenated, uniform sampling). All training envs must
    # share format_prompt + reward_fn (true within the math family).
    env_name: str | list[str] = "gsm8k"
    # Constructor kwargs forwarded to the train env(s). Lets yaml specify
    # reward_overrides etc. directly without baking a named preset in code:
    #   env_kwargs: {reward_overrides: {overlong_penalty: 1.0}}
    # When env_name is a list, env_kwargs must be a list[dict] of matching length.
    env_kwargs: dict | list[dict] | None = None
    # Eval env(s). One or many; each runs every eval cycle, logged under
    # `eval/{env_name}/...` in wandb. None → defaults to [env_name] when
    # env_name is a single string (backward-compat); must be set explicitly
    # when env_name is a list.
    eval_envs: list[str] | None = None
    # Per-eval-env constructor kwargs (parallel list to eval_envs). None →
    # no overrides. Most evals use vanilla envs, so this stays None for most
    # configs; set only when an eval env needs custom reward params.
    eval_env_kwargs: list[dict] | None = None
    algo_name: str = "grpo"

    # ----- execution mode -----
    mode: str = "colocated"          # "colocated" | "disaggregated"
    trainer_gpus: list[int] = field(default_factory=lambda: [0])
    rollout_gpus: list[int] = field(default_factory=lambda: [0])
    # Tensor parallelism within a single inference backend instance (vLLM today,
    # possibly SGLang later). v1 supports only 1.
    tensor_parallel_size: int = 1

    # ----- model dtype + compilation -----
    dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    # HF attention implementation: "flash_attention_2" (default — sm_80+, requires
    # flash-attn package), "sdpa" (torch built-in, falls back to math kernel with
    # 4D masks), or "eager" (slowest, materializes full attention scores tensor).
    # Override via --set attn_implementation=eager for FA2-off ablation runs.
    attn_implementation: str = "flash_attention_2"
    compile_model: bool = False      # torch.compile — can break generate(), off by default
    # Gradient checkpointing: trades ~30% step-time for ~50% peak activation memory.
    # Required for 1.5B + LoRA on a single 4090 with long sequences (math). Calls
    # HF's `gradient_checkpointing_enable(use_reentrant=False)` after PEFT wrap.
    gradient_checkpointing: bool = False

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
    eval_n: int = 500               # number of eval examples; <=0 = full eval set
    eval_batch_size: int = 32        # batch size for eval generation
    eval_use_vllm: bool = True       # use vLLM for eval when available (much faster)
    eval_gpus: list[int] = field(default_factory=list)  # separate eval GPUs (empty = use rollout worker)

    # ----- training-data filter -----
    # Drop training prompts whose tokenized length exceeds this. Bounds the
    # worst-case forward-pass logits allocation under prompt-length variance
    # (e.g. MATH has p99=707 vs p50=60 — one outlier in a microbatch tips the
    # GPU budget). Eval data is never filtered. None = no filter.
    max_prompt_tokens: int | None = None

    # ----- logging -----
    wandb_project: str = "vivace"
    wandb_run_name: str | None = None
    # Optional wandb group: runs that share a group cluster together in the
    # wandb UI (good for seed sweeps — set the group in the YAML once, vary
    # --seed and --run-dir per launch, all runs show up grouped).
    wandb_group: str | None = None
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

    # ----- DDP -----
    # Set True when not every trainable parameter participates in every forward
    # pass (e.g. LoRA on `embed_tokens` with conditional paths, or full-FT with
    # frozen layer ranges). False is correct for typical LoRA on attention/MLP
    # projections — every forward touches every adapter, and the extra graph
    # traversal `True` triggers slows each step. DDP itself warns at run time
    # if you have it set to True with no actual unused params.
    find_unused_parameters: bool = False

    # ----- misc -----
    seed: int = 42


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_free_port(rank_hint: int = 0) -> int:
    """Pick a free loopback port for one-shot rendezvous at startup.

    For multi-rank disagg, each trainer rank calls this independently and a
    naive port-0 grab can produce identical ports across ranks (the OS may
    reuse a just-released port). We try a rank-spaced range first
    (29600 + rank_hint*10 .. +9), falling back to OS allocation if that range
    is fully busy. This eliminates the cross-rank race in practice while
    keeping behavior identical for the single-rank case.
    """
    base = 29600 + rank_hint * 10
    for candidate in range(base, base + 10):
        try:
            s = socket.socket()
            s.bind(("localhost", candidate))
            s.close()
            return candidate
        except OSError:
            continue
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
    """Print model + training config summary after model load. Rank 0 only."""
    if not is_main_process():
        return

    # Unwrap DDP / torch.compile / peft to reach the underlying HF model's .config.
    inner = model
    for _ in range(4):
        if hasattr(inner, "_orig_mod"):              # torch.compile
            inner = inner._orig_mod
        elif hasattr(inner, "module") and not isinstance(inner, nn.ModuleList):
            inner = inner.module                      # DDP / FSDP
        elif hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
            inner = inner.base_model.model            # peft
        else:
            break
    hf_config = inner.config

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_mode = "LoRA" if cfg.use_lora else "full FT"

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
    print(f"  Params:          {n_total / 1e6:.1f}M total, {n_trainable / 1e6:.1f}M trainable ({train_mode})")
    print(f"  Vocab:           {hf_config.vocab_size}")
    print(f"  Hidden:          {hf_config.hidden_size}")
    print(f"  Layers:          {hf_config.num_hidden_layers}")
    if hasattr(hf_config, 'layer_types'):
        print(f"  Layer types:     {hf_config.layer_types}")
    train_str = cfg.env_name if isinstance(cfg.env_name, str) else "+".join(cfg.env_name)
    eval_str = ",".join(cfg.eval_envs) if cfg.eval_envs else train_str
    print(f"  Env:             train={train_str}  eval={eval_str}")
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
        self.train_envs, self.eval_envs = self._build_envs(cfg)
        # `self.env` = first train env. Used by rollout/eval paths that need
        # one env handle (format_prompt, reward_fn). Multi-train assumes these
        # are uniform across train_envs (true within the math family).
        self.env = self.train_envs[0]
        self.train_data = [
            ex for env in self.train_envs for ex in env.load_split("train")
        ]
        # Eval splits are loaded lazily per env on first use, then cached.
        self._eval_data_cache: dict[str, list] = {}

        self.rollout_worker = None

        _print_system_info()
        torch.backends.cuda.matmul.allow_tf32 = True

        # --- Distributed init (must come BEFORE any validation that uses world_size) ---
        rank, local_rank, world_size = init_distributed()
        self.rank, self.local_rank, self.world_size = rank, local_rank, world_size

        # --- Validation (now world_size is in scope) ---
        if cfg.tensor_parallel_size != 1:
            raise NotImplementedError("vLLM TP > 1 not supported yet")
        if len(cfg.trainer_gpus) != world_size:
            raise ValueError(
                f"len(trainer_gpus)={len(cfg.trainer_gpus)} must equal WORLD_SIZE={world_size}"
            )
        if cfg.mode == "colocated":
            if cfg.trainer_gpus != cfg.rollout_gpus:
                raise ValueError(
                    f"colocated mode requires trainer_gpus == rollout_gpus, got "
                    f"{cfg.trainer_gpus} vs {cfg.rollout_gpus}"
                )
        elif cfg.mode == "disaggregated":
            # 1:1 trainer:rollout pairing — each trainer rank gets its own vLLM
            # worker on a dedicated rollout GPU (gpu_ids=[rollout_gpus[local_rank]]
            # in the VLLMRolloutWorker constructor below). The NCCL weight-sync
            # path builds one comm per rank (each rank does its own
            # StatelessProcessGroup rendezvous on a separate port), so DDP across
            # multiple trainer ranks works without a global trainer↔vLLM group.
            if len(cfg.rollout_gpus) != world_size:
                raise ValueError(
                    f"disaggregated mode requires len(rollout_gpus)={len(cfg.rollout_gpus)} "
                    f"to equal WORLD_SIZE={world_size} (1:1 trainer:rollout pairing). "
                    f"Got trainer_gpus={cfg.trainer_gpus}, rollout_gpus={cfg.rollout_gpus}."
                )
            if set(cfg.trainer_gpus) & set(cfg.rollout_gpus):
                raise ValueError(
                    f"disaggregated mode requires disjoint trainer/rollout GPU sets, got "
                    f"trainer_gpus={cfg.trainer_gpus}, rollout_gpus={cfg.rollout_gpus}. "
                    f"Use mode='colocated' for shared-GPU layouts."
                )

        # --- Resolve devices ---
        # trainer_gpus / rollout_gpus are the full GPU pools (per cfg). For DDP,
        # each rank picks its own primary by LOCAL_RANK from these pools.
        # trainer_devices / rollout_devices are kept only for the startup banner
        # (_print_training_info) — not used for any GPU op.
        if torch.cuda.is_available():
            self.trainer_devices = [torch.device(f"cuda:{g}") for g in cfg.trainer_gpus]
            self.rollout_devices = [torch.device(f"cuda:{g}") for g in cfg.rollout_gpus]
            self.device = torch.device(f"cuda:{cfg.trainer_gpus[local_rank]}")
        else:
            self.trainer_devices = [torch.device("cpu")]
            self.rollout_devices = [torch.device("cpu")]
            self.device = self.trainer_devices[0]

        self.seed = cfg.seed + rank
        _set_seed(self.seed)

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # --- Filter long-tail training prompts (memory-bounding, opt-in) ---
        if cfg.max_prompt_tokens is not None:
            before = len(self.train_data)
            cap = cfg.max_prompt_tokens
            self.train_data = [
                ex for ex in self.train_data
                if len(self.tokenizer.encode(self.env.format_prompt(ex))) <= cap
            ]
            after = len(self.train_data)
            if is_main_process():
                pct = 100.0 * (before - after) / before if before else 0.0
                print(f"  Train filter:    max_prompt_tokens={cap}  "
                      f"{before} → {after}  ({pct:.2f}% dropped)")

        # --- Model ---
        model_dtype = _resolve_dtype(cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=model_dtype,
            low_cpu_mem_usage=True, trust_remote_code=True,
            attn_implementation=cfg.attn_implementation,
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

        # Gradient checkpointing (opt-in). Two pieces matter:
        #  - enable_input_require_grads(): under PEFT/LoRA the embedding params
        #    are frozen, so gradient_checkpointing's recomputed forward has no
        #    requires_grad=True input and can't wire backward. The hook on
        #    embed_tokens flips its output's requires_grad to True at forward
        #    time, repairing the chain. Idempotent on full-FT.
        #  - use_reentrant=False: the modern torch.utils.checkpoint path.
        #    The legacy reentrant path is deprecated and breaks with DDP.
        # use_cache=False is already set above (line ~354), which gc requires.
        if cfg.gradient_checkpointing:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Keep a handle to the peft (or bare HF) model so we can call peft methods
        # like merge_adapter / unmerge_adapter / disable_adapter / save_pretrained
        # without hitting DDP's __getattr__ wall after wrapping.
        self._inner_model = self.model

        if world_size > 1:
            # device_ids must match the device the model actually lives on
            # (self.device, set above from cfg.trainer_gpus[local_rank]).
            # Using bare local_rank here breaks whenever trainer_gpus is not
            # 0..N-1 in order — e.g., trainer_gpus=[2,3] with no outer
            # CUDA_VISIBLE_DEVICES mask. NCCL then asserts with
            # "Tensor on cuda:X but backend constrained to cuda:Y".
            self.model = DDP(self.model, device_ids=[self.device.index],
                             find_unused_parameters=cfg.find_unused_parameters)

        # --- Compile (optional) ---
        # Use compiled versions for forward passes (rl_step, log-prob recomputation),
        # raw self.model for generate() — torch.compile and generate() don't mix well.
        # For LoRA: compiled_ref is None (use compiled_model with disable_adapter).
        self.compiled_model = torch.compile(self.model, dynamic=True) if cfg.compile_model else self.model
        if self.ref_model is not None:
            self.compiled_ref = torch.compile(self.ref_model, dynamic=True) if cfg.compile_model else self.ref_model
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
                gpu_ids=[cfg.rollout_gpus[local_rank]],
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
        # Two modes: plain cosine, or cosine→linear-ramp-restart→cosine. The restart
        # uses `warmup_steps` for the ramp length, matching the initial warmup.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.rl.lr,
            betas=(cfg.rl.adam_beta1, cfg.rl.adam_beta2), eps=cfg.rl.adam_eps,
        )
        warmup_steps = cfg.rl.warmup_steps
        post_warmup = max(cfg.num_steps - warmup_steps, 1)
        eta_min = cfg.rl.lr * cfg.rl.eta_min_ratio
        scheds = [torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps)]
        milestones = [warmup_steps]
        if cfg.rl.lr_restart:
            # Restart fires at the midpoint of the whole run (step num_steps // 2),
            # so first cosine runs from end-of-warmup to that midpoint.
            restart_step = cfg.num_steps // 2
            cosine1_steps = max(restart_step - warmup_steps, 1)
            scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cosine1_steps, eta_min=eta_min))
            # LinearLR factors multiply against base lr (cfg.rl.lr): start at eta_min, end at peak.
            scheds.append(torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=cfg.rl.eta_min_ratio,
                end_factor=1.0, total_iters=warmup_steps))
            milestones.append(restart_step)
            milestones.append(restart_step + warmup_steps)
            scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(cfg.num_steps - restart_step - warmup_steps, 1),
                eta_min=eta_min))
        else:
            scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=post_warmup, eta_min=eta_min))
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=scheds, milestones=milestones)

        # EMA of per-step KL(policy ∥ reference) — smooths the noisy batch-level
        # `kl` so the cumulative-drift trend is visible. alpha=0.05 → ~20-step
        # half-life. Logged as `kl_to_ref_ema`.
        self.kl_to_ref_ema: float | None = None

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
        if is_main_process():
            # Default run name to run_dir basename; else wandb generates a random slug.
            run_name = cfg.wandb_run_name or os.path.basename(cfg.run_dir.rstrip("/"))
            init_wandb(cfg, project=cfg.wandb_project, run_name=run_name,
                       group=cfg.wandb_group)

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

            host, port = "localhost", _find_free_port(rank_hint=local_rank)
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

        # IPC sync state. Same-GPU zero-copy via CUDA IPC handles.
        self._ipc_sync_state = None
        if cfg.weight_sync_method == "ipc" and self.rollout_worker is not None:
            if cfg.mode != "colocated":
                raise ValueError(
                    "weight_sync_method='ipc' is only meaningful in colocated mode. "
                    "For disaggregated, use 'nccl' (faster) or 'disk'."
                )
            from vivace.utils.weight_sync import build_param_specs, allocate_fused_buffers
            from vivace.utils.ipc_sync import pack_ipc_handles

            specs, fusion_map = build_param_specs(self.model, fuse=True)
            fused_buffers = allocate_fused_buffers(self.model, specs, self.device)
            # Build IPC handles ONCE: storages are stable across the run because
            # peft merge/unmerge mutate base.weight.data in place and fused buffers
            # are preallocated. vLLM keeps these handles open for the whole run.
            handles = pack_ipc_handles(self.model, specs, fusion_map, fused_buffers)
            self.rollout_worker.init_ipc_sync(handles, specs)
            self._ipc_sync_state = {
                "specs": specs,
                "fusion_map": fusion_map,
                "fused_buffers": fused_buffers,
            }
            print(f"[trainer init] IPC weight sync ready: {len(specs)} params")

        _print_training_info(cfg, self.model, self.trainer_devices, self.rollout_devices)
        if cfg.weight_sync_method == "disk":
            print(f"  [disk-sync] adapter path: {self._resolved_sync_disk_path}")

    @staticmethod
    def _build_envs(cfg: "TrainerConfig") -> tuple[list[Env], list[tuple[str, Env]]]:
        """Resolve cfg.env_name + cfg.eval_envs into concrete envs.

        Returns (train_envs, eval_envs) where eval_envs is a list of
        (name, env) pairs so the eval loop can prefix wandb keys per env.
        """
        train_names = [cfg.env_name] if isinstance(cfg.env_name, str) else list(cfg.env_name)
        if not train_names:
            raise ValueError("env_name must be a non-empty string or list[str]")

        # Per-train-env kwargs. dict → broadcast to all; list[dict] → must match length.
        if cfg.env_kwargs is None:
            train_kwargs: list[dict] = [{} for _ in train_names]
        elif isinstance(cfg.env_kwargs, dict):
            train_kwargs = [cfg.env_kwargs for _ in train_names]
        else:
            if len(cfg.env_kwargs) != len(train_names):
                raise ValueError(
                    f"env_kwargs list length ({len(cfg.env_kwargs)}) must match "
                    f"env_name list length ({len(train_names)})"
                )
            train_kwargs = list(cfg.env_kwargs)
        train_envs = [make_env(n, **kw) for n, kw in zip(train_names, train_kwargs)]

        if cfg.eval_envs is not None:
            eval_names = list(cfg.eval_envs)
        elif len(train_names) == 1:
            eval_names = train_names                  # backward-compat for str env_name
        else:
            raise ValueError(
                "eval_envs must be set explicitly when env_name is a list "
                "(no canonical default for multi-train mixes)"
            )
        if cfg.eval_env_kwargs is None:
            eval_kwargs: list[dict] = [{} for _ in eval_names]
        else:
            if len(cfg.eval_env_kwargs) != len(eval_names):
                raise ValueError(
                    f"eval_env_kwargs list length ({len(cfg.eval_env_kwargs)}) must "
                    f"match eval_envs length ({len(eval_names)})"
                )
            eval_kwargs = list(cfg.eval_env_kwargs)
        eval_envs = [(n, make_env(n, **kw)) for n, kw in zip(eval_names, eval_kwargs)]
        return train_envs, eval_envs

    def _eval_data_for(self, env_name: str, env: Env) -> list:
        """Lazy-load and cache an env's eval split. Loaded once per process."""
        if env_name not in self._eval_data_cache:
            self._eval_data_cache[env_name] = env.load_split("eval")
        return self._eval_data_cache[env_name]

    def _run_eval(self, env: Env, eval_data: list, label: str = "") -> tuple[dict, list, list]:
        """Run evaluation using the best available backend.

        `label` ("baseline" / "step_NNNN" / "final" / etc.) is attached to the
        returned metrics dict and used by `_save_eval_samples` for the JSON
        filename, so multiple eval moments can be persisted side by side.
        """
        n = len(eval_data) if self.cfg.eval_n <= 0 else self.cfg.eval_n
        # Cached deterministic shuffle (cfg.seed, identical across ranks): keeps
        # contiguous slices length/difficulty-balanced on sorted datasets.
        if getattr(self, "_eval_shuffle_idxs", None) is None \
                or len(self._eval_shuffle_idxs) != len(eval_data):
            self._eval_shuffle_idxs = np.random.RandomState(self.cfg.seed).permutation(len(eval_data))

        subset = [eval_data[i] for i in self._eval_shuffle_idxs[:n]]
        data_per_rank = (n + self.world_size - 1) // self.world_size
        offset = self.rank * data_per_rank
        local = subset[offset : offset + data_per_rank]
        local_metrics, local_correct, local_incorrect = evaluate_model(
            self.model, self.tokenizer, local, env,
            n=-1, # already sliced
            batch_size=self.cfg.eval_batch_size,
            max_new_tokens=self.cfg.rl.max_new_tokens,
            device=str(self.device),
            vllm_worker=self.eval_worker,
        )

        # *_sum fields are aggregation plumbing only; never user-facing.
        _raw_keys = ("reward_sum", "token_sum", "char_sum")
        if self.world_size == 1:
            local_metrics["label"] = label
            for k in _raw_keys:
                local_metrics.pop(k, None)
            return local_metrics, local_correct, local_incorrect

        # Cross-rank aggregation (every rank participates in the all_reduce)
        raw = {k: local_metrics[k] for k in
            ["n", "n_correct", "n_format_ok", "n_capped",
                "reward_sum", "token_sum", "char_sum", "eval_time_s"]}
        agg = reduce_metrics(raw, {
            "n": "sum", "n_correct": "sum", "n_format_ok": "sum",
            "n_capped": "sum", "reward_sum": "sum",
            "token_sum": "sum", "char_sum": "sum",
            "eval_time_s": "max",
        })
        metrics = {
            "n":               int(agg["n"]),
            "accuracy_pct":    100.0 * agg["n_correct"]   / agg["n"],
            "format_rate_pct": 100.0 * agg["n_format_ok"] / agg["n"],
            "cap_rate_pct":    100.0 * agg["n_capped"]    / agg["n"],
            "avg_reward":      agg["reward_sum"] / agg["n"],
            "avg_length_tokens": agg["token_sum"] / agg["n"],
            "avg_length_chars":  agg["char_sum"]  / agg["n"],
            "n_correct":       int(agg["n_correct"]),
            "n_format_ok":     int(agg["n_format_ok"]),
            "n_capped":        int(agg["n_capped"]),
            "eval_time_s":     agg["eval_time_s"],
            "eval_backend":    local_metrics["eval_backend"],
            "label":           label,
        }

        # Gather per-sample lists for inspection JSON
        gathered_correct = [None] * self.world_size
        gathered_incorrect = [None] * self.world_size
        dist.all_gather_object(gathered_correct, local_correct)
        dist.all_gather_object(gathered_incorrect, local_incorrect)
        correct = [x for sub in gathered_correct for x in sub]
        incorrect = [x for sub in gathered_incorrect for x in sub]

        return metrics, correct, incorrect

    def _run_eval_all(self, label: str) -> dict[str, tuple[dict, list, list]]:
        """Run `_run_eval` on every cfg.eval_envs. Insertion order = config order;
        the first env is the "primary" (the one that drives stats updates and
        the baseline-vs-final comparison print). All ranks must call this so
        every `_run_eval` collective fires on every rank."""
        out: dict[str, tuple[dict, list, list]] = {}
        for env_name, env in self.eval_envs:
            data = self._eval_data_for(env_name, env)
            out[env_name] = self._run_eval(env, data, label=f"{label}_{env_name}")
        return out

    def _save_eval_samples(self, label: str, correct: list, incorrect: list) -> None:
        """Persist eval samples as eval_samples_{label}.json in the run dir.
        Rank-0-only; no-op if label is empty (caller opted out)."""
        if not is_main_process() or not label:
            return
        import json
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        path = os.path.join(self.cfg.run_dir, f"eval_samples_{label}.json")
        with open(path, "w") as f:
            json.dump({"correct": correct, "incorrect": incorrect}, f, indent=2, default=str)
        print(f"Eval samples saved to {path}")

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
        # Per-component reward accumulator across kept responses for wandb logging.
        component_accum: dict[str, list[float]] = {}
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

            # Compute rewards for ALL responses (needed for adaptive sampling spread).
            # `reward_breakdown` also returns per-component values for wandb logging
            # (envs that don't support it return an empty dict — backward compatible).
            with record_function("reward"):
                tok_counts = response_lengths.tolist() if response_lengths is not None else [None] * len(responses)
                rewards_list, reward_components = self.env.reward_breakdown(
                    responses, examples,
                    response_token_counts=tok_counts,
                    max_new_tokens=rl.max_new_tokens,
                )
                rewards = torch.tensor(rewards_list, device=self.device, dtype=torch.float32)

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
                reward_components = {name: [vals[k] for k in keep]
                                     for name, vals in reward_components.items()}

            for name, vals in reward_components.items():
                component_accum.setdefault(name, []).extend(vals)

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
                        with self._inner_model.disable_adapter():
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
        # Fraction of training rollouts that hit max_new_tokens — leading indicator
        # of length-gaming or policy regression (lagging eval-time cap_rate_pct).
        all_lens = torch.cat([mb["response_lengths"] for mb in micro_batches])
        self._last_cap_rate = (all_lens >= self.cfg.rl.max_new_tokens).float().mean().item()
        # Per-component reward means (kept responses, this step). Empty for envs
        # that don't override Env.reward_breakdown — wandb just won't log anything.
        self._last_reward_components = {name: float(np.mean(vals))
                                        for name, vals in component_accum.items() if vals}

        return micro_batches

    def train_phase(self, micro_batches: list[dict]) -> dict:
        """Multi-epoch optimization. Thin wrapper around `rl_step`; lives here
        so Trainer owns the optimizer and `kl_ema` across steps.

        Also reduces learning metrics across DDP ranks before returning so
        wandb / stats see globally-correct numbers, not rank-0-local. Stds use
        Σx²-based sufficient stats because mean-of-stds underestimates spread
        (see reduce_metrics docstring); ratios like clip_frac and format_rate
        are recomputed from globally-summed counts. grad_norm is already global
        — DDP all-reduces gradients in backward, so clip_grad_norm matches on
        every rank.
        """
        metrics, self.kl_ema = rl_step(
            self.rl_cfg, micro_batches, self.compiled_model, self.ref_model,
            self.optimizer, self.stats, self.step, self.kl_ema,
        )
        return self._reduce_learning_metrics(metrics)

    @staticmethod
    def _reduce_learning_metrics(metrics: dict) -> dict:
        """Cross-rank reduction of training-loop metrics. No-op at world_size==1."""
        ops = {
            "loss": "mean", "reward": "mean", "kl": "mean", "entropy": "mean",
            "length_mean": "mean",
            "reward_max": "max", "length_max": "max",
            "reward_min": "min", "length_min": "min",
            "_reward_sum": "sum", "_reward_sumsq": "sum",
            "_length_sum": "sum", "_length_sumsq": "sum",
            "_advantage_sum": "sum", "_advantage_sumsq": "sum",
            "_n": "sum", "_format_ok": "sum",
            "_clip_count": "sum", "_clip_tokens": "sum",
        }
        agg = reduce_metrics(metrics, ops)

        def _global_std(s: float, sq: float, n: float) -> float:
            # Unbiased (ddof=1) to match torch.std default → single-rank values
            # stay numerically identical to the pre-reduction implementation.
            if n < 2:
                return 0.0
            var = (sq - s * s / n) / (n - 1)
            return float(var ** 0.5) if var > 0 else 0.0

        n_g = max(int(agg["_n"]), 1)
        out = dict(metrics)
        # Mean / extrema reductions overwrite the per-rank values.
        for k in ("loss", "reward", "kl", "entropy", "length_mean",
                  "reward_max", "reward_min", "length_max", "length_min"):
            out[k] = agg[k]
        out["reward_std"] = _global_std(agg["_reward_sum"], agg["_reward_sumsq"], n_g)
        out["length_std"] = _global_std(agg["_length_sum"], agg["_length_sumsq"], n_g)
        out["advantage_std"] = _global_std(agg["_advantage_sum"], agg["_advantage_sumsq"], n_g)
        out["format_rate"] = agg["_format_ok"] / n_g
        out["clip_frac"] = (agg["_clip_count"] / agg["_clip_tokens"]) if agg["_clip_tokens"] > 0 else 0.0
        # grad_norm passes through unreduced (already global under DDP gradient sync).
        for k in ("_n", "_format_ok",
                  "_reward_sum", "_reward_sumsq",
                  "_length_sum", "_length_sumsq",
                  "_advantage_sum", "_advantage_sumsq",
                  "_clip_count", "_clip_tokens"):
            out.pop(k, None)
        return out
    
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
        elif self.cfg.weight_sync_method == "ipc":
            self._sync_weights_ipc()
        else:
            raise ValueError(
                f"unknown weight_sync_method {self.cfg.weight_sync_method!r}, "
                f"expected 'disk', 'nccl', or 'ipc'"
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
            self._inner_model.save_pretrained(adapter_path)
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
            self._inner_model.merge_adapter()
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
                self._inner_model.unmerge_adapter()

    def _sync_weights_ipc(self) -> None:
        """Same-GPU CUDA IPC weight sync. Trainer fills stable buffers; vLLM's
        worker copies from aliased views via collective_rpc.

        Init in `__init__` builds + pushes IPC handles once (storages stable).
        Per step: merge → fill fused buffers → trainer-side cuda.synchronize →
        vLLM copy_ via aliased views (worker-side cuda.synchronize too) →
        unmerge in `finally`.
        """
        from vivace.utils.ipc_sync import fill_fused_buffers

        state = self._ipc_sync_state
        assert state is not None, (
            "IPC sync state not initialized — check Trainer.__init__ for "
            "the weight_sync_method='ipc' setup block"
        )

        if self.cfg.use_lora:
            self._inner_model.merge_adapter()
        try:
            # Trainer-side: fill fused buffers; non-fused params are already
            # aliased directly via IPC, no copy needed on this side.
            fill_fused_buffers(
                self.model, state["specs"], state["fusion_map"], state["fused_buffers"]
            )
            # Make sure the merge_adapter writes + fused-buffer cats are
            # globally visible before vLLM reads them through its alias.
            torch.cuda.synchronize()
            self.rollout_worker.update_weights_via_ipc()
        finally:
            if self.cfg.use_lora:
                self._inner_model.unmerge_adapter()

    def train(self) -> None:
        """Main training loop: optional SFT warmup, baseline eval, N steps of
        (rollout -> train -> sync -> log/eval/ckpt), final eval, cleanup.
        """
        self.maybe_sft_warmup()

        # --- Baseline eval (every env in cfg.eval_envs; primary = first) ---
        baseline_results = self._run_eval_all("baseline")
        primary_name = self.eval_envs[0][0]
        baseline_metrics = baseline_results[primary_name][0]
        for env_name, (m, correct, incorrect) in baseline_results.items():
            self._save_eval_samples(f"baseline_{env_name}", correct, incorrect)

        if is_main_process():
            print("Evaluating baseline...")
            for env_name, (m, _, _) in baseline_results.items():
                backend = m.get("eval_backend", "hf")
                print(
                    f"  [{env_name}] accuracy={m['accuracy_pct']:.1f}%  "
                    f"format={m['format_rate_pct']:.1f}%  "
                    f"capped={m.get('cap_rate_pct', 0):.1f}%  "
                    f"({backend}, {m.get('eval_time_s', 0):.1f}s)"
                )
                log_metrics({f"eval/{env_name}/{k}": v for k, v in m.items()}, step=0)
            self.stats.eval_steps.append(0)
            self.stats.eval_accuracy.append(baseline_metrics["accuracy_pct"])
            self.stats.eval_format_rate.append(baseline_metrics["format_rate_pct"])
            self.stats.eval_reward.append(baseline_metrics["avg_reward"])
        barrier()

        # --- Initial weight sync ---
        # vLLM starts with the base model only. Sync the initial LoRA adapter
        # so vLLM and the trainer use the same policy from step 0.
        self.sync_weights()

        # --- Profiler (optional) ---
        prof = create_profiler(self.profiling_cfg)
        if prof is not None:
            prof.__enter__()
            if is_main_process():
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
                        # Drain trainer cache before sleep so vLLM's `freed_bytes >= 0`
                        # invariant sees a stable baseline. See colocated allocator note below.
                        gc.collect()
                        torch.cuda.empty_cache()
                        with record_function("vllm_sleep"):
                            self.rollout_worker.sleep()
                        # Also after sleep — clears any blocks freed by sleep itself,
                        # so the next wake_up starts on a clean trainer pool.
                        gc.collect()
                        torch.cuda.empty_cache()
                with record_function("train_phase"), Timer() as train_t:
                    metrics = self.train_phase(micro_batches)
                with record_function("weight_sync"):
                    self.sync_weights()
                # Release the trainer's cached-but-unused CUDA blocks each step.
                # Variable T (per-step prompt + response length) creates blocks
                # of many shapes the allocator can't reuse — pool drifts upward
                # across steps. In colocated mode this squeezes vLLM and trips
                # its sleep-time `freed_bytes >= 0` assertion; in disaggregated
                # mode it OOMs the trainer outright (~step 150-200 on 1.5B + MATH).
                # gc.collect() before empty_cache() is necessary — Python ref
                # cycles in rl_step's closures / stats accumulators keep tensors
                # alive past their scope; empty_cache alone only frees cached
                # blocks, not blocks still referenced through dead cycles.
                # No-op cost when there's nothing to release.
                if self.rollout_worker is not None:
                    gc.collect()
                    torch.cuda.empty_cache()

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

            # Throughput is per-rank work happening concurrently → SUM across ranks
            # for system-level numbers. Wall-clock times are bottlenecked by the
            # slowest rank → MAX. No-op in single-rank runs.
            perf_raw = {
                "rollout_tokens": rollout_tokens,
                "rollout_samples": rollout_samples,
                "tokens_per_sec": throughput(rollout_tokens, rollout_t.dt),
                "samples_per_sec": throughput(rollout_samples, rollout_t.dt),
                "rollout_time": rollout_t.dt,
                "train_time": train_t.dt,
                "step_time": step_t.dt,
                "cap_rate": self._last_cap_rate,
                "alive_rate": self._last_alive_rate,
                "spread_mean": self._last_spread_mean,
                "spread_max": self._last_spread_max,
            }
            perf = reduce_metrics(perf_raw, {
                "rollout_tokens": "sum",
                "rollout_samples": "sum",
                "tokens_per_sec": "sum",
                "samples_per_sec": "sum",
                "rollout_time": "max",
                "train_time": "max",
                "step_time": "max",
                "cap_rate": "mean",
                "alive_rate": "mean",
                "spread_mean": "mean",
                "spread_max": "max",
            })

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
                cap_rates=perf["cap_rate"],
                advantage_std=metrics["advantage_std"],
                # Adaptive sampling (0 if disabled)
                alive_rates=perf["alive_rate"],
                # Performance (cross-rank reduced — see perf_raw above)
                rollout_time=perf["rollout_time"],
                train_time=perf["train_time"],
                step_time=perf["step_time"],
                rollout_tokens=perf["rollout_tokens"],
                rollout_samples=perf["rollout_samples"],
                tokens_per_sec=perf["tokens_per_sec"],
                samples_per_sec=perf["samples_per_sec"],
                # LR
                lrs=cur_lr,
            )

            # --- Logging ---
            perf_info = {
                "tokens_per_sec": perf["tokens_per_sec"],
                "samples_per_sec": perf["samples_per_sec"],
                "step_time": perf["step_time"],
                "rollout_time": perf["rollout_time"],
                "train_time": perf["train_time"],
            }

            if is_main_process():
                if step % self.cfg.log_interval == 0:
                    self.console.log(step, perf=perf_info, **metrics)
                    aux = [f"capped={perf['cap_rate']:.1%}"]
                    if self.cfg.rl.adaptive_sampling:
                        aux = [f"alive={perf['alive_rate']:.1%}",
                               f"spread_mean={perf['spread_mean']:.3f}",
                               f"spread_max={perf['spread_max']:.3f}"] + aux
                    # Memory-leak diagnostic (disagg OOMs at step ~170; track which
                    # category grows). Tear down once the leak is found.
                    # Peak is captured since last reset_peak_memory_stats — call
                    # it AFTER reading so the next step's peak measurement is clean.
                    if torch.cuda.is_available():
                        mem_alloc = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        mem_peak = torch.cuda.max_memory_allocated() / 1e9
                        aux.append(
                            f"mem_alloc={mem_alloc:.2f}G "
                            f"mem_res={mem_reserved:.2f}G "
                            f"mem_peak={mem_peak:.2f}G"
                        )
                        torch.cuda.reset_peak_memory_stats()
                    print("    " + " ".join(aux))
                    # Track smoothed KL drift from reference.
                    cur_kl = metrics["kl"]
                    if self.kl_to_ref_ema is None:
                        self.kl_to_ref_ema = cur_kl
                    else:
                        self.kl_to_ref_ema = 0.95 * self.kl_to_ref_ema + 0.05 * cur_kl
                    # Core metrics (flat — wandb top level)
                    log_metrics({
                        "loss": metrics["loss"],
                        "reward": metrics["reward"],
                        "kl": metrics["kl"],
                        "kl_to_ref_ema": self.kl_to_ref_ema,
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
                        "length/cap_rate": perf["cap_rate"],
                    }, step)
                    # Reward distribution
                    log_metrics({
                        "reward_dist/std": metrics["reward_std"],
                        "reward_dist/max": metrics["reward_max"],
                        "reward_dist/min": metrics["reward_min"],
                        "reward_dist/advantage_std": metrics["advantage_std"],
                    }, step)
                    # Per-component reward means (empty for envs without breakdown)
                    if self._last_reward_components:
                        log_metrics({f"reward_components/{name}": v
                                     for name, v in self._last_reward_components.items()}, step)
                    # Adaptive sampling (logged even when disabled — 0 values)
                    if self.cfg.rl.adaptive_sampling:
                        log_metrics({
                            "sampling/alive_rate": perf["alive_rate"],
                            "sampling/spread_mean": perf["spread_mean"],
                            "sampling/spread_max": perf["spread_max"],
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

            # --- Periodic eval (out of is_main_process: all ranks must enter
            # _run_eval to participate in its all_reduce) ---
            if step > 0 and step % self.cfg.eval_interval == 0:
                if self.rollout_worker and self.rollout_worker.colocated:
                    self.rollout_worker.wake_up()
                eval_results = self._run_eval_all(f"step_{step:04d}")
                if self.rollout_worker and self.rollout_worker.colocated:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.rollout_worker.sleep()
                if is_main_process():
                    primary_metrics = eval_results[self.eval_envs[0][0]][0]
                    for env_name, (m, _, _) in eval_results.items():
                        print(
                            f"  [eval/{env_name}] Step {step:04d} | "
                            f"accuracy={m['accuracy_pct']:.1f}%  "
                            f"format={m['format_rate_pct']:.1f}%  "
                            f"capped={m.get('cap_rate_pct', 0):.1f}%  "
                            f"reward={m['avg_reward']:.3f}  "
                            f"({m.get('eval_time_s', 0):.1f}s)"
                        )
                        log_metrics({f"eval/{env_name}/{k}": v for k, v in m.items()}, step)
                    self.stats.eval_steps.append(step)
                    self.stats.eval_accuracy.append(primary_metrics["accuracy_pct"])
                    self.stats.eval_format_rate.append(primary_metrics["format_rate_pct"])
                    self.stats.eval_reward.append(primary_metrics["avg_reward"])

            # --- Periodic checkpoint (stub — save_checkpoint not implemented yet) ---
            # if step > 0 and step % self.cfg.checkpoint_interval == 0:
            #     save_checkpoint(self.model, self.optimizer, self.scheduler,
            #                     step, f"{self.cfg.run_dir}/ckpt-{step}")

            # Non-rank-0 ranks wait here so they don't race ahead during rank 0's
            # log/checkpoint work above.
            barrier()

        # --- Clean up profiler if training ended before profiling window ---
        if prof is not None:
            prof.__exit__(None, None, None)
            export_and_summarize(prof, self.profiling_cfg, self.cfg.run_dir)
            prof = None

        # --- Final eval (all ranks participate in _run_eval's all_reduce) ---
        if is_main_process():
            print("\nFinal evaluation...")
        if self.rollout_worker and self.rollout_worker.colocated:
            self.rollout_worker.wake_up()
        final_results = self._run_eval_all("final")
        if self.rollout_worker and self.rollout_worker.colocated:
            gc.collect()
            torch.cuda.empty_cache()
            self.rollout_worker.sleep()
        primary_name = self.eval_envs[0][0]
        final_metrics = final_results[primary_name][0]
        if is_main_process():
            for env_name, (m, _, _) in final_results.items():
                print(
                    f"  [final/{env_name}] accuracy={m['accuracy_pct']:.1f}%  "
                    f"format={m['format_rate_pct']:.1f}%  "
                    f"capped={m.get('cap_rate_pct', 0):.1f}%  "
                    f"({m.get('eval_time_s', 0):.1f}s)"
                )
                log_metrics(
                    {f"eval/{env_name}/{k}": v for k, v in m.items()},
                    step=self.cfg.num_steps,
                )
            # One Before/After table per eval env so a multi-env run shows the
            # full picture, not just the primary's delta.
            for env_name, (final_metrics_env, _, _) in final_results.items():
                baseline_metrics_env = baseline_results[env_name][0]
                compare_metrics(baseline_metrics_env, final_metrics_env,
                                label=f"RL training [{env_name}]")
            self.stats.eval_steps.append(self.cfg.num_steps)
            self.stats.eval_accuracy.append(final_metrics["accuracy_pct"])
            self.stats.eval_format_rate.append(final_metrics["format_rate_pct"])
            self.stats.eval_reward.append(final_metrics["avg_reward"])
            finish_wandb()
        for env_name, (_, correct, incorrect) in final_results.items():
            self._save_eval_samples(f"final_{env_name}", correct, incorrect)
        barrier()

        # --- Save stats + plots (rank 0 only) ---
        if is_main_process():
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

        if dist.is_initialized():
            dist.destroy_process_group()
