"""Main training orchestrator.

Wires the rollout backend, the algos/policy_gradient module, the eval
runner, the logging, and the checkpoint saver into one loop. The loop
itself is what's interesting; everything else is delegated.

==============================================================================
WHAT NEEDS TO BE IMPLEMENTED
==============================================================================

The class skeleton, the dataclass, and the train() outer loop structure
are provided. The bodies of __init__, rollout_phase, train_phase,
sync_weights, and maybe_sft_warmup are stubs.

Suggested order:
  1. __init__ — single GPU, no DDP, HF sampler, no LoRA
  2. rollout_phase — call hf_sampler.sample_responses, build micro-batches
  3. train_phase — call algos/policy_gradient.rl_step
  4. train() body — wire it together end-to-end on a single GPU
  5. LoRA path in __init__ + reference model via disable_adapter
  6. DDP wrap in __init__ (graduate to multi-GPU)
  7. sync_weights — disk-based LoRA sync via vllm_worker.update_lora
  8. maybe_sft_warmup
  9. Disaggregated mode + in-place weight sync (much later)
"""

from __future__ import annotations

import gc
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from vivace.algos.types import RLConfig, SFTConfig, validate_rl_config
from vivace.algos.policy_gradient import (
    compute_advantages, compute_token_logprobs, compute_kl, compute_loss, rl_step,
)
from vivace.envs.base import Env, Example
from vivace.envs.gsm8k import GSM8KEnv
from vivace.rollout.hf_sampler import sample_responses
from vivace.rewards import extract_answer
from vivace.eval.runner import evaluate_model, compare_metrics
from vivace.utils.stats import TrainingStats
from vivace.utils.logging import ConsoleLogger, init_wandb, log_metrics, finish_wandb
from vivace.utils.perf import Timer, throughput
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
    "float8_e5m2": torch.float8_e5m2,           # FP8 alternate format (wider range, less precision)
}


@dataclass
class TrainerConfig:
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

    # ----- rollout -----
    rollout_batch_size: int = 64
    group_size: int = 8
    max_new_tokens: int = 1024      # max response length (HF generate / vLLM max_tokens)
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1

    # ----- training -----
    learning_rate: float = 1e-6
    kl_coef: float = 0.02
    num_steps: int = 500
    micro_batch_size: int = 4
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0

    # ----- composable PG (mirrors RLConfig) -----
    loss_type: str = "grpo"
    adv_type: str = "grpo"
    clip_low: float = 0.2
    clip_high: float = 0.2
    clip_ratio: float = 0.25
    clip_cispo: float = 5.0
    optim_epochs: int = 2
    adv_eps: float = 1e-4
    dg_eta: float = 1.0

    # ----- adaptive sampling / LR -----
    adaptive_sampling: bool = False
    oversample_factor: float = 2.0
    min_reward_spread: float = 0.5
    use_adaptive_lr: bool = False
    kl_target: float = 0.05
    kl_factor: float = 1.5
    kl_ema: float = 0.2
    lr_factor: float = 1.5
    min_lr: float = 1e-7
    max_lr: float = 2e-5

    # ----- SFT warmup (optional) -----
    sft_warmup: bool = False
    sft_warmup_steps: int = 0

    # ----- logging -----
    wandb_project: str = "vivace"
    wandb_run_name: str | None = None
    log_interval: int = 1
    eval_interval: int = 50
    checkpoint_interval: int = 100
    run_dir: str = "runs/default"

    # ----- misc -----
    seed: int = 42


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
    print(f"  Algo:            {cfg.loss_type} / {cfg.adv_type}")
    print(f"  Batch:           {cfg.rollout_batch_size} prompts x {cfg.group_size} group x {cfg.grad_accum_steps} accum")
    print(f"  LR:              {cfg.learning_rate}")
    print(f"  KL coef:         {cfg.kl_coef}")
    print(f"  Steps:           {cfg.num_steps}")
    print(f"  Run dir:         {cfg.run_dir}")
    print(f"{'=' * 60}\n")


class Trainer:
    """The training orchestrator. Owns the model, the rollout backend, the env,
    the algo (RLConfig), and the loop.

    Key invariants the loop maintains (do NOT break these in your impl):
      - Weight sync happens AFTER optimizer.step() and BEFORE the next
        rollout_phase. If you flip this order, the rollout sees stale
        weights and the importance ratio is wrong by one step (technically
        debuggable but very confusing).
      - Eval and checkpoint happen on rank 0 only. The other ranks must
        barrier() at the same point so they don't drift ahead.
      - Logging metrics use distributed all-reduce mean BEFORE the rank-0
        print, otherwise the printed loss is just rank 0's view.
    """

    def __init__(self, cfg: TrainerConfig):
        """Build everything.

        THEORY — wrap order matters
        ----------
        The correct order for setting up a LoRA + DDP model is:
            1. Load base model (HF, bf16, low_cpu_mem_usage=True)
            2. peft.get_peft_model(base, lora_cfg)        <-- LoRA wrap
            3. .to(device)
            4. DDP(model, device_ids=[local_rank])         <-- DDP wrap
        Other orders LOOK like they work but break checkpointing or
        gradient sync silently. Stick with this order.

        For full FT (no LoRA): skip step 2.

        Reference model handling
        ------------------------
        With LoRA: there is no separate ref model. Use peft's
        `with model.disable_adapter():` context inside compute_token_logprobs
        to forward the BASE model and treat that as the reference. Zero
        extra memory.

        Without LoRA: you need a separate frozen copy:
            ref_model = copy.deepcopy(base_model)
            ref_model.eval()
            for p in ref_model.parameters(): p.requires_grad_(False)
        ~2x model VRAM. Acceptable for small models, painful for big ones.

        GOTCHAS
        -------
        - Tokenizer: set padding_side="left", pad_token=eos_token BEFORE any
          generate calls. Forgetting this is the most common bug.
        - model.config.use_cache = False during training (else gradient
          checkpointing complains).
        - torch.backends.cuda.matmul.allow_tf32 = True for TF32 perf on
          ampere+ (free speedup).
        - Seed EVERYTHING after init: random, numpy, torch, torch.cuda.

        WHAT TO BUILD
        -------------
        - self.cfg = cfg
        - self.rank, self.local_rank, self.world_size = init_distributed()
        - tokenizer (with the padding fix)
        - base model (bf16, on local_rank GPU)
        - LoRA wrap (if cfg.use_lora) via peft.get_peft_model
        - DDP wrap (if world_size > 1)
        - ref_model handle (None if LoRA — disable_adapter trick;
          deepcopy if full FT)
        - env (build via cfg.env_name -> GSM8KEnv etc.)
        - rollout backend (HFSampler or VLLMRolloutWorker depending on mode)
        - optimizer (AdamW)
        - scheduler (LinearLR warmup + CosineAnnealingLR via SequentialLR)
        - TrainingStats, ConsoleLogger
        - init_wandb (rank 0 only)

        REFERENCES
        ----------
        - PyTorch DDP tutorial: pytorch.org/tutorials/intermediate/ddp_tutorial.html
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

        # --- RLConfig (built from TrainerConfig fields) ---
        self.rl_cfg = RLConfig(
            loss_type=cfg.loss_type,
            adv_type=cfg.adv_type,
            batch_size=cfg.micro_batch_size,
            group_size=cfg.group_size,
            lr=cfg.learning_rate,
            grad_accum_steps=cfg.grad_accum_steps,
            optim_epochs=cfg.optim_epochs,
            grad_clip=cfg.max_grad_norm,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            clip_low=cfg.clip_low,
            clip_high=cfg.clip_high,
            clip_cispo=cfg.clip_cispo,
            kl_coef=cfg.kl_coef,
            adv_eps=cfg.adv_eps,
            dg_eta=cfg.dg_eta,
            adaptive_sampling=cfg.adaptive_sampling,
            oversample_factor=cfg.oversample_factor,
            min_reward_spread=cfg.min_reward_spread,
            use_adaptive_lr=cfg.use_adaptive_lr,
            kl_target=cfg.kl_target,
            kl_factor=cfg.kl_factor,
            kl_ema=cfg.kl_ema,
            lr_factor=cfg.lr_factor,
            min_lr=cfg.min_lr,
            max_lr=cfg.max_lr,
            log_interval=cfg.log_interval,
        )
        validate_rl_config(self.rl_cfg)

        # --- Optimizer + scheduler ---
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        warmup_steps = 10
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(cfg.num_steps - warmup_steps, 1),
            eta_min=cfg.learning_rate * 0.2)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        # --- Stats + logging ---
        label = f"{cfg.loss_type}/{cfg.adv_type}"
        if cfg.use_lora:
            label += f" LoRA r={cfg.lora_rank}"
        self.stats = TrainingStats(method=label)
        self.console = ConsoleLogger(label=label)
        init_wandb(cfg, project=cfg.wandb_project, run_name=cfg.wandb_run_name)

        _print_training_info(cfg, self.model, self.trainer_devices, self.rollout_devices)

    def maybe_sft_warmup(self) -> None:
        """If cfg.sft_warmup, run cfg.sft_warmup_steps of SFT before RL.

        THEORY
        ------
        For most base models on GSM8K with the system prompt format,
        SFT is unnecessary. Keep this OFF by default.

        When you DO need it, the recipe is:
          - Build SFT data via vivace.algos.sft.build_sft_data
          - Build a separate AdamW with a higher LR (1e-5 typical)
          - Run vivace.algos.sft.sft_train_loop for cfg.sft_warmup_steps
          - The model is now warmed up; proceed to RL.

        HINTS
        -----
        - if not self.cfg.sft_warmup: return
        - sft_cfg = SFTConfig(steps=self.cfg.sft_warmup_steps, ...)
        - data = build_sft_data(self.env.load_split("train"), n=sft_cfg.n_examples,
                                tokenizer=self.tokenizer, make_prompt=self.env.format_prompt)
        - sft_opt = torch.optim.AdamW(self.model.parameters(), lr=sft_cfg.lr)
        - sft_train_loop(self.model, self.tokenizer, data, sft_cfg, sft_opt)
        """
        pass

    def rollout_phase(self) -> list[dict]:
        """Sample prompts -> generate -> compute rewards -> compute advantages.

        Returns a list of micro-batches in the format `rl_step` expects:
            {full_ids, plen, adv, old_logp, ref_logp, mask, token_count,
             responses, rewards}

        THEORY
        ------
        This is the rollout collection phase, pulled out into the trainer
        because the policy_gradient module shouldn't know about envs or
        sampling backends.

        Loop cfg.grad_accum_steps times. Each iteration:
          1. Sample B prompts from the env (or oversample for adaptive sampling)
          2. Repeat each prompt G times (group_size)
          3. Call sample_responses (HF) or vllm_worker.generate to get
             [B*G] sequences
          4. Compute rewards via env.reward_fn
          5. (Optional) adaptive sampling: filter dead groups by reward spread
          6. Compute advantages via algos/policy_gradient.compute_advantages
          7. Compute old_logp (under current model) and ref_logp (under ref)
             via compute_token_logprobs (no_grad)
          8. Pack into micro_batch dict

        GOTCHAS
        -------
        - old_logp is computed under the SAME model that just sampled.
          With colocated mode and a single optimizer step between rollouts
          this is fine. With disaggregated + multi-step queues, you need
          to be more careful.
        - For LoRA ref logp: use `with self.model.disable_adapter():`
          inside the forward pass.
        - Group sampling: prompts = [p for ex in batch_ex for p in [make_prompt(ex)] * G]
          gives you the right [B*G] order so view(B, G) works downstream.

        """
        micro_batches, alive_rates_step, spread_step = [], [], []
        B, G = self.cfg.micro_batch_size, self.cfg.group_size

        # {full_ids, plen, adv, old_logp, ref_logp, mask, token_count, responses, rewards}
        for i in range(self.cfg.grad_accum_steps):
            n_prompts = int(self.cfg.oversample_factor * B) if self.cfg.adaptive_sampling else B

            batch_ex = [random.choice(self.train_data) for _ in range(n_prompts)]
            # TODO: numpy batched way
            #idxs = np.random.choice(len(self.train_data), size=n_prompts, replace=False)
            #batch_ex = [self.train_data[i] for i in idxs]

            # Flatten: each prompt repeated G times, each answer repeated G times
            # TODO: investigate vectorizing format_prompt — would need Env.format_prompt_batch(examples)
            # or a batched string template. Current per-example call is ~microseconds so low priority,
            # but at large batch sizes (256+ prompts) the Python loop may show up in profiles.
            prompts = [self.env.format_prompt(ex) for ex in batch_ex for _ in range(G)]
            examples = [ex for ex in batch_ex for _ in range(G)]
            # TODO: vectorized repeat via np.repeat instead of nested comprehension
            # prompts = np.repeat([self.env.format_prompt(ex) for ex in batch_ex], G).tolist()

            full_ids, responses, plen = sample_responses(self.model, self.tokenizer, prompts, 
                                                        max_new_tokens=self.cfg.max_new_tokens,
                                                        temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                                                        device=str(self.device))
            torch.cuda.empty_cache()

            if self.cfg.adaptive_sampling:
                # TODO: to implement
                pass
            
            # reward_fn takes (response, Example) — full example available for
            # problem-aware rewards (e.g. bonus for using numbers from the problem)
            rewards = torch.tensor(
                [self.env.reward_fn(resp, ex) for resp, ex in zip(responses, examples)],
                device=self.device, dtype=torch.float32,
            )
            adv = compute_advantages(rewards.view(B, G), self.rl_cfg)

            with torch.no_grad():
                self.compiled_model.eval()
                old_logp, mask, _ = compute_token_logprobs(
                    self.compiled_model, full_ids, plen, self.cfg.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                self.compiled_model.train()

                if self.compiled_ref is not None:
                    ref_logp, _, _ = compute_token_logprobs(
                        self.compiled_ref, full_ids, plen, self.cfg.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    # LoRA: disable adapter to forward through base weights
                    with self.model.disable_adapter():
                        self.compiled_model.eval()
                        ref_logp, _, _ = compute_token_logprobs(
                            self.compiled_model, full_ids, plen, self.cfg.temperature,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        self.compiled_model.train()

            micro_batches.append({
                "full_ids": full_ids, "plen": plen, "adv": adv,
                "old_logp": old_logp, "ref_logp": ref_logp, "mask": mask,
                "token_count": mask.sum(dim=1).clamp(min=1.0),
                "responses": responses, "rewards": rewards,
                "pad_token_id": self.tokenizer.pad_token_id,
            })
        return micro_batches

    def train_phase(self, micro_batches: list[dict]) -> dict:
        """Multi-epoch optimization over collected micro-batches.

        Thin wrapper around vivace.algos.policy_gradient.rl_step. The reason
        it lives on Trainer at all is so the Trainer owns the optimizer
        and the kl_ema state across steps.

        HINTS
        -----
        - return rl_step(self.rl_cfg, micro_batches, self.model, self.ref_model,
                         self.optimizer, self.stats, self.step, self.kl_ema)
        - Capture the returned new_kl_ema and store it on self.
        """
        metrics, self.kl_ema = rl_step(
            self.rl_cfg, micro_batches, self.compiled_model, self.ref_model,
            self.optimizer, self.stats, self.step, self.kl_ema,
        )
        return metrics
        
    def sync_weights(self) -> None:
        """Push trainer weights to the rollout worker.

        Two paths:
          - LoRA: save adapter to disk, call vllm_worker.update_lora(path)
                  Simple, slow, always works.
          - Full FT: build NCCL subgroup once, broadcast each parameter,
                     call vllm_worker.update_weights(state_dict).
                     Hard, fast.

        Single-GPU dev with HFSampler: this is a no-op (the trainer's
        model IS the sampler's model — they share weights by reference).
        Skip the whole thing in that mode.
        """
        # HF sampler shares model by reference — weights already in sync.
        if not hasattr(self, 'rollout_worker') or self.rollout_worker is None:
            return

        adapter_path = os.path.join(self.cfg.run_dir, "adapter_sync")
        if self.cfg.use_lora:
            if is_main_process():
                self.model.save_pretrained(adapter_path)
            barrier()
            self.rollout_worker.update_lora(adapter_path)
        else:
            # full FT: in-place NCCL broadcast (much later)
            raise NotImplementedError("full FT weight sync not yet implemented")

    def train(self) -> None:
        """Outer training loop.

        Structure (this is the part you DON'T need to change):
          1. Optional SFT warmup
          2. Baseline eval
          3. For step in range(num_steps):
              a. rollout_phase
              b. train_phase
              c. sync_weights
              d. (rank 0) periodic logging via wandb
              e. (rank 0) periodic eval
              f. (rank 0) periodic checkpoint
          4. Final eval
          5. wandb finish
        """
        self.maybe_sft_warmup()

        # --- Baseline eval ---
        eval_data = self.env.load_split("eval")
        print("Evaluating baseline...")
        baseline_metrics, baseline_correct, baseline_incorrect = evaluate_model(
            self.model, self.tokenizer, eval_data, self.env, n=200,
            device=str(self.device),
        )
        print(f"Baseline accuracy: {baseline_metrics['accuracy_pct']:.1f}%  format: {baseline_metrics['format_rate_pct']:.1f}%")
        if is_main_process():
            log_metrics({"eval/" + k: v for k, v in baseline_metrics.items()}, step=0)

        # --- Training loop ---
        self.kl_ema = self.cfg.kl_target
        for step in range(self.cfg.num_steps):
            self.step = step

            with Timer() as step_t:
                with Timer() as rollout_t:
                    micro_batches = self.rollout_phase()
                with Timer() as train_t:
                    metrics = self.train_phase(micro_batches)
                self.sync_weights()

            self.scheduler.step()

            # --- Log everything in one stats.log() call ---
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
                    eval_metrics, _, _ = evaluate_model(
                        self.model, self.tokenizer, eval_data, self.env, n=200,
                        device=str(self.device),
                    )
                    print(f"  [eval] Step {step:04d} | accuracy={eval_metrics['accuracy_pct']:.1f}%")
                    log_metrics({"eval/" + k: v for k, v in eval_metrics.items()}, step)

                # --- Periodic checkpoint (stub — save_checkpoint not implemented yet) ---
                # if step > 0 and step % self.cfg.checkpoint_interval == 0:
                #     save_checkpoint(self.model, self.optimizer, self.scheduler,
                #                     step, f"{self.cfg.run_dir}/ckpt-{step}")

        # --- Final eval ---
        print("\nFinal evaluation...")
        final_metrics, final_correct, final_incorrect = evaluate_model(
            self.model, self.tokenizer, eval_data, self.env, n=200,
            device=str(self.device),
        )
        print(f"Final accuracy: {final_metrics['accuracy_pct']:.1f}%  format: {final_metrics['format_rate_pct']:.1f}%")
        compare_metrics(baseline_metrics, final_metrics, label="RL training")

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
        run_label = f"{self.cfg.loss_type}_{self.cfg.adv_type}_{self.cfg.num_steps}steps"
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
