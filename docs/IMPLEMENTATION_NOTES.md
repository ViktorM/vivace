# Implementation Notes

Pending work, landed infrastructure, and experiment notes without a home yet.
Priority 1 (single-GPU slice) and Priority 5 (vLLM rollout) are done. Update
as items land.

## SFT warmup
Stubs exist; never actually needed for GSM8K + Qwen instruct. Implement when
required by an env that genuinely benefits from cold-start SFT.

- `vivace/algos/sft.py`: `build_sft_data`, `encode_sft_batch`, `sft_loss`,
  `sft_train_loop`
- `vivace/train/trainer.py::Trainer.maybe_sft_warmup` — currently a no-op
- `vivace/scripts/sft.py::main` — raises NotImplementedError

## Checkpointing
`Trainer.train` saves the final policy to `{run_dir}/final_model` (LoRA:
adapter + tokenizer; full FT: full HF checkpoint). No mid-run checkpoint or
resume: the periodic `save_checkpoint` call is commented out.

- `vivace/utils/checkpointing.py`: all four functions raise NotImplementedError
- `vivace/scripts/eval.py::main` — wire up `load_checkpoint` for eval-only

## Distributed (DDP)

What works:
- N-rank DDP under torchrun, colocated and disaggregated. DDP wrap in
  `Trainer.__init__` (peft → DDP order; `_inner_model` saved before wrap so
  peft methods stay reachable through DDP's `__getattr__` wall;
  `device_ids=[self.device.index]`, not `local_rank`, so `trainer_gpus=[2,3]`
  works without an outer CUDA_VISIBLE_DEVICES mask).
- Disaggregated DDP: 1:1 trainer:rollout pairing (`len(rollout_gpus) ==
  WORLD_SIZE`, disjoint sets); each rank owns its vLLM worker and its own NCCL
  sync comm on a rank-spaced port (29600 + 10·rank) — no global trainer↔vLLM
  group.
- Per-rank seeding `cfg.seed + 1000*rank` (seed 41 rank 1 ≠ seed 42 rank 0) —
  verified by `tests/test_rank_divergence.py`.
- Sharded eval: `_run_eval` gives each rank a contiguous slice of a seed-fixed
  shuffle, all-reduces counts/sums, `all_gather_object`s the per-sample lists.
  Every rank must enter `_run_eval_all` or the collectives deadlock.
- Learning metrics reduced across ranks (`_reduce_learning_metrics`): means /
  extrema directly, stds from Σx/Σx² (mean-of-stds under-estimates spread),
  rates from summed counts; grad_norm passes through — DDP already averaged
  the gradients.
- Throughput metrics (`tokens_per_sec`, `samples_per_sec`, `rollout_tokens`,
  `rollout_samples`) summed across ranks; wall-clock times max'd across
  ranks — via `reduce_metrics` in `vivace/utils/distributed.py`.
- DDP `no_sync()` on every micro-batch but the last per epoch in `rl_step`.
  Bit-identical math; ~0.1% saving at LoRA r=16 / 2×4090 PCIe, ~20% at
  full-FT 0.5B.
- Token-mean losses (dapo; cispo token/hybrid) divide by the global token
  count — all-reduced, then / (n_micro × world_size). Per-micro-batch or
  per-rank denominators upweight short micro-batches and low-token ranks.
- Allocator hygiene: colocated does `gc.collect()` + `empty_cache()` before
  `vllm.sleep()` and `empty_cache()` before `wake_up()`, so the trainer's pool
  doesn't squeeze vLLM's and trip its `freed_bytes >= 0` sleep assertion;
  disaggregated does both once per step (variable-T blocks otherwise drift
  the pool up until OOM ~step 150-200 on 1.5B + MATH).
- `cfg.find_unused_parameters` exposed; default False is correct for LoRA on
  attention+MLP projections.

Not done:
- **Sharded vLLM weight sync (TP > 1).** `tensor_parallel_size != 1` raises
  at init; the worker hard-codes `tp_rank = 0` and `pack_ipc_handles` builds
  one handle per full tensor.
- **Dedicated eval workers** (`cfg.eval_gpus`): field exists, construction is
  a TODO; eval reuses the rollout worker.

## Training-quality ideas to explore later

Notes from the Qwen2.5-0.5B + GSM8K + DAPO/RLOO/LoRA-r16 sweep showing
accuracy plateaus around 50–52% by step 100 and stays there for the
remaining 400 steps. Most learning is "produce `\boxed{...}` and stop";
reasoning gains are marginal at this model size on this env.

- **Process Reward Models (PRM).** Step-level rewards on each reasoning
  step rather than only final-answer correctness. Strong signal for small
  models that struggle to produce coherent multi-step chains.
- **Length penalty in reward function.** Discourage rambling explicitly.
  Small but consistent gains; cheap to add (modify `env.reward_fn` to
  subtract a coefficient × length).
- **LR schedule experiments — done, all three negative.** `cfg.rl.eta_min_ratio`
  and `cfg.rl.lr_restart` are exposed. Tested on Qwen2.5-0.5B / GSM8K /
  DAPO-RLOO / LoRA-r16 / 500 steps, single seed=42:
  - `eta_min_ratio: 0.2 → 0.4`: within noise (53.2% → 52.4%).
  - `lr_restart: True`, hard restart (jump from eta_min to peak in one
    step): meaningfully worse (53.2% → 48.6%, ~2σ). kl/grad_norm spike at
    step 320, no recovery. Hard restart knocked the policy into an
    inferior basin.
  - `lr_restart: True`, soft restart (linear ramp eta_min → peak over
    `warmup_steps`): within noise (53.2% → 53.0%). Soft enough to not
    destabilize, gentle enough not to shake the policy out of its basin.
  - **Conclusion**: 50–53% ceiling is structural at this model size on
    GSM8K. LR schedule isn't the bottleneck. Pushing past requires bigger
    model, harder env, or different reward (PRM, length penalty).
- **Larger model (1.5B disaggregated locally, 7B+ on cluster).** 0.5B is
  near its ceiling on GSM8K with this recipe. Configs exist for 1.5B and
  0.8B already; 7B+ needs the cluster path.
- **Benchmark horizon = 200 steps × more seeds.** Most learning is done
  by step 100; spending wall-clock on more seeds (3–5) at 200 steps
  beats single-seed 500-step runs for variance-bar quality on the
  loss-comparison plot.

## Commit-time improvements landed

- `compile_model: true` tested on 0.5B + DDP: ~13% per-step speedup but
  variable-T causes recompile overhead (~6 min upfront), break-even at
  ~1500 steps. Worse: it changes training dynamics enough to invalidate
  algorithm comparisons (length-cap pinning at 192 vs healthy 130 for
  uncompiled). Default stays False; only flip for long production runs
  with cache-warm second-runs.
- `max_new_tokens: 192 → 256` on the colocated DDP DAPO config. Baseline
  cap rate ~70%, drops to ~5% by step 100; final accuracy comparable
  (47–50% range, within noise). 256 is the new default for this config.

## CISPO algorithmic improvements (2026-05)

The step-200 collapse pattern documented in earlier runs is now traceable
to two interacting issues, both fixed in the May 2026 commit.

**1. Canonical CISPO is now the default (eq. 5 only).** The previous
implementation unconditionally applied the MiniMax-M1 eq. 7 token-dropping
mask — zeroing the gradient for tokens where `ratio > clip_cispo_high`
AND `advantage > 0`. That created an asymmetric trust region: high-ratio
positive-advantage tokens stopped contributing while negative-advantage
tokens kept full gradient. The M1 paper presents eq. 7 as optional;
vivace was running CISPO+mask and attributing instability to CISPO. The
mask is now opt-in via `rl.cispo_use_token_mask: true`; default is the
M1-canonical eq. 5 clipped-IS recipe.

**2. `rl.cispo_normalization`** (token | sequence | hybrid, default
`hybrid`). Token-level normalization (the previous fixed behavior)
disproportionately weights long sequences with negative advantage —
exactly the "pattern collapse" failure mode the M1 report calls out.
Hybrid averages token- and sequence-mean losses for length-robustness.

**3. fp32 reductions** in `compute_kl` and `compute_loss`. bf16 has only
~3 decimal digits of mantissa precision; summing thousands of log-prob
differences in bf16 accumulates real error. All reductions now cast to
fp32 before the sum/divide.

**4. `rl.kl_coef` default bumped 0.01 → 0.02.** Empirically validated as
the lowest stable anchor across the May 2026 sweep. Pre-existing configs
that set `kl_coef: 0.01` explicitly keep that value.

**5. `clip_frac` metric** now reports tokens whose IS weight was clipped,
not tokens dropped by the (now opt-in) mask. Comparable to PPO `clip_frac`.

**2×2 ablation on ep=4 (1.5B, gs=16 bs=2, max=1024, lr=3e-5, kl=0.02):**

| mask | norm  | gsm8k | math500 | KL₁₉₉ | entropy₁₉₉ |
|------|-------|-------|---------|-------|-----------|
| true  | token   | 74.75 | 39.00 | 0.084 | 0.33 |
| true  | hybrid  | 74.22 | 37.80 | 0.053 | 0.27 |
| false | hybrid  | **74.60** | **38.40** | **0.065** | **0.24** |
| false | token   | 73.09 | 35.40 | **0.285** | **1.94** ✗ collapsed |

Either stabilizer (mask OR hybrid norm) prevents the collapse seen with
neither. Canonical+hybrid is the principled default — algorithmically
matches the M1 paper claim AND has the lowest entropy at finish.
