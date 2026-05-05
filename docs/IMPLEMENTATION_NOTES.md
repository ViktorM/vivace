# Implementation Notes

What's still pending in the codebase. Priority 1 (single-GPU vertical slice)
and Priority 5 (vLLM rollout) from the original plan are done. Update this
file as items land.

## SFT warmup
Stubs exist; never actually needed for GSM8K + Qwen instruct. Implement when
required by an env that genuinely benefits from cold-start SFT.

- `vivace/algos/sft.py`: `build_sft_data`, `encode_sft_batch`, `sft_loss`,
  `sft_train_loop`
- `vivace/train/trainer.py::Trainer.maybe_sft_warmup` — currently a no-op
- `vivace/scripts/sft.py::main` — raises NotImplementedError

## Checkpointing
LoRA adapter save/load works via peft for the disk weight-sync path. Full FT
checkpoint round-trip is not wired up.

- `vivace/utils/checkpointing.py`: `save_checkpoint`, `load_checkpoint`,
  `save_lora_adapter`, `load_lora_adapter`
- `vivace/scripts/eval.py::main` — wire up `load_checkpoint` for eval-only

## Distributed (DDP)

What works:
- N-rank DDP under torchrun in colocated mode. DDP wrap in `Trainer.__init__`
  (peft → DDP order; `_inner_model` saved before wrap so peft methods stay
  reachable through DDP's `__getattr__` wall).
- Per-rank seeding (`cfg.seed + rank`) — verified by `tests/test_rank_divergence.py`.
- Eval / logging / wandb / checkpoint-save gated on rank 0 with barriers.
- `torch.cuda.empty_cache()` between train_phase and the next iteration's
  vLLM wake_up (colocated only) — keeps the trainer's PyTorch allocator pool
  from squeezing vLLM's pool over long runs (otherwise trips vLLM's
  `freed_bytes >= 0` sleep-time assertion after several hundred steps).
- `cfg.find_unused_parameters` exposed; default False is correct for LoRA on
  attention+MLP projections.
- Throughput metrics (`tokens_per_sec`, `samples_per_sec`, `rollout_tokens`,
  `rollout_samples`) summed across ranks; wall-clock times max'd across
  ranks — via `reduce_metrics` in `vivace/utils/distributed.py`.

In progress:
- Sharded eval across ranks (currently rank-0-only with a barrier on the
  others). Eval is fast at small scale but becomes a real cost at larger
  models / eval sets.
- Cross-rank reduction of *learning* metrics (loss/reward/kl/grad_norm/...).
  Helper exists; only throughput is wired today, so wandb learning curves
  reflect rank-0-local values.

Future optimizations (enable when these constraints actually bind):
- **DDP `no_sync()` during gradient accumulation.** Suppresses all_reduce on
  every backward except the last per optimizer step. Bit-identical math;
  pure plumbing. At LoRA r=16 / 2 GPUs PCIe the saving is ~0.1% (below noise);
  at full-FT 0.5B it's ~20%; at 4+ GPU or larger LoRA ranks it's a
  meaningful win. Wire when switching to full-FT or scaling past 2 ranks.
  ~5 lines: a `_grad_sync_ctx(model, sync_now)` helper returning
  `model.no_sync()` if DDP-wrapped and not the last micro-batch, else
  `nullcontext()`. Apply at the `loss.backward()` call site in `rl_step`.
- **Disaggregated DDP (4+ GPUs).** Trainer ranks on one set of GPUs, vLLM
  workers on another. Today's colocated DDP shares GPUs via vllm.sleep;
  splits to separate trainer vs rollout pools when GPU count grows.
- **Sharded vLLM weight sync (TP > 1).** v1 forces tensor_parallel_size=1.
  Lifting requires per-shard handle building in `pack_ipc_handles`.

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
- **LR schedule experiments — done, both negative.** `cfg.rl.eta_min_ratio`
  and `cfg.rl.lr_restart` are now exposed. Tested on Qwen2.5-0.5B / GSM8K /
  DAPO-RLOO / LoRA-r16 / 500 steps:
  - `eta_min_ratio: 0.2 → 0.4`: within noise (53.2% → 52.4%).
  - `lr_restart: True` (T_0 = (num_steps - warmup) // 2): meaningfully
    worse (53.2% → 48.6%, ~2σ). Hard restart at step ~255 knocked the
    policy into an inferior basin; kl/grad_norm spike at step 320, no
    recovery. Conclusion: 50–53% ceiling is structural at this model size,
    not LR-decay-limited.
  - **Possible softer-restart variant to try later**: `T_0=100, T_mult=2`
    (PyTorch's `CosineAnnealingWarmRestarts` second arg = period growth
    factor). Shorter first cycle, doubling lengths after, gives gentler
    perturbations earlier in training where the policy is still adapting.
    Would need a small extension to the config schema (T_0/T_mult fields).
    Low priority — the negative result above suggests this whole class of
    knob isn't where the gains are.
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
