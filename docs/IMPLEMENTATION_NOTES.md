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
