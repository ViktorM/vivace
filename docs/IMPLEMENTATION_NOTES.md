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
`vivace/utils/distributed.py` has `is_main_process`, `barrier`, `get_rank`,
`get_world_size`. The DDP wrap in `Trainer.__init__` and the main-process
gating in `train()` aren't done — see `docs/training_theory.md` "Wrap order"
for the order to follow when implementing.
