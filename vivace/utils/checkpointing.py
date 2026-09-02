"""Checkpoint save / load.

==============================================================================
WHY THIS IS A STUB
==============================================================================

You asked to implement checkpointing yourself for learning value. trainer.py
already saves weights via `save_pretrained` (weight sync, `final_model`); periodic
`ckpt-<step>` waits on `save_checkpoint`. Comments below are heavy on the "why":
what to save, what it costs, how it fails.

==============================================================================
WHAT TO SAVE IN A CHECKPOINT
==============================================================================

A "complete" checkpoint that survives a process restart contains:

  1. Model state                  - the parameters themselves
  2. Optimizer state              - moments (Adam: m1, m2), step counter
  3. Scheduler state              - LR + step counter so resumes hit the
                                    same LR as if no interruption
  4. Step counter                 - global training step
  5. RNG state                    - random / numpy / torch / torch.cuda
                                    so the next batch is reproducible
  6. (Optional) wandb run id      - so resume continues the same wandb run
  7. (Optional) data loader state - which examples have been seen,
                                    if you want bit-exact resume

For research, items 1-5 are mandatory. 6 is nice. 7 is overkill until
you're publishing.

==============================================================================
SIZE
==============================================================================

For Qwen2.5-0.5B (~0.5B params):
  - Model bf16        :   1 GB
  - AdamW state fp32  :   4 GB   (m1: 2 GB + m2: 2 GB, both at fp32 even
                                  if model is bf16)
  - Scheduler         :   < 1 KB
  - RNG               :   < 1 KB

The optimizer state is roughly 4x the model size. Knowing this prevents
"why is my checkpoint 5x bigger than my model file" surprise.

For LoRA, the model state is REPLACED by the adapter state, which is tiny
(MBs). Save the adapter via peft instead of saving full state_dict.

==============================================================================
THE GOLDEN RULES
==============================================================================

Rule 1: ONLY rank 0 writes to disk. Other ranks call barrier() and wait.
        Two ranks writing to the same file at the same time = corruption.

Rule 2: Atomic rename. Write to "ckpt.pt.tmp", then os.rename to "ckpt.pt".
        If the process dies mid-write, the old checkpoint is intact and
        the .tmp file is incomplete-but-isolated.

Rule 3: Save async if you can afford the dev cost. The save itself is
        I/O-bound and blocks the optimizer. Spawn a thread to do the
        actual write. Skip this until checkpointing becomes a bottleneck.

Rule 4: For LoRA, save the ADAPTER ONLY (peft.PeftModel.save_pretrained).
        Base weights never change; saving them every checkpoint wastes
        disk and time.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def save_checkpoint(model, optimizer, scheduler, step: int, path: str) -> None:
    """Save model + optimizer + scheduler + RNG state to `path`.

    HINTS
    -----
    - Build a state dict:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }
    - tmp_path = path + ".tmp"
    - torch.save(ckpt, tmp_path)
    - os.replace(tmp_path, path)    # atomic rename
    - Only do all of this on rank 0. Other ranks barrier and skip.

    GOTCHAS
    -------
    - For DDP: model.state_dict() works on the wrapped DDP module but
      stores parameter names with `module.` prefix. Either save
      model.module.state_dict() or strip the prefix on load.
    - os.replace is atomic on the same filesystem. Across filesystems,
      it's not. Keep the .tmp file in the same directory as the target.
    - Don't pickle the optimizer's parameter REFERENCES — torch.save handles
      this correctly via its own serialization protocol. Just call .state_dict().
    """
    # TODO: implement.
    raise NotImplementedError


def load_checkpoint(path: str) -> dict:
    """Load and return the raw checkpoint dict.

    The caller is responsible for calling
    `model.load_state_dict(ckpt['model'])` etc. — this function just
    returns the dict so the caller can decide whether to load everything
    or only some pieces (e.g., model only when fine-tuning from a different
    base).

    HINTS
    -----
    - return torch.load(path, map_location="cpu")
    - For DDP: load on cpu first, then `model.load_state_dict(ckpt['model'])`
      — DDP will move things to the right device on first forward.
    - Restore RNG state if present:
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    """
    # TODO: implement.
    raise NotImplementedError


def save_lora_adapter(model, path: str) -> None:
    """Save just the LoRA adapter (no base weights).

    For peft-wrapped models; adapters are ~MBs, not GBs. trainer.py already
    does this via `_inner_model.save_pretrained` on rank 0.

    HINTS
    -----
    - Detect peft wrap: `from peft import PeftModel; isinstance(model, PeftModel)`
    - For DDP-wrapped peft models: use model.module.save_pretrained(path)
    - Use model.save_pretrained(path) — peft writes adapter_config.json
      + adapter_model.bin (or .safetensors)
    """
    # TODO: implement.
    raise NotImplementedError


def load_lora_adapter(model, path: str) -> None:
    """Load a LoRA adapter into an already-wrapped peft model.

    HINTS
    -----
    - The model passed in must already be wrapped with the SAME LoraConfig
      that was used to save the adapter. peft can't infer it on load.
    - Use `model.load_adapter(path, adapter_name="default")` or for full
      replace: re-wrap from base + load.
    - For DDP-wrapped models, do the load on model.module.
    """
    # TODO: implement.
    raise NotImplementedError
