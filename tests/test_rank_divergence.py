"""Verify that DDP ranks pick distinct rollouts.

Mirrors the trainer's per-rank seeding (`cfg.seed + 1000*rank`; here
`SEED + rank`): distinct numpy seeds make `np.random.choice` over the train set
yield different prompt indices per rank. If this fails, all ranks sample the
same prompts and DDP gradient averaging collapses to single-GPU dynamics at
2x the wall-clock cost.

Scope:
  - Asserts data-sharding divergence (np.random with seed+rank → different draws).
  - Does NOT exercise vLLM here — vLLM trajectory divergence is a property of
    each EngineCore subprocess having its own RNG state by default. To
    re-verify end-to-end, run the trainer for `--num-steps 1` on the DDP
    config and inspect the rollout response hashes.

Usage:
    torchrun --nproc_per_node=2 -m tests.test_rank_divergence
    torchrun --nproc_per_node=4 -m tests.test_rank_divergence  # works at any N>=2

Exit code:
    0 — every rank's draw is unique (data sharding works)
    1 — at least two ranks drew identical prompts (sharding broken)
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import torch.distributed as dist

from vivace.utils.distributed import init_distributed


SEED = 42
N_TRAIN = 7473        # GSM8K train size; any non-trivial pool works
N_PROMPTS = 32        # match a typical step's prompt count


def main() -> int:
    rank, _, world_size = init_distributed()
    if world_size < 2:
        if rank == 0:
            print("ERROR: need world_size >= 2; launch via torchrun --nproc_per_node=N",
                  file=sys.stderr)
        return 1

    np.random.seed(SEED + rank)
    idxs = np.random.choice(N_TRAIN, size=N_PROMPTS, replace=False)
    prompt_hash = hashlib.md5(idxs.tobytes()).hexdigest()[:12]

    local = {"rank": rank, "hash": prompt_hash, "first": idxs[:8].tolist()}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)

    if rank == 0:
        print(f"[test_rank_divergence] world_size={world_size} seed={SEED}")
        for g in gathered:
            print(f"  rank={g['rank']:<3} hash={g['hash']} first8={g['first']}")
        unique = {g["hash"] for g in gathered}
        if len(unique) < world_size:
            print(f"FAIL: only {len(unique)} unique hashes across {world_size} ranks "
                  f"— at least two ranks drew identical prompts", file=sys.stderr)
            dist.destroy_process_group()
            return 1
        print(f"PASS: {len(unique)}/{world_size} ranks have distinct prompt sets")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
