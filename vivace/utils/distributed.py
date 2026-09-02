"""torch.distributed helpers for DDP / multi-rank training.

Env vars required (set automatically by `torchrun`):
    RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

When these are absent (single-process dev), helpers return sensible defaults
(rank=0, world_size=1) without touching torch.distributed.

NCCL weight sync to the vLLM worker lives in `vivace/utils/weight_sync.py`
and uses a separate `StatelessProcessGroup`: vLLM's EngineCore is a subprocess,
not a torchrun rank, so each DDP rank pairs with its own worker over a 2-rank comm.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Initialize the default torch.distributed process group.

    Returns (rank, local_rank, world_size). In single-process mode (env vars
    not set) returns (0, 0, 1) without calling init_process_group.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return rank, local_rank, world_size


def is_main_process() -> bool:
    """True on rank 0 or in single-process mode."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Global rank. 0 if not distributed."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Number of processes. 1 if not distributed."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Block until all ranks arrive. No-op if not distributed."""
    if not dist.is_initialized():
        return
    dist.barrier()


def all_reduce_mean(tensor):
    """In-place all-reduce then divide by world_size. No-op if not distributed.

    Mutates `tensor`. Float tensors only.
    """
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def reduce_metrics(
    metrics: dict[str, float],
    ops: dict[str, str],
) -> dict[str, float]:
    """Reduce a flat scalar dict across DDP ranks. No-op when world_size == 1.

    ops maps each key to one of {"mean", "sum", "max", "min"}. Keys in `metrics`
    but not in `ops` are passed through unreduced (use for already-global values
    like the LR or the step counter). Keys in `ops` but missing from `metrics`
    are silently ignored, so a single op map can serve runs with different
    metric subsets.

    Packs values per op into one tensor and issues one collective per op kind
    (≤ 4 collectives per call, each on a tiny tensor — overhead is microseconds,
    well below gradient-sync cost).

    Notes:
      - Returns a new dict; does not mutate `metrics`.
      - All ranks must call this every step or the collectives deadlock.
      - Std-of-stds is not exposed here on purpose: mean-of-stds under-estimates
        global spread. Use all_gather to recompute std globally if you need it.
    """
    groups: dict[str, list[str]] = {"mean": [], "sum": [], "max": [], "min": []}
    for k, op in ops.items():
        if op not in groups:
            raise ValueError(f"unknown reduce op {op!r} for key {k!r}; "
                             f"expected one of {list(groups)}")
        if k in metrics:
            groups[op].append(k)

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return dict(metrics)

    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    out = dict(metrics)
    op_to_reduce = {
        "mean": dist.ReduceOp.SUM,   # divided by world_size after reduce
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    for op_name, keys in groups.items():
        if not keys:
            continue
        # float64 to keep large counters (rollout_tokens) exact under SUM and to
        # avoid surprising precision loss on small float metrics under MEAN.
        vals = torch.tensor([float(metrics[k]) for k in keys],
                            dtype=torch.float64, device=device)
        dist.all_reduce(vals, op=op_to_reduce[op_name])
        if op_name == "mean":
            vals /= world_size
        for i, k in enumerate(keys):
            out[k] = float(vals[i].item())
    return out
