"""torch.distributed helpers for DDP / multi-rank training.

Env vars required (set automatically by `torchrun`):
    RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

When these are absent (single-process dev), helpers return sensible defaults
(rank=0, world_size=1) without touching torch.distributed.

NCCL weight sync to the vLLM worker lives in `vivace/utils/weight_sync.py`
and uses a separate `StatelessProcessGroup` rather than the default group
here — see `docs/weight_sync_approaches.md` for why.
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
