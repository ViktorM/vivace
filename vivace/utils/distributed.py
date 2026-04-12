"""Distributed training helpers.

==============================================================================
THEORY OVERVIEW (read this first)
==============================================================================

PyTorch's distributed package builds on top of one of three backends:
  - NCCL  : NVIDIA's collective comms library. GPU-to-GPU. FAST.
            What you want for training.
  - GLOO  : CPU fallback. SLOW, but works without CUDA. Good for tests.
  - MPI   : if you must. Rarely used in PyTorch land.

To get any distributed op working, every process needs four pieces of
information:

  1. WORLD_SIZE  : total number of processes across all nodes
  2. RANK        : this process's GLOBAL ID (0 .. world_size-1)
                   Unique across the entire job.
  3. LOCAL_RANK  : this process's GPU index ON ITS OWN NODE
                   (0 .. gpus_per_node-1). Used for set_device.
  4. MASTER_ADDR + MASTER_PORT : rendezvous point for the TCP/file store
                                 used during init_process_group

`torchrun` sets all four env vars for you:
    torchrun --nproc_per_node=2 vivace/scripts/train.py --config ...
sets RANK and LOCAL_RANK per worker, plus WORLD_SIZE = nproc_per_node.

If you launch via mpirun or SLURM, you set them yourself by reading
SLURM_PROCID, SLURM_LOCALID, etc. and exporting them as env vars.

==============================================================================
init_process_group — what does it actually do?
==============================================================================

`dist.init_process_group(backend="nccl", ...)` opens NCCL communicators
between all ranks listed in the WORLD. After this call:

  - dist.broadcast(tensor, src=0)   sends rank 0's tensor to all ranks
  - dist.all_reduce(tensor, op=SUM) sums the tensor across all ranks
  - dist.barrier()                  blocks until all ranks reach this point

UNTIL you call init_process_group, none of these will work.

Each NCCL communicator owns one ring buffer + bookkeeping per device.
Initialization costs ~hundreds of MB of GPU memory PER COMMUNICATOR.
This is why you don't make a million subgroups — they're not free.

==============================================================================
DDP — how gradients sync
==============================================================================

`torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])`
wraps your model so that, during the BACKWARD pass, every parameter's
gradient gets all-reduced across ranks via autograd hooks. By the time
backward returns, every rank has the same gradients (the average).

This means you do `loss.backward(); optimizer.step()` exactly as in
single-GPU code, and DDP makes the gradients consistent. The cost is
N rounds of NCCL all-reduce per backward, one per parameter (DDP buckets
them for efficiency).

==============================================================================
FSDP — when DDP isn't enough
==============================================================================

DDP keeps a FULL copy of the model on EVERY rank. For Qwen2.5-0.5B that's
fine. For 7B+ you start running out of GPU memory because of the optimizer
state (Adam: 4x model size in fp32). FSDP shards the model itself across
ranks — every rank holds 1/N of the parameters and only gathers the full
parameter when it's needed for a forward/backward pass.

Use FSDP when:
  - Model + optimizer state > GPU memory
  - You're willing to pay the gather overhead

Use DDP when:
  - Model fits comfortably in GPU memory
  - You want max throughput per step

vivace targets DDP first (because Qwen2.5-0.5B is small) and graduates
to FSDP later for larger models.

==============================================================================
THE WEIGHT SYNC SUBGROUP (vLLM bridge)
==============================================================================

When the trainer and the vLLM rollout worker live in different processes,
the trainer needs to push freshly-updated weights to vLLM after each
optimizer step. The cleanest way is a SECOND process group that includes
both the trainer ranks AND the vLLM worker ranks.

  - Trainer ranks       : 0, 1            (DDP world)
  - vLLM worker ranks   : 2, 3            (vLLM tp=2 world)
  - Weight sync group   : 0, 1, 2, 3      (used purely for broadcasting)

`dist.new_group([0, 1, 2, 3])` builds this subgroup. Then:

    dist.broadcast(param.data, src=0, group=sync_group)

sends rank 0's parameter tensor to all four ranks. The vLLM workers
receive into a buffer and copy_() into vLLM's underlying model.

`make_weight_sync_group` is a thin helper that owns the call to
`dist.new_group` and stashes the result somewhere `vllm_worker.update_weights`
can find it.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Initialize the default torch.distributed process group.

    Returns (rank, local_rank, world_size).

    THEORY
    ------
    See module docstring §init_process_group. This function is the entry
    point that turns a regular Python process into a distributed one.

    GOTCHAS
    -------
    - The "device_id" hint to init_process_group is needed in newer PyTorch
      to avoid a deprecation warning + slow eager init. Pass
      `device_id=torch.device(f"cuda:{local_rank}")`.
    - If init hangs, MASTER_PORT is probably already in use. Pick another.
    - If you forget set_device(local_rank), you OOM on GPU 0 with N copies
      of the model and wonder why. Always set device IMMEDIATELY after init.
    - When running without distributed (single process, no torchrun): the
      env vars are missing. Detect this and return (0, 0, 1) without
      calling init_process_group at all.

    HINTS
    -----
    - If "RANK" not in os.environ: return (0, 0, 1)
    - rank = int(os.environ["RANK"])
    - local_rank = int(os.environ["LOCAL_RANK"])
    - world_size = int(os.environ["WORLD_SIZE"])
    - import torch; import torch.distributed as dist
    - torch.cuda.set_device(local_rank)
    - dist.init_process_group(
          backend="nccl",
          device_id=torch.device(f"cuda:{local_rank}"),
      )
    - return rank, local_rank, world_size

    REFERENCES
    ----------
    - https://pytorch.org/docs/stable/distributed.html
    - PyTorch DDP tutorial: pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    # TODO: implement.
    pass


def is_main_process() -> bool:
    """True if this is rank 0 (or single-process). Used to gate logging/eval/saves.

    HINTS
    -----
    - import torch.distributed as dist
    - if not dist.is_available() or not dist.is_initialized(): return True
    - return dist.get_rank() == 0
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_rank() -> int:
    """This process's global rank. 0 if not distributed.
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Total number of distributed processes. 1 if not distributed.

    HINTS
    -----
    - if not dist.is_initialized(): return 1
    - return dist.get_world_size()
    """
    # TODO: implement.
    raise NotImplementedError


def barrier() -> None:
    """Block until all ranks reach this point.

    THEORY
    ------
    Critical when only rank 0 does some work (e.g., writing a checkpoint)
    and you need the other ranks to wait for it before proceeding. Without
    a barrier, rank 1 could race ahead and try to read a half-written file.

    Cheap but not free — barrier is a NCCL collective. Don't sprinkle them
    inside hot loops.

    HINTS
    -----
    - if not dist.is_initialized(): return
    - dist.barrier()
    """
    # TODO: implement.
    raise NotImplementedError


def all_reduce_mean(tensor):
    """All-reduce a tensor across ranks and divide by world_size.

    Used for logging — when you want the displayed loss to be the average
    across all ranks, not just rank 0's view.

    HINTS
    -----
    - import torch.distributed as dist
    - if not dist.is_initialized(): return tensor
    - dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    - tensor /= dist.get_world_size()
    - return tensor

    GOTCHAS
    -------
    - all_reduce is IN-PLACE. The argument is mutated. Pass a clone if
      you need to preserve the original.
    - Float tensors only. For int tensors you'd use ReduceOp.AVG which
      is supported in newer NCCL.
    """
    # TODO: implement.
    raise NotImplementedError


def make_weight_sync_group(
    trainer_ranks: list[int], rollout_ranks: list[int]
) -> Optional[object]:
    """Build a NCCL subgroup spanning trainer + rollout ranks.

    THEORY
    ------
    See module docstring §"THE WEIGHT SYNC SUBGROUP". This is the
    one-time setup. The returned ProcessGroup is passed to
    `dist.broadcast(..., group=...)` calls inside `vllm_worker.update_weights`.

    Build this AFTER init_distributed() but BEFORE the first weight sync.

    GOTCHAS
    -------
    - dist.new_group must be called BY ALL RANKS IN THE WORLD, even ranks
      not in the new group. This is a common footgun: if you only call it
      on the ranks that will be in the subgroup, the others hang at the
      next collective.
    - If trainer_ranks and rollout_ranks overlap (colocated mode), this
      function should return None — there's no inter-process sync needed,
      just an in-place copy in the same process.
    - Returned group is opaque (a ProcessGroup object). Pass it through
      to vllm_worker as-is.

    HINTS
    -----
    - if set(trainer_ranks) & set(rollout_ranks):
          return None  # colocated, no subgroup needed
    - all_ranks = sorted(set(trainer_ranks) | set(rollout_ranks))
    - import torch.distributed as dist
    - return dist.new_group(ranks=all_ranks, backend="nccl")

    REFERENCES
    ----------
    - https://pytorch.org/docs/stable/distributed.html#torch.distributed.new_group
    - OpenRLHF / verl / slime weight-sync references in vllm_worker.py
    """
    # TODO: implement.
    raise NotImplementedError
