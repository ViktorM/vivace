"""Performance measurement helpers.

Wall-clock timers with proper CUDA synchronization. Used by the trainer
to populate rollout/train phase timings and throughput stats on
`TrainingStats`. See `vivace/utils/stats.py` for the fields that store them.

WHY THE SYNC
------------
`torch.cuda` kernels are asynchronous. If you just time `t0 = time.time();
do_gpu_work(); dt = time.time() - t0`, you measure kernel LAUNCH time, not
actual work — because the kernels are still running on the GPU when you
read the clock. The timing comes out ridiculously small (microseconds for
a forward pass) and is useless for comparing backends.

`torch.cuda.synchronize()` blocks the CPU until all queued GPU work has
finished. Calling it on enter AND exit pins the measurement to real work:
start = "everything before now is done"; stop = "everything we did is done".

The sync is a no-op when CUDA isn't available (CPU tests), so this Timer
is safe to use unconditionally.
"""

from __future__ import annotations

import time

import torch


class Timer:
    """Context manager for wall-clock timing of (optionally CUDA) work.

    Usage:
        with Timer() as t:
            do_work()
        print(f"took {t.dt:.3f}s")

    After exiting, `t.dt` is the elapsed seconds as a float.

    Args:
        sync: if True (default), synchronize CUDA on enter and exit.
              Set False if you KNOW the block has no GPU work — saves
              a sync but gives you wrong numbers if you're wrong.
    """

    def __init__(self, sync: bool = True):
        self.sync = sync and torch.cuda.is_available()
        self.dt: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        if self.sync:
            torch.cuda.synchronize()
        self._t0 = time.time()
        return self

    def __exit__(self, *_) -> None:
        if self.sync:
            torch.cuda.synchronize()
        self.dt = time.time() - self._t0


def throughput(count: int | float, seconds: float) -> float:
    """Safe count/seconds division. Returns 0.0 if seconds is 0 or negative."""
    if seconds <= 0.0:
        return 0.0
    return float(count) / seconds
