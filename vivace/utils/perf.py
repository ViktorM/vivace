"""Performance measurement helpers.

Wall-clock timers with proper CUDA synchronization. Used by the trainer
to populate rollout/train phase timings and throughput stats on
`TrainingStats`. See `vivace/utils/stats.py` for the fields that store them.

WHY THE SYNC
------------
CUDA kernels are asynchronous, so `t0 = time.time(); do_gpu_work(); dt = time.time() - t0`
measures kernel LAUNCH time — microseconds for a forward pass — not the work.
`torch.cuda.synchronize()` blocks until all queued GPU work has finished; syncing on
enter AND exit pins the measurement: start = "everything before now is done",
stop = "everything we did is done". The sync is skipped without CUDA (CPU tests),
so the Timer is safe to use unconditionally.
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
