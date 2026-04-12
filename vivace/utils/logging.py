"""Logging helpers — wandb wrapper + console formatter.

wandb is optional. If `wandb` isn't installed or `WANDB_MODE=disabled`
in the env, all wandb calls turn into no-ops and training works exactly
the same. The console logger is always available.

Main-process gating is the caller's responsibility — pass
`is_main=True` only on rank 0 in distributed runs.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any


_WANDB_RUN = None
_WANDB_OK = False


def init_wandb(
    cfg: Any,
    project: str = "vivace",
    run_name: str | None = None,
    is_main: bool = True,
) -> None:
    """Initialise a wandb run if available + enabled + main process.

    `cfg` may be a dataclass (turned into a dict) or a plain dict.
    """
    global _WANDB_RUN, _WANDB_OK
    if not is_main:
        return
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return
    try:
        import wandb
    except ImportError:
        return

    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    _WANDB_RUN = wandb.init(project=project, name=run_name, config=cfg_dict)
    _WANDB_OK = True


def log_metrics(metrics: dict[str, float], step: int, is_main: bool = True) -> None:
    """Log metrics to wandb (no-op if disabled / not main)."""
    if not is_main or not _WANDB_OK:
        return
    import wandb

    wandb.log(metrics, step=step)


def finish_wandb(is_main: bool = True) -> None:
    """Close the wandb run if active."""
    global _WANDB_OK
    if not is_main or not _WANDB_OK:
        return
    import wandb

    wandb.finish()
    _WANDB_OK = False


class ConsoleLogger:
    """Cheap step-rate console logger.

    Per-step print line:
        [label] Step 0042 | loss=0.123 reward=1.456 kl=0.012 | 12s
    """

    def __init__(self, label: str, is_main: bool = True):
        self.label = label
        self.is_main = is_main
        self.t0 = time.time()

    # Keys to show in the console. Everything else is logged to wandb/stats only.
    CONSOLE_KEYS = ("loss", "reward", "kl", "clip_frac", "grad_norm", "entropy", "length_mean")

    def log(self, step: int, perf: dict | None = None, **metrics: float) -> None:
        if not self.is_main:
            return
        bits = " ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if k in self.CONSOLE_KEYS
        )
        elapsed = int(time.time() - self.t0)
        perf_str = ""
        if perf:
            tok_s = perf.get("tokens_per_sec", 0)
            step_t = perf.get("step_time", 0)
            perf_str = f" | {tok_s:.0f} tok/s {step_t:.1f}s/step"
        print(f"  [{self.label}] Step {step:04d} | {bits}{perf_str} | {elapsed}s")
