"""Compare per-step time, throughput, cumulative wall, and training reward across N runs.

Loads N runs (paths can be either run-dir or specific stats_*.pt) and overlays
them on the same axes. Useful for sweeping a single dimension — e.g.
weight_sync_method (disk-nvme vs disk-tmpfs vs ipc), or DDP world size, or
any A/B/... change to training infrastructure. Two-run comparisons are just
N=2.

Usage:
    python -m tests.compare_sync_perf \\
        --runs   runs/exp_disk_nvme  runs/exp_disk_shm  runs/exp_ipc \\
        --labels "disk (NVMe)"       "disk (/dev/shm)"  "IPC"        \\
        --out    runs/sync_compare_3way.png

If a run dir contains multiple `stats_*.pt` (e.g. multiple sessions), the
newest one is used. The first run is treated as the baseline for the
"% vs baseline" wall-clock numbers printed to the console.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch


def _load_stats(path: str):
    """Accept a run-dir (newest stats_*.pt picked) OR a specific .pt file."""
    if os.path.isfile(path):
        return torch.load(path, weights_only=False)
    files = sorted(glob.glob(os.path.join(path, "stats_*.pt")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no stats_*.pt in {path}")
    return torch.load(files[-1], weights_only=False)


def _arr(stats, name):
    if isinstance(stats, dict):
        return np.asarray(stats[name], dtype=float)
    return np.asarray(getattr(stats, name), dtype=float)


def _eval(stats, name):
    if isinstance(stats, dict):
        return list(stats[name])
    return list(getattr(stats, name))


def _smooth(x, window=5):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    pad = window // 2
    padded = np.concatenate([np.full(pad, x[0]), x, np.full(window - pad - 1, x[-1])])
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="run-dir paths or stats_*.pt paths, in plot order (first = baseline)")
    p.add_argument("--labels", nargs="+", required=True,
                   help="display label per run (must match --runs length)")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--title", default="training-run comparison",
                   help="figure title")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if len(args.runs) != len(args.labels):
        raise SystemExit(
            f"--runs has {len(args.runs)} entries, --labels has {len(args.labels)}; must match"
        )

    runs = [_load_stats(r) for r in args.runs]

    series = []
    for stats, label in zip(runs, args.labels):
        steps = _arr(stats, "steps")
        st = _arr(stats, "step_time")
        tps = _arr(stats, "tokens_per_sec")
        rw = _arr(stats, "rewards")
        eval_steps = _eval(stats, "eval_steps")
        eval_acc = _eval(stats, "eval_accuracy")
        series.append({
            "label": label,
            "steps": steps,
            "step_time": st,
            "tokens_per_sec": tps,
            "rewards": rw,
            "eval_steps": eval_steps,
            "eval_acc": eval_acc,
            "total": float(np.sum(st)),
            "step_avg": float(np.mean(st)),
        })

    print(f"\n=== {args.title} ===")
    baseline = series[0]
    for s in series:
        delta = (baseline["total"] - s["total"]) / baseline["total"] * 100 if baseline["total"] > 0 else 0.0
        marker = " (baseline)" if s is baseline else f"  [{delta:+.1f}% vs {baseline['label']}]"
        print(f"  {s['label']:<24} {len(s['step_time']):>4} steps  total {s['total']:>6.1f}s  step_avg {s['step_avg']:.2f}s{marker}")
    print()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    # 1. Per-step wall time
    ax = axes[0, 0]
    for s, color in zip(series, palette):
        ax.plot(s["steps"], s["step_time"], color=color, alpha=0.3)
        ax.plot(s["steps"], _smooth(s["step_time"]), color=color, lw=2,
                label=f"{s['label']} (mean {s['step_avg']:.2f}s)")
    ax.set_title("step time")
    ax.set_xlabel("step")
    ax.set_ylabel("seconds / step")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # 2. Throughput
    ax = axes[0, 1]
    for s, color in zip(series, palette):
        ax.plot(s["steps"], s["tokens_per_sec"], color=color, alpha=0.3)
        ax.plot(s["steps"], _smooth(s["tokens_per_sec"]), color=color, lw=2, label=s["label"])
    ax.set_title("tokens / sec")
    ax.set_xlabel("step")
    ax.set_ylabel("tok/s")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. Cumulative wall time
    ax = axes[1, 0]
    for s, color in zip(series, palette):
        ax.plot(s["steps"], np.cumsum(s["step_time"]), color=color, lw=2,
                label=f"{s['label']} ({s['total']:.0f}s)")
    ax.set_title(f"cumulative wall time — baseline: {baseline['label']}")
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative seconds")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4. Training reward + eval accuracy markers
    ax = axes[1, 1]
    has_eval = any(s["eval_steps"] for s in series)
    ax2 = ax.twinx() if has_eval else None
    for s, color in zip(series, palette):
        ax.plot(s["steps"], _smooth(s["rewards"]), color=color, lw=2, label=f"{s['label']} reward")
        if ax2 is not None and s["eval_steps"]:
            ax2.scatter(s["eval_steps"], s["eval_acc"], color=color, marker="o", s=60,
                        edgecolors="k", linewidth=0.5, zorder=5)
    ax.set_title("training reward (smoothed) + eval accuracy markers")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left", fontsize=9)
    if ax2 is not None:
        ax2.set_ylabel("eval accuracy (%)")
    ax.grid(alpha=0.3)

    fig.suptitle(args.title, y=0.995)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
