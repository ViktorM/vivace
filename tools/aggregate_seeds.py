"""Aggregate seed-replicate runs in a wandb group → mean ± std table + CSV.

Usage:
  .venv/bin/python tools/aggregate_seeds.py <WANDB_GROUP> [--out FILE.csv]
                                            [--metrics m1,m2,...]
                                            [--by KEY1,KEY2,...]

Defaults:
  - Project: 'vivace' (override with --project)
  - Metrics: eval/{env}/accuracy_pct + format_rate_pct + avg_length_tokens + cap_rate_pct
  - Grouping: rl.lr, rl.optim_epochs, rl.temperature, rl.kl_coef
              (one row per unique combination)

Output:
  - Stdout: table of mean ± std per cell, sorted by accuracy descending
  - Optional --out: CSV with one row per cell × metric
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from typing import Any


def _stats(values: list[float]) -> tuple[float, float, int]:
    """Return (mean, sample_std (ddof=1), n) — skips NaN."""
    xs = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var), n


def _get(d: dict | object, dotted: str, default=None) -> Any:
    """Read dotted key from dict-or-object summary/config (e.g. 'rl.lr')."""
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
        if cur is default:
            return default
    return cur


def _eval_metric_keys(summary: dict) -> list[str]:
    """Auto-discover eval/{env}/{metric} keys present in this run's summary."""
    out = set()
    pat = re.compile(r"^eval/[^/]+/(accuracy_pct|format_rate_pct|avg_length_tokens|cap_rate_pct|avg_reward)$")
    for k in summary.keys():
        if pat.match(k):
            out.add(k)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", help="wandb group name (or substring; regex match supported)")
    ap.add_argument("--project", default="vivace")
    ap.add_argument("--metrics", default=None,
                    help="comma-separated list of summary keys. Default: all "
                         "eval/{env}/{accuracy_pct,format_rate_pct,avg_length_tokens,"
                         "cap_rate_pct,avg_reward} keys present in the runs.")
    ap.add_argument("--by", default="rl.lr,rl.optim_epochs,rl.temperature,rl.kl_coef",
                    help="comma-separated dotted config keys to group on")
    ap.add_argument("--out", default=None, help="write CSV to this path")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run: uv pip install wandb", file=sys.stderr)
        return 2

    api = wandb.Api()
    # Accept exact group OR regex
    if any(c in args.group for c in r".*+?[](){}|\^$"):
        runs = api.runs(args.project, filters={"group": {"$regex": args.group}})
    else:
        runs = api.runs(args.project, filters={"group": args.group})
    runs = list(runs)
    if not runs:
        print(f"No runs found in project={args.project} group={args.group!r}", file=sys.stderr)
        return 1
    print(f"Found {len(runs)} runs in group(s) matching {args.group!r}")

    # Auto-discover metrics if not specified.
    if args.metrics:
        metrics = args.metrics.split(",")
    else:
        seen = set()
        for r in runs:
            seen.update(_eval_metric_keys(dict(r.summary)))
        if not seen:
            print("No eval/{env}/{...} keys found in any run.summary", file=sys.stderr)
            return 1
        metrics = sorted(seen)

    by_keys = args.by.split(",")

    # Bucket runs by cell.
    cells: dict[tuple, list[Any]] = defaultdict(list)
    for r in runs:
        cfg = dict(r.config)
        key = tuple(_get(cfg, k, "?") for k in by_keys)
        cells[key].append(r)

    # Compute per-cell mean ± std per metric.
    rows = []
    for cell, cell_runs in cells.items():
        row: dict[str, Any] = dict(zip(by_keys, cell_runs[0].config and [_get(dict(cell_runs[0].config), k, "?") for k in by_keys]))
        row["n_seeds"] = len(cell_runs)
        for m in metrics:
            vals = [_get(dict(r.summary), m, float("nan")) for r in cell_runs]
            mean, std, n = _stats([float(v) if v is not None else float("nan") for v in vals])
            row[m + "/mean"] = mean
            row[m + "/std"] = std
            row[m + "/n"] = n
        rows.append(row)

    # Sort by first metric mean descending (typically eval/.../accuracy_pct).
    primary = metrics[0] + "/mean"
    rows.sort(key=lambda r: (-r[primary] if r[primary] == r[primary] else 0))

    # Print table.
    print()
    hdr = by_keys + ["n_seeds"] + [f"{m} (mean ± std)" for m in metrics]
    widths = [max(len(h), 12) for h in hdr]
    print("  ".join(f"{h:<{w}}" for h, w in zip(hdr, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        cells_str = [str(row[k]) for k in by_keys]
        cells_str.append(str(row["n_seeds"]))
        for m in metrics:
            mean, std = row[m + "/mean"], row[m + "/std"]
            s = f"{mean:6.2f} ± {std:5.2f}" if mean == mean else "n/a"
            cells_str.append(s)
        print("  ".join(f"{s:<{w}}" for s, w in zip(cells_str, widths)))

    # CSV output.
    if args.out:
        cols = by_keys + ["n_seeds"]
        for m in metrics:
            cols += [m + "/mean", m + "/std", m + "/n"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"\nWrote {len(rows)} rows × {len(cols)} cols → {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
