"""N-algorithm comparison figure for a 3-seed benchmark sweep (gsm8k or math).

Two panels:
  (left)  --metric (gsm8k | math500) eval accuracy vs step — one line per algo
          (seed mean), band = ±1 std; base pass@8 / greedy as reference lines.
  (right) final accuracy bar chart, ±std across seeds.

Pulls wandb groups <--group-prefix><--date>_<algo> (default overnight-1.5b-gsm8k-).

Run: .venv/bin/python tools/plot_benchmark.py [--date 20260609_0140] [--out FILE.png]
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALGOS = ["grpo", "dr_grpo", "gspo", "dapo", "dapo_ep2", "cispo"]
COLORS = {"grpo": "#888", "dr_grpo": "#1f77b4", "gspo": "#2ca02c",
          "dapo": "#d62728", "dapo_ep2": "#ff9896", "cispo": "#9467bd"}
# (pass@1, pass@8) base references from tools/pass_at_k.py, Qwen2.5-1.5B
BASE_PASS = {"gsm8k": (5.08, 27.37), "math500": (4.40, 24.20)}


def _agg_curve(histories: list[list[tuple[int, float]]]):
    """Given per-seed [(step, acc), ...], return (steps, mean, std) on common steps."""
    from collections import defaultdict
    by_step = defaultdict(list)
    for h in histories:
        for step, acc in h:
            if acc is not None and not (isinstance(acc, float) and math.isnan(acc)):
                by_step[step].append(acc)
    steps = sorted(by_step)
    mean = [sum(by_step[s]) / len(by_step[s]) for s in steps]
    std = [(sum((x - m) ** 2 for x in by_step[s]) / (len(by_step[s]) - 1)) ** 0.5
           if len(by_step[s]) > 1 else 0.0 for s, m in zip(steps, mean)]
    return steps, mean, std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260609_0140")
    ap.add_argument("--group-prefix", default="overnight-1.5b-gsm8k-",
                    help="wandb group = <prefix><date>_<algo>")
    ap.add_argument("--title-tag", default="v1.0")
    ap.add_argument("--metric", default="gsm8k", choices=list(BASE_PASS),
                    help="eval env whose accuracy_pct is plotted")
    ap.add_argument("--train-env", default="gsm8k", help="for the title only")
    ap.add_argument("--project", default="vivace")
    ap.add_argument("--out", default="docs/figures/v1_1.5b_gsm8k_5algo.png")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import wandb
    api = wandb.Api()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    finals_mean, finals_std, labels = [], [], []
    for algo in ALGOS:
        runs = list(api.runs(args.project,
                             filters={"group": f"{args.group_prefix}{args.date}_{algo}"}))
        if not runs:
            continue
        histories, finals = [], []
        for r in runs:
            h = []
            for row in r.scan_history(keys=["_step", f"eval/{args.metric}/accuracy_pct"]):
                acc = row.get(f"eval/{args.metric}/accuracy_pct")
                if acc is not None:
                    h.append((row["_step"], acc))
            if h:
                histories.append(h)
                finals.append(h[-1][1])
        if not histories:
            continue
        steps, mean, std = _agg_curve(histories)
        c = COLORS[algo]
        axL.plot(steps, mean, label=algo, color=c, lw=2)
        axL.fill_between(steps, [m - s for m, s in zip(mean, std)],
                         [m + s for m, s in zip(mean, std)], color=c, alpha=0.15)
        fm = sum(finals) / len(finals)
        fs = (sum((x - fm) ** 2 for x in finals) / (len(finals) - 1)) ** 0.5 if len(finals) > 1 else 0
        finals_mean.append(fm); finals_std.append(fs); labels.append(algo)

    # Left: learning curves + base reference
    p1, p8 = BASE_PASS[args.metric]
    axL.axhline(p8, ls="--", color="gray", lw=1, label=f"base pass@8 ({p8:.0f}%)")
    axL.axhline(p1, ls=":", color="gray", lw=1, label=f"base greedy ({p1:.0f}%)")
    axL.set_xlabel("training step"); axL.set_ylabel(f"{args.metric} accuracy (%)")
    axL.set_title(f"1.5B / {args.metric} — eval accuracy (mean ± std, 3 seeds)")
    axL.legend(fontsize=8, loc="lower right"); axL.grid(alpha=0.3)

    # Right: final accuracy bars with error bars, sorted desc
    order = sorted(range(len(labels)), key=lambda i: -finals_mean[i])
    labels = [labels[i] for i in order]
    finals_mean = [finals_mean[i] for i in order]
    finals_std = [finals_std[i] for i in order]
    bars = axR.bar(labels, finals_mean, yerr=finals_std, capsize=5,
                   color=[COLORS[a] for a in labels], alpha=0.85)
    axR.axhline(p8, ls="--", color="gray", lw=1)
    axR.set_ylabel(f"final {args.metric} accuracy (%)")
    axR.set_title("final accuracy (3-seed mean ± std)")
    axR.set_ylim(min(finals_mean) - 5, max(finals_mean) + 4)
    for b, m, s in zip(bars, finals_mean, finals_std):
        axR.text(b.get_x() + b.get_width() / 2, m + s + 0.3, f"{m:.1f}",
                 ha="center", va="bottom", fontsize=9)
    axR.tick_params(axis="x", rotation=20); axR.grid(alpha=0.3, axis="y")

    fig.suptitle(f"vivace {args.title_tag} — {len(labels)}-config RL comparison "
                 f"(Qwen2.5-1.5B, train={args.train_env}, LoRA r=16)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
