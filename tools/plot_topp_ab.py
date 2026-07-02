"""DAPO top_p 0.95 vs 1.0 A/B figure (1.5B / gsm8k, 3 seeds per arm).

Two panels:
  (left)  gsm8k eval accuracy vs step — mean ± std per arm.
  (right) per-seed paired bars + 3-seed mean ± std per arm.

Run: .venv/bin/python tools/plot_topp_ab.py [--date 20260610_0210] [--out FILE.png]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_benchmark import _agg_curve  # noqa: E402

ARMS = {"0.95": "#1f77b4", "1.0": "#d62728"}
SEEDS = [7, 13, 42]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260610_0210")
    ap.add_argument("--project", default="vivace")
    ap.add_argument("--out", default="docs/figures/topp_ab_dapo_1.5b_gsm8k.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import wandb
    api = wandb.Api()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    finals = {}  # arm -> {seed: final_acc}
    for arm, color in ARMS.items():
        group = f"topp-ab-dapo-1.5b-gsm8k-{args.date}_tp{arm}"
        runs = list(api.runs(args.project, filters={"group": group}))
        histories = []
        finals[arm] = {}
        for r in runs:
            h = [(row["_step"], row["eval/gsm8k/accuracy_pct"])
                 for row in r.scan_history(keys=["_step", "eval/gsm8k/accuracy_pct"])
                 if row.get("eval/gsm8k/accuracy_pct") is not None]
            if h:
                histories.append(h)
                finals[arm][r.config.get("seed")] = h[-1][1]
        steps, mean, std = _agg_curve(histories)
        axL.plot(steps, mean, label=f"top_p={arm}", color=color, lw=2)
        axL.fill_between(steps, [m - s for m, s in zip(mean, std)],
                         [m + s for m, s in zip(mean, std)], color=color, alpha=0.15)

    axL.set_xlabel("training step"); axL.set_ylabel("gsm8k accuracy (%)")
    axL.set_title("DAPO rollout top_p A/B — eval accuracy (mean ± std, 3 seeds)")
    axL.legend(loc="lower right"); axL.grid(alpha=0.3)

    # Right: paired per-seed bars, then the 3-seed mean ± std pair
    w = 0.35
    xticks, xlabels = [], []
    for i, seed in enumerate(SEEDS):
        for j, (arm, color) in enumerate(ARMS.items()):
            v = finals[arm].get(seed)
            if v is None:
                continue
            axR.bar(i + (j - 0.5) * w, v, w, color=color, alpha=0.85)
            axR.text(i + (j - 0.5) * w, v + 0.08, f"{v:.1f}", ha="center",
                     va="bottom", fontsize=8)
        xticks.append(i); xlabels.append(f"seed {seed}")
    for j, (arm, color) in enumerate(ARMS.items()):
        vals = list(finals[arm].values())
        m = sum(vals) / len(vals)
        s = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        axR.bar(len(SEEDS) + 0.25 + (j - 0.5) * w, m, w, yerr=s, capsize=4,
                color=color, alpha=0.85)
        axR.text(len(SEEDS) + 0.25 + (j - 0.5) * w, m + s + 0.08, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    xticks.append(len(SEEDS) + 0.25); xlabels.append("mean ± std")
    axR.set_xticks(xticks); axR.set_xticklabels(xlabels)
    allv = [v for d in finals.values() for v in d.values()]
    axR.set_ylim(min(allv) - 2, max(allv) + 1.5)
    axR.set_ylabel("final gsm8k accuracy (%)")
    axR.set_title("final accuracy by seed")
    axR.grid(alpha=0.3, axis="y")

    fig.suptitle("DAPO top_p 0.95 vs 1.0 (Qwen2.5-1.5B, gsm8k, LoRA r=16) — 0.95 wins, used for v2 sweep",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
