"""Wall-clock figures for the gsm8k 3-seed sweeps.

Figure 1 (v2 only): per-algo per-step time breakdown (rollout / train / other)
and total run wall-clock per seed.
Figure 2 (v1 vs v2): optimization impact — total run minutes and per-step
breakdown, last night's pre-optimization runs vs today's.

Run: .venv/bin/python tools/plot_wallclock.py [--v1-date 20260609_0140] [--v2-date 20260610_1029]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALGOS = ["grpo", "dr_grpo", "gspo", "dapo", "cispo"]
COLORS = {"grpo": "#888", "dr_grpo": "#1f77b4", "gspo": "#2ca02c",
          "dapo": "#d62728", "cispo": "#9467bd"}
PERF_KEYS = ["perf/step_time", "perf/rollout_time", "perf/train_time"]


def fetch(api, project, group):
    """Per-run dict: seed, total_min, mean rollout/train/other per step."""
    out = []
    for r in api.runs(project, filters={"group": group}):
        if r.state == "running":
            continue
        hist = [row for row in r.scan_history(keys=PERF_KEYS)]
        if not hist:
            continue
        n = len(hist)
        roll = sum(h["perf/rollout_time"] for h in hist) / n
        train = sum(h["perf/train_time"] for h in hist) / n
        step = sum(h["perf/step_time"] for h in hist) / n
        out.append({
            "seed": r.config.get("seed"),
            "total_min": r.summary.get("_runtime", 0) / 60,
            "rollout": roll, "train": train, "other": max(step - roll - train, 0),
            "step": step,
        })
    return out


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-date", default="20260609_0140")
    ap.add_argument("--v2-date", default="20260610_1029")
    ap.add_argument("--project", default="vivace")
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import wandb
    api = wandb.Api()

    v1 = {a: fetch(api, args.project, f"overnight-1.5b-gsm8k-{args.v1_date}_{a}") for a in ALGOS}
    v2 = {a: fetch(api, args.project, f"gsm8k-v2-{args.v2_date}_{a}") for a in ALGOS}
    algos2 = [a for a in ALGOS if v2[a]]

    # ---- Figure 1: v2 only ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(algos2))
    for part, color, bottom_of in [("rollout", "#1f77b4", None),
                                   ("train", "#ff7f0e", "rollout"),
                                   ("other", "#bbb", "rollout+train")]:
        vals = [mean_std([r[part] for r in v2[a]])[0] for a in algos2]
        bottoms = [0.0] * len(algos2)
        if bottom_of:
            for i, a in enumerate(algos2):
                bottoms[i] = sum(mean_std([r[p] for r in v2[a]])[0]
                                 for p in bottom_of.split("+"))
        axA.bar(x, vals, 0.6, bottom=bottoms, label=part, color=color, alpha=0.85)
    for i, a in enumerate(algos2):
        m, _ = mean_std([r["step"] for r in v2[a]])
        axA.text(i, m + 0.05, f"{m:.2f}s", ha="center", va="bottom", fontsize=9)
    axA.set_xticks(list(x)); axA.set_xticklabels(algos2, rotation=20)
    axA.set_ylabel("seconds per training step (mean over run)")
    axA.set_title("v2 per-step time breakdown"); axA.legend(); axA.grid(alpha=0.3, axis="y")

    w = 0.25
    for j, (seed, color) in enumerate(zip([7, 13, 42], ["#1f77b4", "#ff7f0e", "#2ca02c"])):
        vals, xs = [], []
        for i, a in enumerate(algos2):
            match = [r for r in v2[a] if r["seed"] == seed]
            if match:
                xs.append(i + (j - 1) * w); vals.append(match[0]["total_min"])
        axB.bar(xs, vals, w, label=f"seed {seed}", color=color, alpha=0.85)
    axB.set_xticks(list(x)); axB.set_xticklabels(algos2, rotation=20)
    axB.set_ylabel("total run wall-clock (min)")
    axB.set_title("v2 total wall-clock per run (200 steps + evals)")
    axB.legend(); axB.grid(alpha=0.3, axis="y")
    fig.suptitle(f"vivace v2 sweep wall-clock (Qwen2.5-1.5B, gsm8k, 2x4090, {args.v2_date})")
    fig.tight_layout()
    out1 = os.path.join(args.outdir, "v2_wallclock.png")
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    print(f"saved {out1}")

    # ---- Figure 2: v1 vs v2 ----
    both = [a for a in ALGOS if v1[a] and v2[a]]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(both))
    for j, (tag, data, color) in enumerate([("v1 (pre-opt)", v1, "#d62728"),
                                            ("v2 (optimized)", v2, "#2ca02c")]):
        ms = [mean_std([r["total_min"] for r in data[a]]) for a in both]
        axA.bar([i + (j - 0.5) * 0.35 for i in x], [m for m, _ in ms], 0.35,
                yerr=[s for _, s in ms], capsize=4, label=tag, color=color, alpha=0.85)
        for i, (m, s) in enumerate(ms):
            axA.text(i + (j - 0.5) * 0.35, m + s + 0.3, f"{m:.1f}", ha="center",
                     va="bottom", fontsize=8)
    axA.set_xticks(list(x)); axA.set_xticklabels(both, rotation=20)
    axA.set_ylabel("total run wall-clock (min, mean ± std over seeds)")
    axA.set_title("total run time"); axA.legend(); axA.grid(alpha=0.3, axis="y")

    for j, (tag, data) in enumerate([("v1", v1), ("v2", v2)]):
        bottoms = [0.0] * len(both)
        for part, color in [("rollout", "#1f77b4"), ("train", "#ff7f0e"), ("other", "#bbb")]:
            vals = [mean_std([r[part] for r in data[a]])[0] for a in both]
            axB.bar([i + (j - 0.5) * 0.35 for i in x], vals, 0.35, bottom=bottoms,
                    color=color, alpha=0.85 if j else 0.55,
                    label=part if j else None)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        for i, b in enumerate(bottoms):
            axB.text(i + (j - 0.5) * 0.35, b + 0.05, tag, ha="center", va="bottom",
                     fontsize=7, color="#555")
    axB.set_xticks(list(x)); axB.set_xticklabels(both, rotation=20)
    axB.set_ylabel("seconds per training step")
    axB.set_title("per-step breakdown (faded = v1, solid = v2)")
    axB.legend(); axB.grid(alpha=0.3, axis="y")
    fig.suptitle("Optimization impact: last night (v1) vs today (v2) — "
                 "no_sync, LoRA-only sync, fused AdamW, fast eval, real sleep")
    fig.tight_layout()
    out2 = os.path.join(args.outdir, "v1_vs_v2_wallclock.png")
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"saved {out2}")

    for a in both:
        m1, _ = mean_std([r["total_min"] for r in v1[a]])
        m2, _ = mean_std([r["total_min"] for r in v2[a]])
        print(f"{a:8s} v1={m1:5.1f}min  v2={m2:5.1f}min  speedup={m1/m2:.2f}x")


if __name__ == "__main__":
    main()
