"""View training plots from a saved stats file.

Usage:
    python -m vivace.scripts.plot runs/gspo_gsm8k_0.8b_lora/stats.pt
    python -m vivace.scripts.plot runs/gspo_gsm8k_0.8b_lora/stats.pt --save
    python -m vivace.scripts.plot runs/gspo_gsm8k_0.8b_lora/stats.pt --only perf
    python -m vivace.scripts.plot run1/stats.pt run2/stats.pt --compare
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vivace-plot", description="View training plots")
    p.add_argument("stats", nargs="+", help="path(s) to stats.pt file(s)")
    p.add_argument("--save", action="store_true", help="save PNGs instead of showing")
    p.add_argument("--only", choices=["stats", "perf", "health", "wallclock"], default=None,
                   help="show only one plot type")
    p.add_argument("--compare", action="store_true",
                   help="overlay multiple runs on the same plots")
    p.add_argument("--min-steps", type=int, default=0,
                   help="skip runs with fewer steps (filters out short test runs)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vivace.utils.stats import (
        plot_stats, plot_perf, plot_health, plot_wallclock,
        plot_comparison, plot_perf_comparison, plot_wallclock_comparison,
    )

    # Load stats
    stats_list = []
    for path in args.stats:
        s = torch.load(path, weights_only=False)
        if len(s.steps) < args.min_steps:
            print(f"Skipped: {path}  ({len(s.steps)} steps < --min-steps {args.min_steps})")
            continue
        stats_list.append((path, s))
        print(f"Loaded: {path}  ({s.method}, {len(s.steps)} steps)")

    if not stats_list:
        print("No stats files to plot.")
        return

    if args.compare and len(stats_list) > 1:
        # Comparison mode: overlay multiple runs
        stats_dict = {}
        for path, s in stats_list:
            # Label: algo method + step count (concise for legends)
            label = f"{s.method} {len(s.steps)}st"
            # Deduplicate
            if label in stats_dict:
                run_dir = os.path.basename(os.path.dirname(path))
                label = f"{label} [{run_dir}]"
            stats_dict[label] = s

        plots = []
        if args.only is None or args.only == "stats":
            plot_comparison(stats_dict)
            plots.append("comparison")
        if args.only is None or args.only == "perf":
            plot_perf_comparison(stats_dict)
            plots.append("perf_comparison")
        if args.only is None or args.only == "wallclock":
            plot_wallclock_comparison(stats_dict)
            plots.append("wallclock_comparison")

        if args.save:
            out_dir = os.path.dirname(args.stats[0]) or "."
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for fig, name in zip(figs, plots):
                path = os.path.join(out_dir, f"plot_{name}.png")
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved: {path}")
            plt.close("all")
        else:
            plt.show()
    else:
        # Single run mode (or multiple runs shown separately)
        for path, s in stats_list:
            title = f"{s.method} — {len(s.steps)} steps"
            out_dir = os.path.dirname(path) or "."

            plot_fns = []
            if args.only is None or args.only == "stats":
                plot_fns.append((plot_stats, "stats"))
            if args.only is None or args.only == "perf":
                plot_fns.append((plot_perf, "perf"))
            if args.only is None or args.only == "health":
                plot_fns.append((plot_health, "health"))
            if args.only is None or args.only == "wallclock":
                plot_fns.append((plot_wallclock, "wallclock"))

            for fn, name in plot_fns:
                fn(s, title=title)
                if args.save:
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(out_dir, f"plot_{name}_{base}.png")
                    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                    print(f"Saved: {out_path}")
                    plt.close()

        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
