"""Training statistics + plotting.

Used by the trainer to log per-step metrics and by experiment scripts
to render side-by-side comparisons.

Plotting uses matplotlib (no seaborn dependency). Plots are not shown
automatically — call `plot_stats(stats, "label")` after training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TrainingStats:
    """All time-series metrics from a training run.

    Each list grows by one entry per `log()` call. The trainer is expected
    to call `log()` once per optimizer step.
    """

    method: str
    steps: list[int] = field(default_factory=list)
    # --- learning dynamics ---
    losses: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    kl_values: list[float] = field(default_factory=list)
    clip_fracs: list[float] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    format_rates: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    alive_rates: list[float] = field(default_factory=list)
    # --- distribution diagnostics (collapse / runaway detection) ---
    reward_std: list[float] = field(default_factory=list)      # std of rewards across B*G — drops to 0 on mode collapse
    reward_max: list[float] = field(default_factory=list)      # any sample solving the problem?
    reward_min: list[float] = field(default_factory=list)       # is the floor rising?
    length_mean: list[float] = field(default_factory=list)      # mean response length (response tokens)
    length_std: list[float] = field(default_factory=list)       # drops to 0 alongside reward_std → identical outputs
    length_max: list[float] = field(default_factory=list)       # = max_new_tokens every time → model never stops
    length_min: list[float] = field(default_factory=list)       # → 1 means model learned to output one token and stop
    cap_rates: list[float] = field(default_factory=list)        # fraction of rollouts hitting max_new_tokens — leading indicator of length-gaming
    advantage_std: list[float] = field(default_factory=list)    # ~0 means no learning signal (all samples equally good/bad)
    # --- performance (wall-clock + throughput) ---
    # Populated by the trainer using vivace.utils.perf.Timer around each phase.
    # Filling them is optional — plot_perf tolerates empty fields so old runs
    # still plot fine.
    rollout_time: list[float] = field(default_factory=list)    # seconds in rollout_phase
    train_time: list[float] = field(default_factory=list)      # seconds in train_phase
    step_time: list[float] = field(default_factory=list)       # total wall-clock per step
    rollout_tokens: list[int] = field(default_factory=list)    # response tokens generated
    rollout_samples: list[int] = field(default_factory=list)   # sequences generated (B*G)
    tokens_per_sec: list[float] = field(default_factory=list)  # rollout_tokens / rollout_time
    samples_per_sec: list[float] = field(default_factory=list) # rollout_samples / rollout_time
    # --- eval checkpoints (sparse — only populated at eval_interval) ---
    eval_steps: list[int] = field(default_factory=list)
    eval_accuracy: list[float] = field(default_factory=list)
    eval_format_rate: list[float] = field(default_factory=list)
    eval_reward: list[float] = field(default_factory=list)

    def log(self, step: int, **kw) -> None:
        self.steps.append(step)
        for k, v in kw.items():
            if hasattr(self, k):
                getattr(self, k).append(v)


def smooth(vals: Iterable[float], w: float = 0.9) -> list[float]:
    """EMA smoothing for plot overlays. w is the decay (0..1, higher = smoother).

    TODO: try replacing the fixed w with an adaptive
    factor that gets smoother as len(vals) grows. Useful for short runs
    where w=0.9 oversmooths.
    """
    vals = list(vals)
    if not vals:
        return []
    out, last = [], vals[0]
    for v in vals:
        last = w * last + (1 - w) * v
        out.append(last)
    return out


def plot_stats(stats: TrainingStats, title: str = "") -> None:
    """Render a 3x3 grid of metric curves with raw + smoothed traces."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle(title or stats.method, fontsize=13, fontweight="bold")
    plots = [
        ("rewards", "Reward", "tab:green"),
        ("losses", "Loss", "tab:red"),
        ("kl_values", "KL", "tab:blue"),
        ("clip_fracs", "Clip Frac", "tab:cyan"),
        ("lrs", "Learning Rate", "tab:olive"),
        ("lengths", "Resp Length", "tab:orange"),
        ("format_rates", "Format Rate", "tab:purple"),
        ("grad_norms", "Grad Norm", "tab:brown"),
        ("entropies", "Entropy", "tab:gray"),
    ]
    for ax, (attr, label, color) in zip(axes.flat, plots):
        vals = getattr(stats, attr, [])
        if vals:
            ax.plot(vals, alpha=0.3, color=color, linewidth=0.8)
            if len(vals) > 5:
                ax.plot(smooth(vals), color=color, linewidth=2)
            if attr == "lrs":
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, -3))
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(stats_dict: dict[str, TrainingStats]) -> None:
    """Overlay multiple TrainingStats on the same 3x3 grid.

    Legend shown once (top-right subplot only) to avoid clutter.
    """
    import matplotlib.pyplot as plt

    metrics = [
        ("rewards", "Reward"),
        ("losses", "Loss"),
        ("kl_values", "KL"),
        ("clip_fracs", "Clip Frac"),
        ("lrs", "Learning Rate"),
        ("lengths", "Length"),
        ("format_rates", "Format Rate"),
        ("grad_norms", "Grad Norm"),
        ("entropies", "Entropy"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(stats_dict)}
    for ax_idx, (ax, (attr, title)) in enumerate(zip(axes.flat, metrics)):
        for name, st in stats_dict.items():
            vals = getattr(st, attr, [])
            if vals:
                ax.plot(vals, alpha=0.15, color=colors[name])
                ax.plot(smooth(vals), label=name, color=colors[name], linewidth=2)
        if attr == "lrs":
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, -3))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    # Single legend in the top-right subplot
    axes.flat[2].legend(fontsize=8, loc="best")
    plt.suptitle("Training Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_perf(stats: TrainingStats, title: str = "") -> None:
    """Render a 2x3 grid of wall-clock + throughput metrics.

    Separate from `plot_stats` because performance is a different
    axis of comparison than learning dynamics — you want to see them
    side by side, not mixed into the same grid.

    Panels:
        (0,0) rollout_time + train_time overlaid (phase breakdown)
        (0,1) step_time (total wall-clock per step)
        (0,2) rollout_time as % of step_time (stacked view)
        (1,0) tokens_per_sec
        (1,1) samples_per_sec
        (1,2) cumulative tokens generated
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle(f"{title or stats.method} — performance", fontsize=13, fontweight="bold")

    # (0,0) phase breakdown
    ax = axes[0, 0]
    if stats.rollout_time:
        ax.plot(stats.rollout_time, alpha=0.3, color="tab:orange", linewidth=0.8)
        if len(stats.rollout_time) > 5:
            ax.plot(smooth(stats.rollout_time), color="tab:orange", linewidth=2, label="rollout")
    if stats.train_time:
        ax.plot(stats.train_time, alpha=0.3, color="tab:blue", linewidth=0.8)
        if len(stats.train_time) > 5:
            ax.plot(smooth(stats.train_time), color="tab:blue", linewidth=2, label="train")
    ax.set_title("Phase Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) total step wall-clock
    ax = axes[0, 1]
    if stats.step_time:
        ax.plot(stats.step_time, alpha=0.3, color="tab:green", linewidth=0.8)
        if len(stats.step_time) > 5:
            ax.plot(smooth(stats.step_time), color="tab:green", linewidth=2)
    ax.set_title("Step Time (s)")
    ax.grid(True, alpha=0.3)

    # (0,2) rollout fraction
    ax = axes[0, 2]
    if stats.rollout_time and stats.step_time:
        n = min(len(stats.rollout_time), len(stats.step_time))
        frac = [
            stats.rollout_time[i] / stats.step_time[i] if stats.step_time[i] > 0 else 0.0
            for i in range(n)
        ]
        ax.plot(frac, alpha=0.3, color="tab:purple", linewidth=0.8)
        if len(frac) > 5:
            ax.plot(smooth(frac), color="tab:purple", linewidth=2)
        ax.set_ylim(0, 1)
    ax.set_title("Rollout / Step Ratio")
    ax.grid(True, alpha=0.3)

    # (1,0) tokens/sec
    ax = axes[1, 0]
    if stats.tokens_per_sec:
        ax.plot(stats.tokens_per_sec, alpha=0.3, color="tab:red", linewidth=0.8)
        if len(stats.tokens_per_sec) > 5:
            ax.plot(smooth(stats.tokens_per_sec), color="tab:red", linewidth=2)
    ax.set_title("Tokens / sec (rollout)")
    ax.grid(True, alpha=0.3)

    # (1,1) samples/sec
    ax = axes[1, 1]
    if stats.samples_per_sec:
        ax.plot(stats.samples_per_sec, alpha=0.3, color="tab:cyan", linewidth=0.8)
        if len(stats.samples_per_sec) > 5:
            ax.plot(smooth(stats.samples_per_sec), color="tab:cyan", linewidth=2)
    ax.set_title("Samples / sec (rollout)")
    ax.grid(True, alpha=0.3)

    # (1,2) cumulative tokens (useful for "how many tokens did this run see")
    ax = axes[1, 2]
    if stats.rollout_tokens:
        cum = []
        total = 0
        for n in stats.rollout_tokens:
            total += n
            cum.append(total)
        ax.plot(cum, color="tab:brown", linewidth=2)
    ax.set_title("Cumulative Tokens")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_perf_comparison(stats_dict: dict[str, TrainingStats]) -> None:
    """Overlay perf curves from multiple runs — the hf vs vllm view.

    Same 2x3 layout as `plot_perf` but with one trace per run per panel.
    Legends show which line is which backend/config.
    """
    import matplotlib.pyplot as plt

    panels = [
        ("rollout_time", "Rollout Time (s)"),
        ("train_time", "Train Time (s)"),
        ("step_time", "Step Time (s)"),
        ("tokens_per_sec", "Tokens / sec"),
        ("samples_per_sec", "Samples / sec"),
        ("rollout_tokens", "Tokens per step"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(stats_dict)}
    for ax, (attr, title) in zip(axes.flat, panels):
        for name, st in stats_dict.items():
            vals = getattr(st, attr, [])
            if vals:
                ax.plot(vals, alpha=0.15, color=colors[name])
                ax.plot(smooth(vals), label=name, color=colors[name], linewidth=2)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Performance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_health(stats: TrainingStats, title: str = "") -> None:
    """Render a 2x3 grid of training health diagnostics.

    Separate from `plot_stats` because these are failure-mode detectors,
    not the usual learning curves. Look at these when something feels
    wrong — mode collapse, dead advantages, runaway lengths.

    Panels:
        (0,0) reward_std + advantage_std  — both dropping to 0 = mode collapse
        (0,1) reward_max / reward_min     — ceiling rising? floor rising?
        (0,2) format_rates                — model still producing valid output?
        (1,0) length_mean + length_std    — distribution of response lengths
        (1,1) length_max / length_min     — hitting max_tokens? one-token responses?
        (1,2) entropies                   — policy diversity (duplicate from main grid,
                                            but useful to see next to length/reward)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle(f"{title or stats.method} — health", fontsize=13, fontweight="bold")

    # (0,0) reward_std + advantage_std
    ax = axes[0, 0]
    if stats.reward_std:
        ax.plot(stats.reward_std, alpha=0.3, color="tab:green", linewidth=0.8)
        if len(stats.reward_std) > 5:
            ax.plot(smooth(stats.reward_std), color="tab:green", linewidth=2, label="reward std")
    if stats.advantage_std:
        ax.plot(stats.advantage_std, alpha=0.3, color="tab:orange", linewidth=0.8)
        if len(stats.advantage_std) > 5:
            ax.plot(smooth(stats.advantage_std), color="tab:orange", linewidth=2, label="advantage std")
    ax.set_title("Reward & Advantage Std")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) reward_max / reward_min
    ax = axes[0, 1]
    if stats.reward_max:
        ax.plot(stats.reward_max, alpha=0.3, color="tab:blue", linewidth=0.8)
        if len(stats.reward_max) > 5:
            ax.plot(smooth(stats.reward_max), color="tab:blue", linewidth=2, label="max")
    if stats.reward_min:
        ax.plot(stats.reward_min, alpha=0.3, color="tab:red", linewidth=0.8)
        if len(stats.reward_min) > 5:
            ax.plot(smooth(stats.reward_min), color="tab:red", linewidth=2, label="min")
    ax.set_title("Reward Max / Min")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) format_rates
    ax = axes[0, 2]
    if stats.format_rates:
        ax.plot(stats.format_rates, alpha=0.3, color="tab:purple", linewidth=0.8)
        if len(stats.format_rates) > 5:
            ax.plot(smooth(stats.format_rates), color="tab:purple", linewidth=2)
        ax.set_ylim(0, 1.05)
    ax.set_title("Format Rate")
    ax.grid(True, alpha=0.3)

    # (1,0) length_mean + length_std
    ax = axes[1, 0]
    if stats.length_mean:
        ax.plot(stats.length_mean, alpha=0.3, color="tab:cyan", linewidth=0.8)
        if len(stats.length_mean) > 5:
            ax.plot(smooth(stats.length_mean), color="tab:cyan", linewidth=2, label="mean")
    if stats.length_std:
        ax.plot(stats.length_std, alpha=0.3, color="tab:gray", linewidth=0.8)
        if len(stats.length_std) > 5:
            ax.plot(smooth(stats.length_std), color="tab:gray", linewidth=2, label="std")
    ax.set_title("Response Length (mean / std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) length_max / length_min
    ax = axes[1, 1]
    if stats.length_max:
        ax.plot(stats.length_max, alpha=0.3, color="tab:blue", linewidth=0.8)
        if len(stats.length_max) > 5:
            ax.plot(smooth(stats.length_max), color="tab:blue", linewidth=2, label="max")
    if stats.length_min:
        ax.plot(stats.length_min, alpha=0.3, color="tab:red", linewidth=0.8)
        if len(stats.length_min) > 5:
            ax.plot(smooth(stats.length_min), color="tab:red", linewidth=2, label="min")
    ax.set_title("Response Length (max / min)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) entropies
    ax = axes[1, 2]
    if stats.entropies:
        ax.plot(stats.entropies, alpha=0.3, color="tab:brown", linewidth=0.8)
        if len(stats.entropies) > 5:
            ax.plot(smooth(stats.entropies), color="tab:brown", linewidth=2)
    ax.set_title("Entropy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _cumulative_time_minutes(stats: TrainingStats) -> list[float]:
    """Build a cumulative wall-clock axis in minutes from step_time."""
    if not stats.step_time:
        return []
    cum = []
    total = 0.0
    for dt in stats.step_time:
        total += dt
        cum.append(total / 60.0)
    return cum


def plot_wallclock(stats: TrainingStats, title: str = "") -> None:
    """Key metrics plotted against wall-clock time (minutes) instead of step.

    Shows reward, loss, KL, and format rate over real time — useful for
    comparing runs with different step times (e.g. HF sampler vs vLLM).
    """
    import matplotlib.pyplot as plt

    time_min = _cumulative_time_minutes(stats)
    if not time_min:
        print("No step_time data — cannot plot wall-clock. Run with perf timing enabled.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{title or stats.method} — wall clock", fontsize=13, fontweight="bold")

    plots = [
        (axes[0, 0], "rewards", "Reward", "tab:green"),
        (axes[0, 1], "losses", "Loss", "tab:red"),
        (axes[1, 0], "kl_values", "KL", "tab:blue"),
        (axes[1, 1], "format_rates", "Format Rate", "tab:purple"),
    ]

    for ax, attr, label, color in plots:
        vals = getattr(stats, attr, [])
        n = min(len(vals), len(time_min))
        if n > 0:
            ax.plot(time_min[:n], vals[:n], alpha=0.3, color=color, linewidth=0.8)
            if n > 5:
                ax.plot(time_min[:n], smooth(vals[:n]), color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_wallclock_comparison(stats_dict: dict[str, TrainingStats]) -> None:
    """Overlay multiple runs on wall-clock x-axis — the HF vs vLLM view."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Wall Clock Comparison", fontsize=13, fontweight="bold")
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(stats_dict)}

    panels = [
        (axes[0, 0], "rewards", "Reward"),
        (axes[0, 1], "losses", "Loss"),
        (axes[1, 0], "kl_values", "KL"),
        (axes[1, 1], "format_rates", "Format Rate"),
    ]

    for ax, attr, label in panels:
        for name, st in stats_dict.items():
            time_min = _cumulative_time_minutes(st)
            vals = getattr(st, attr, [])
            n = min(len(vals), len(time_min))
            if n > 0:
                ax.plot(time_min[:n], vals[:n], alpha=0.15, color=colors[name])
                if n > 5:
                    ax.plot(time_min[:n], smooth(vals[:n]), label=name,
                            color=colors[name], linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
