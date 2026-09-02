"""PyTorch Profiler integration for vivace.

Provides a config-driven profiler that wraps a window of training steps,
exports a Chrome trace (.json) for visual inspection, and prints a
top-N kernel summary table to console.

Usage in YAML config:
    profiling:
      enabled: true
      start_step: 5
      end_step: 8

Then view the trace:
    chrome://tracing  or  https://ui.perfetto.dev
    File → Open → runs/<your_run>/profiling/trace_step5-8_<timestamp>.json

The profiler adds zero overhead when disabled (not even instantiated).
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.profiler import profile, ProfilerActivity, schedule


@dataclass
class ProfilingConfig:
    enabled: bool = False
    start_step: int = 5          # profiler warmup step; recording covers start_step+1 .. end_step-1
    end_step: int = 8            # exclusive (defaults record steps 6-7)
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False     # captures Python stack — expensive, off by default
    with_flops: bool = True
    output_dir: str | None = None  # defaults to {run_dir}/profiling/


def create_profiler(cfg: ProfilingConfig) -> profile | None:
    """Create a torch.profiler.profile context manager, or None if disabled.

    Schedule: wait = start_step (no overhead), warmup = 1 (step start_step, not
    recorded), active = end_step - start_step - 1 (steps start_step+1 .. end_step-1),
    repeat = 1. Defaults 5/8 record steps 6-7.

    The caller must call prof.step() at the end of each training step.
    """
    if not cfg.enabled:
        return None

    n_active = max(cfg.end_step - cfg.start_step - 1, 1)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof = profile(
        activities=activities,
        schedule=schedule(
            wait=cfg.start_step,
            warmup=1,
            active=n_active,
            repeat=1,
        ),
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        with_flops=cfg.with_flops,
    )
    return prof


def export_and_summarize(
    prof: profile,
    cfg: ProfilingConfig,
    run_dir: str,
    top_n: int = 20,
) -> str:
    """Export Chrome trace and write kernel summary. Returns the trace path.

    Files written to `{output_dir}/`:
      - trace_step{start}-{end}_{timestamp}.json   — Chrome/Perfetto trace
      - summary_step{start}-{end}_{timestamp}.txt  — kernel tables + metadata

    The timestamp prevents overwriting when re-profiling the same config.
    The summary is also echoed to stdout (compact version).
    """
    output_dir = cfg.output_dir or os.path.join(run_dir, "profiling")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"step{cfg.start_step}-{cfg.end_step}_{timestamp}"
    trace_path = os.path.join(output_dir, f"trace_{base}.json")
    summary_path = os.path.join(output_dir, f"summary_{base}.txt")

    prof.export_chrome_trace(trace_path)

    # Full summary goes to the file; stdout gets the compact block below.
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"{'=' * 80}")
        print(f"  Profiler Summary (steps {cfg.start_step}-{cfg.end_step})")
        print(f"  Captured: {timestamp}")
        print(f"  Run dir:  {run_dir}")
        print(f"  Config:   record_shapes={cfg.record_shapes}, "
              f"profile_memory={cfg.profile_memory}, "
              f"with_stack={cfg.with_stack}, with_flops={cfg.with_flops}")
        print(f"{'=' * 80}\n")

        if torch.cuda.is_available():
            print("Top CUDA kernels by total GPU time:")
            print(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=top_n,
            ))

        print("\nTop CPU operators by total CPU time:")
        print(prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=min(top_n, 10),
        ))

        if cfg.profile_memory and torch.cuda.is_available():
            print("\nTop CUDA memory allocators:")
            print(prof.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=min(top_n, 10),
            ))

        print(f"\nChrome trace: {trace_path}")
        print(f"Open with:    chrome://tracing  or  https://ui.perfetto.dev")

    full_summary = buf.getvalue()

    with open(summary_path, "w") as f:
        f.write(full_summary)

    # Compact stdout version — the tables can be huge; the file is the source of truth.
    print(f"\n{'=' * 80}")
    print(f"  Profiler Summary (steps {cfg.start_step}-{cfg.end_step})")
    print(f"  Trace:   {trace_path}")
    print(f"  Summary: {summary_path}")
    print(f"  View trace: chrome://tracing  or  https://ui.perfetto.dev")
    print(f"{'=' * 80}\n")

    return trace_path
