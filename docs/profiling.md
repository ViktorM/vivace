# Profiling

vivace has built-in PyTorch Profiler support. Add a `profiling:` block to your YAML config to capture GPU kernel timelines, memory allocations, and operator breakdowns.

## Quick start

Add to any existing config:

```yaml
profiling:
  enabled: true
  start_step: 5
  end_step: 8
```

Run training as usual:

```bash
vivace-train --config vivace/configs/your_config.yaml --num-steps 10
```

After step 8, the profiler will:
1. Print a summary table to console (top CUDA kernels, CPU operators, memory allocators)
2. Save a Chrome trace to `{run_dir}/profiling/trace_step5-8.json`

## Viewing the trace

Open the trace in either:
- **Chrome**: navigate to `chrome://tracing`, click "Load", select the `.json` file
- **Perfetto**: go to https://ui.perfetto.dev, drag and drop the file

The trace shows a timeline of all CUDA kernels, CPU operators, and memory events across the profiled steps. You can zoom into individual steps to see rollout vs. train phase breakdown at the kernel level.

## Config options

```yaml
profiling:
  enabled: false          # master switch (default: false)
  start_step: 5           # first step to profile (skip JIT/allocator warmup)
  end_step: 8             # last step to profile (exclusive)
  record_shapes: true     # log tensor shapes per operator
  profile_memory: true    # track CUDA memory allocations
  with_stack: false       # capture Python call stacks (expensive, ~2x slower)
  with_flops: true        # estimate FLOPs per operator
  output_dir: null        # defaults to {run_dir}/profiling/
```

### Choosing the profiling window

- **Skip warmup**: the first few steps have JIT compilation, CUDA allocator warmup, and vLLM initialization overhead. Start at step 5+ for representative numbers.
- **Keep it short**: 3-5 steps is enough. Profiling adds overhead (~10-20%) and traces grow large.
- **For before/after comparisons**: use the same window (e.g., steps 5-8) across both runs.

### with_stack

When `with_stack: true`, the trace includes Python call stacks for each operator. This lets you see *which line of code* launched each kernel. Very useful for understanding unfamiliar code paths, but roughly doubles profiling overhead. Use it for targeted debugging, not routine profiling.

## Reading the summary table

The console output includes three tables:

1. **Top CUDA kernels by GPU time** — shows which GPU operations dominate. Look for:
   - `aten::mm` / `aten::bmm` — matrix multiplications (forward + backward)
   - `flash_*` / `mem_efficient_*` — attention kernels
   - `nccl*` — collective communication (weight sync, DDP)
   - `aten::copy_` — CPU/GPU transfers (should be small)

2. **Top CPU operators** — shows CPU-side overhead. Large values here suggest CPU bottlenecks.

3. **Top memory allocators** — shows which operators allocate the most GPU memory.

## Example: profiling before/after NCCL weight sync

```yaml
# Before: disk-based LoRA sync
profiling:
  enabled: true
  start_step: 5
  end_step: 8
  output_dir: runs/profile_disk_sync

# After: NCCL weight sync (same config, different output_dir)
profiling:
  enabled: true
  start_step: 5
  end_step: 8
  output_dir: runs/profile_nccl_sync
```

Compare the sync_weights time in the summary tables and visually in the traces.
