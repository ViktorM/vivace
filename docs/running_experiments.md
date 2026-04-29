# Running experiments

Pass `--run-dir` as a literal path. Don't rely on shell variables — bash expands
them before inline assignments take effect, which silently routes output to
stale folders.

## Launching a run

```bash
python -m vivace.scripts.train \
    --config vivace/configs/dapo_gsm8k_qw25_0.5b_lora_colo.yaml \
    --num-steps 200 \
    --run-dir "runs/qw25_0.5b_ipc_$(date +%Y%m%d_%H%M%S)"
```

To save the trainer log:

```bash
RUN_DIR="runs/qw25_0.5b_ipc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
python -m vivace.scripts.train \
    --config vivace/configs/dapo_gsm8k_qw25_0.5b_lora_colo.yaml \
    --num-steps 200 \
    --run-dir "$RUN_DIR" > "$RUN_DIR/train.log" 2>&1
```

Set `WANDB_MODE=disabled` if you don't want metrics pushed to wandb.

## Sync method overrides

```bash
--weight-sync-method ipc                          # same-GPU zero-copy (colocated default)
--weight-sync-method nccl                         # disaggregated only
--weight-sync-method disk                         # /dev/shm by default
--weight-sync-method disk --weight-sync-disk-path "$RUN_DIR/adapter"   # disk on NVMe
```

The trainer prints which path actually ran at startup:
- `[trainer init] IPC weight sync ready: N params`
- `[trainer init] NCCL weight sync ready: N params`
- `[disk-sync] adapter path: /dev/shm/...` or `<run_dir>/adapter`

## Comparing runs

```bash
# Pair (a vs b)
python -m tests.compare_sync_perf \
    --a runs/<a> --a-label "A" \
    --b runs/<b> --b-label "B" \
    --out runs/compare.png

# N-way
python -m tests.compare_sync_perf_n \
    --runs   runs/<a> runs/<b> runs/<c> \
    --labels "A"      "B"      "C"      \
    --out    runs/compare.png
```

Both accept either a run-dir (newest `stats_*.pt` is used) or an explicit
`stats_*.pt` path.
