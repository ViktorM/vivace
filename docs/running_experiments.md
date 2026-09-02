# Running experiments

Don't assign and use a shell variable on one command line
(`RUN_DIR=... python ... --run-dir "$RUN_DIR"`): bash expands `$RUN_DIR`
before the assignment, so output lands in a stale folder. Assign on its own
line, or pass a literal path.

## Launching a run

```bash
python -m vivace.scripts.train \
    --config vivace/configs/gsm8k/dapo_0.5b_colo.yaml \
    --num-steps 200 \
    --run-dir "runs/qw25_0.5b_ipc_$(date +%Y%m%d_%H%M%S)"
```

To save the trainer log:

```bash
RUN_DIR="runs/qw25_0.5b_ipc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
python -m vivace.scripts.train \
    --config vivace/configs/gsm8k/dapo_0.5b_colo.yaml \
    --num-steps 200 \
    --run-dir "$RUN_DIR" > "$RUN_DIR/train.log" 2>&1
```

Set `WANDB_MODE=disabled` if you don't want metrics pushed to wandb.

## wandb logging

One-time setup (per machine):

```bash
wandb login                    # paste API key, stored in ~/.netrc
export WANDB_ENTITY=<your-entity>   # add to ~/.bashrc to make permanent
```

`wandb_project` defaults to `vivace` ([trainer.py](../vivace/train/trainer.py)
TrainerConfig). The trainer reads three knobs from TrainerConfig (or YAML) and
exposes them as CLI flags too:

| field | YAML key | CLI flag | default |
|---|---|---|---|
| project | `wandb_project` | `--wandb-project` | `"vivace"` |
| per-run name | `wandb_run_name` | `--wandb-run-name` | `basename(run_dir)` |
| group (clusters runs in UI) | `wandb_group` | `--wandb-group` | `None` (no group) |

Precedence: **CLI > YAML > auto-derive (run_dir basename) > wandb auto-slug.**
Setting `wandb_run_name` or `--wandb-run-name` is rarely needed — the
`run_dir` basename is usually meaningful enough (e.g. `ddp_500_20260503_120000`).

### Seed sweeps and grouped reporting

`wandb_group` is the aggregation bucket: wandb's Group-by → `Group` reduces
the runs in a group to mean ± stddev. One group per variant, seeds inside it —
two algorithms in one group get averaged together. The `tools/` sweep scripts
use `<prefix>-<timestamp>_<algo>`.

Pin the group in the YAML for an experiment family, or pass it per launch and
vary `--seed` and `--run-dir`:

```bash
TS=$(date +%Y%m%d_%H%M%S)
for ALGO in cispo gspo; do            # 2-GPU DDP-colo configs ([0,1] GPU lists)
  for SEED in 42 43 44; do
    torchrun --nproc_per_node=2 -m vivace.scripts.train \
        --config "vivace/configs/gsm8k/${ALGO}_0.5b_colo.yaml" \
        --num-steps 200 --seed $SEED \
        --run-dir "runs/${ALGO}_seed${SEED}_${TS}" \
        --wandb-group "gsm8k-${TS}_${ALGO}"
  done
done
```

(`gsm8k/dapo_0.5b_colo.yaml` is single-GPU-shaped for the README quickstart —
launch it with plain `python`, or set its GPU lists to `[0, 1]` for torchrun.)

Two groups of 3 runs, each one mean line with a variance band.
`tests/wandb_regroup.py` regroups runs launched without a group.

For one-off overrides without editing YAML, the equivalent env vars also
work: `WANDB_NAME`, `WANDB_TAGS=tag1,tag2`, `WANDB_NOTES="..."`. CLI flags
are easier when you want them in your shell history.

## Sync method overrides

```bash
--weight-sync-method ipc                          # same-GPU CUDA-IPC aliasing; colocated only (all colo yamls)
--weight-sync-method nccl                         # disaggregated only (TrainerConfig default)
--weight-sync-method disk                         # LoRA only, either mode; /dev/shm by default
--weight-sync-method disk --weight-sync-disk-path "$RUN_DIR/adapter"   # disk on NVMe
```

The trainer prints which path actually ran at startup:
- `[trainer init] IPC weight sync ready: N params`
- `[trainer init] NCCL weight sync ready: N params`
- `[disk-sync] adapter path: /dev/shm/...` or `<run_dir>/adapter`

## Comparing runs

```bash
python -m tests.compare_sync_perf \
    --runs   runs/<a> runs/<b> runs/<c> \
    --labels "A"      "B"      "C"      \
    --out    runs/compare.png
```

Accepts N ≥ 2 runs (first is the baseline for wall-clock deltas) and either
run-dirs (newest `stats_*.pt` used) or explicit `stats_*.pt` paths.
