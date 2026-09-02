#!/usr/bin/env bash
# 3B cells for the release tables (v1 protocol, configs in v1_3b/, colo DDP on 2x4090):
#   1. CISPO 3B/math x seeds 7,13,42        — replaces the README's single-seed 3B row
#   2. gsm8k: grpo, dr_grpo, gspo, cispo x 3 seeds (seed OUTER loop) — completes the
#      5-algo 3B/gsm8k cell next to the finished DAPO cell (dryrun-3b-gsm8k-20260901_1410_dapo)
# Single GPU 0 tonight (GPU 1 is in use): ~3.5 h/math run, ~65 min/gsm8k run -> ~23 h total.
# One wandb group per cell; groups carry a _1gpu suffix.
# Status log: /tmp/vivace_3b_cells_status.log. PYTHONUNBUFFERED so the logs flush per line.

set -u
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M)
STATUS_LOG=/tmp/vivace_3b_cells_status.log
CFG=vivace/configs/experiments/v1_3b
export PYTHONUNBUFFERED=1

log_status() { echo "[$(date +%H:%M:%S)] $1 exit=$2" >> "${STATUS_LOG}"; }

run_cell() {  # $1=algo  $2=env  $3=seed
    local algo=$1 env=$2 seed=$3
    local run_dir="runs/v1_3b/${algo}_3b_${env}_seed${seed}_1gpu_${TS}"
    # Single GPU 0 (no DDP): GPU 1 carries the desktop plus ~5-8GB of other work, which
    # left rank 1 no room for vLLM at any pool. grad_accum 8 keeps 64 traj/optimizer step
    # (= 2 ranks x accum 4); pool 0.5 + 32/64 seqs is the path that ran the math smoke.
    local seqs=64; [ "${env}" = math ] && seqs=32
    echo "=== $(date +%H:%M:%S) :: ${algo}_3b_${env} seed=${seed} (single GPU 0, accum 8) ==="
    .venv/bin/python -m vivace.scripts.train \
        --config ${CFG}/${algo}_3b_${env}.yaml \
        --set "trainer_gpus=[0]" --set "rollout_gpus=[0]" \
        --set "gpu_memory_utilization=${MEM_UTIL:-0.5}" --set "vllm_max_num_seqs=${seqs}" \
        --set "rl.grad_accum_steps=8" --set "seed=${seed}" \
        --run-dir "${run_dir}" \
        --wandb-group "v1-3b-${env}-${TS}_${algo}_1gpu" > "${run_dir}.log" 2>&1
    log_status "${algo}_3b_${env}_seed${seed}_1gpu" "$?"
}

mkdir -p runs/v1_3b
echo "=== 3B cells started $(date) (TS=${TS}) ===" > "${STATUS_LOG}"

for seed in 7 13 42; do run_cell cispo math "${seed}"; done
for seed in 7 13 42; do
    for algo in grpo dr_grpo gspo cispo; do run_cell "${algo}" gsm8k "${seed}"; done
done

echo "=== 3B cells complete $(date) ===" | tee -a "${STATUS_LOG}"
echo "  ok: $(grep -c 'exit=0' "${STATUS_LOG}")  failed: $(grep -cE 'exit=[^0]' "${STATUS_LOG}")"
echo "  groups: v1-3b-math-${TS}_cispo_1gpu  v1-3b-gsm8k-${TS}_{grpo,dr_grpo,gspo,cispo}_1gpu"
