#!/usr/bin/env bash
# 3B local dry run (v1_benchmark_plan.md sequencing steps 3 + 5) and the GSPO
# eps 3-seed check (gspo_diagnostic.md §3/§4) on 2x4090:
#   A. DAPO 3B/gsm8k seeds 7,13,42 — DDP colo on both GPUs, one 3-seed cell
#   B. in parallel once A is done:
#        GPU 0: DAPO 3B/math seed 7, single-GPU colo with grad_accum 8 (same 64
#               traj/optimizer step as the 2-rank accum-4 recipe). Single GPU
#               because the 1280-token backward at 3B (~19-21GB peak: 5.8GB
#               weights + full-vocab logits) does not fit next to the ~4GB
#               desktop session on GPU 1; GPU 0 has no desktop, so the vLLM
#               pool goes back up to 0.5 there.
#        GPU 1: GSPO 0.5B eps {0.01, 0.2} x seeds {7,13,42} — the preset decision
#               left open after two single-seed sweeps (both flat within noise).
#
# One wandb group per cell; aggregate with tools/aggregate_seeds.py <group>.
# Status log: /tmp/vivace_3b_dryrun_status.log (one line per run exit).

set -u
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M)
STATUS_LOG=/tmp/vivace_3b_dryrun_status.log
CFG=vivace/configs/experiments/v1_3b

log_status() { echo "[$(date +%H:%M:%S)] $1 exit=$2" >> "${STATUS_LOG}"; }

run_gsm8k_ddp() {  # $1=seed
    local seed=$1
    local run_dir="runs/v1_3b/dapo_3b_gsm8k_seed${seed}_${TS}"   # separate `local`s: one statement expands ${seed} before assigning it
    echo "=== $(date +%H:%M:%S) :: dapo_3b_gsm8k seed=${seed} (DDP 2 GPUs) ==="
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 --master-port=29500 \
        -m vivace.scripts.train \
        --config ${CFG}/dapo_3b_gsm8k.yaml \
        --set "seed=${seed}" \
        --run-dir "${run_dir}" \
        --wandb-group "dryrun-3b-gsm8k-${TS}_dapo" > "${run_dir}.log" 2>&1
    log_status "dapo_3b_gsm8k_seed${seed}" "$?"
}

run_math_single() {  # $1=seed  $2=gpu
    local seed=$1 gpu=$2
    local run_dir="runs/v1_3b/dapo_3b_math_seed${seed}_1gpu_${TS}"
    echo "=== $(date +%H:%M:%S) :: dapo_3b_math seed=${seed} (single GPU ${gpu}, accum 8) ==="
    .venv/bin/python -m vivace.scripts.train \
        --config ${CFG}/dapo_3b_math.yaml \
        --set "trainer_gpus=[${gpu}]" --set "rollout_gpus=[${gpu}]" \
        --set "gpu_memory_utilization=0.5" --set "vllm_max_num_seqs=32" \
        --set "rl.grad_accum_steps=8" --set "seed=${seed}" \
        --run-dir "${run_dir}" \
        --wandb-group "dryrun-3b-math-${TS}_dapo_1gpu" > "${run_dir}.log" 2>&1
    log_status "dapo_3b_math_seed${seed}_1gpu" "$?"
}

run_eps() {  # $1=eps  $2=seed  $3=gpu
    local eps=$1 seed=$2 gpu=$3
    local run_dir="runs/gspo_eps3seed/gspo_eps${eps}_seed${seed}_${TS}"
    echo "=== $(date +%H:%M:%S) :: gspo_eps${eps} seed=${seed} (GPU ${gpu}) ==="
    .venv/bin/python -m vivace.scripts.train \
        --config vivace/configs/gsm8k/gspo_0.5b_colo.yaml \
        --num-steps 200 \
        --set "trainer_gpus=[${gpu}]" --set "rollout_gpus=[${gpu}]" \
        --set "rl.clip_low=${eps}" --set "rl.clip_high=${eps}" --set "seed=${seed}" \
        --wandb-group "gspo-eps3seed-0.5b-gsm8k-${TS}_eps${eps}" \
        --wandb-run-name "gspo_eps${eps}_seed${seed}" \
        --run-dir "${run_dir}" > "${run_dir}.log" 2>&1
    log_status "gspo_eps${eps}_seed${seed}" "$?"
}

mkdir -p runs/v1_3b runs/gspo_eps3seed
echo "=== 3B dry run + GSPO eps 3-seed started $(date) (TS=${TS}) ===" > "${STATUS_LOG}"

# A: the 3-seed gsm8k cell, DDP (PHASE=b skips it: relaunch of the second half only)
if [ "${PHASE:-all}" != "b" ]; then
    for seed in 7 13 42; do run_gsm8k_ddp "${seed}"; done
fi

# B: math smoke on GPU 0 alongside the eps cells on GPU 1
run_math_single 7 0 &
p_math=$!
( for seed in 7 13 42; do run_eps 0.01 "${seed}" 1; run_eps 0.2 "${seed}" 1; done ) &
p_eps=$!
wait "$p_math" "$p_eps"

echo "=== chain complete $(date) ===" | tee -a "${STATUS_LOG}"
echo "  ok: $(grep -c 'exit=0' "${STATUS_LOG}")  failed: $(grep -cE 'exit=[^0]' "${STATUS_LOG}")"
echo "  groups: dryrun-3b-gsm8k-${TS}_dapo  dryrun-3b-math-${TS}_dapo_1gpu  gspo-eps3seed-0.5b-gsm8k-${TS}_eps{0.01,0.2}"
