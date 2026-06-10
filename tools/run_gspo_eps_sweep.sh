#!/usr/bin/env bash
# GSPO epsilon sweep — docs/gspo_diagnostic.md §4 + 2026-06-09 addendum.
#
# Sweeps the sequence-level trust region on the tracked known-good config
# (gsm8k/gspo_0.5b_colo.yaml: 0.5B-Instruct, lr 6e-5, ep=2, kl 0.01),
# single-GPU colocated per run, two runs in parallel (GPU 0 + GPU 1).
#
# Range anchored by measurement: one optimizer update moves the seq-ratio
# p50 ~5e-4 and p99 ~7e-3; the noise floor is < 3e-4. 0.2 is the control
# cell (today's preset, clip provably inert). Watch gspo/seq_ratio_p99 and
# clip_frac per cell; pick the eps where clip_frac lands in ~0.01-0.10
# without tanking reward.
#
#   STEPS=200 tools/run_gspo_eps_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M)
GROUP="gspo-eps-sweep-0.5b-gsm8k-${TS}"
EPS=(0.002 0.005 0.01 0.02 0.05 0.2)
STEPS=${STEPS:-200}

run_one() {  # $1=eps  $2=gpu
    local eps=$1 gpu=$2
    local tag="gspo_eps${eps}_gpu${gpu}_${TS}"
    .venv/bin/python -m vivace.scripts.train \
        --config vivace/configs/gsm8k/gspo_0.5b_colo.yaml \
        --num-steps "${STEPS}" \
        --set "trainer_gpus=[${gpu}]" --set "rollout_gpus=[${gpu}]" \
        --set "rl.clip_low=${eps}" --set "rl.clip_high=${eps}" \
        --wandb-group "${GROUP}" --wandb-run-name "gspo_eps${eps}" \
        --run-dir "runs/${tag}" > "runs/${tag}.log" 2>&1
    echo "[sweep] eps=${eps} (gpu ${gpu}) done: $(tail -1 runs/${tag}.log)"
}

echo "[sweep] group=${GROUP} steps=${STEPS} eps={${EPS[*]}}"
for ((i = 0; i < ${#EPS[@]}; i += 2)); do
    run_one "${EPS[i]}" 0 &
    p0=$!
    p1=
    if (( i + 1 < ${#EPS[@]} )); then
        run_one "${EPS[i+1]}" 1 &
        p1=$!
    fi
    wait "$p0" ${p1:+"$p1"}
done
echo "[sweep] all cells done: wandb group ${GROUP}"
