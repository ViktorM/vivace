#!/usr/bin/env bash
# 2x4090 3-seed gsm8k benchmark: 5 algos x 3 seeds = 15 runs at 1.5B.
#
# Seeds are the OUTER loop (7 -> 13 -> 42) so after ~1/3 of the night we have a
# full 5-algo ranking at 1 seed, firming up to mean+-std by morning. If anything
# breaks mid-run, the completed seeds still give a usable comparison.
#
# One wandb GROUP per algo (seed is a separate run in the group), so
# tools/aggregate_seeds.py <group> computes mean+-std per algo.
#
# Configs carry their own per-algo (lr, ep, kl) — no overrides here.
# Wall: ~20-25 min/run x 15 ~= 5-6h (post perf-batch + max_num_seqs fix).

set -u

REPO=/home/viktor/Projects/Research/vivace
DATE_TAG=$(date +%Y%m%d_%H%M)
STATUS_LOG=/tmp/vivace_gsm8k_3seed_status.log
CFG_DIR=${REPO}/vivace/configs/experiments/v1_1.5b

ALGOS=(grpo dr_grpo gspo dapo cispo)
# Spread seeds (not consecutive) — with the cfg.seed + 1000*rank derivation these
# are collision-proof, but spread values are conventional for independent reps.
SEEDS=(7 13 42)

log_status() { echo "[$(date +%H:%M:%S)] $1 exit=$2" >> "${STATUS_LOG}"; }

echo "=== gsm8k 3-seed sweep started $(date) ===" > "${STATUS_LOG}"
echo "  5 algos x 3 seeds = 15 runs at Qwen2.5-1.5B / gsm8k" >> "${STATUS_LOG}"

for seed in "${SEEDS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    cfg=${CFG_DIR}/${algo}_1.5b_gsm8k.yaml
    group="overnight-1.5b-gsm8k-${DATE_TAG}_${algo}"
    run_dir="runs/v1_1.5b/${algo}_1.5b_gsm8k_seed${seed}_${DATE_TAG}"
    name="${algo}_seed${seed}"

    echo "=== $(date +%H:%M:%S) :: ${name} ==="
    CUDA_VISIBLE_DEVICES=0,1 ${REPO}/.venv/bin/torchrun --nproc_per_node=2 --master-port=29500 \
      -m vivace.scripts.train \
      --config ${cfg} \
      --set "seed=${seed}" \
      --run-dir ${run_dir} \
      --wandb-group ${group}
    log_status "${name}" "$?"
  done
done

echo "=== gsm8k 3-seed sweep complete $(date) ===" | tee -a "${STATUS_LOG}"
grep -c "exit=0"  "${STATUS_LOG}" | xargs -I{} echo "  successful: {}"
grep -cE "exit=[^0]" "${STATUS_LOG}" | xargs -I{} echo "  failed: {}"
echo
echo "Per-algo wandb groups: overnight-1.5b-gsm8k-${DATE_TAG}_{grpo,dr_grpo,gspo,dapo,cispo}"
echo "Aggregate each: .venv/bin/python tools/aggregate_seeds.py overnight-1.5b-gsm8k-${DATE_TAG}_dapo --by rl.lr,rl.optim_epochs"
