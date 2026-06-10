#!/usr/bin/env bash
# Math 3-seed benchmark: top-3 algos (DAPO, CISPO, Dr.GRPO) x 3 seeds = 9 runs
# at Qwen2.5-1.5B, trained on Hendrycks MATH, eval on gsm8k + math500 + AIME 24/25/26.
#
# Seeds OUTER loop (7 -> 13 -> 42): a full 3-algo ranking lands after the first
# third; mean+-std firms as the later seeds complete. ~2-2.5h/run pre-fix; expect
# roughly half that after the Jun-10 perf batch + max_num_seqs/mem-util fixes.
#
# One wandb GROUP per algo; aggregate with:
#   .venv/bin/python tools/aggregate_seeds.py math-1.5b-<date>_<algo>

set -u

REPO=/home/viktor/Projects/Research/vivace
DATE_TAG=$(date +%Y%m%d_%H%M)
STATUS_LOG=/tmp/vivace_math_3seed_status.log
CFG_DIR=${REPO}/vivace/configs/experiments/overnight

ALGOS=(dapo cispo dr_grpo)
SEEDS=(7 13 42)

log_status() { echo "[$(date +%H:%M:%S)] $1 exit=$2" >> "${STATUS_LOG}"; }

echo "=== math 3-seed sweep started $(date) ===" > "${STATUS_LOG}"
echo "  3 algos x 3 seeds = 9 runs at Qwen2.5-1.5B / Hendrycks MATH" >> "${STATUS_LOG}"
echo "  eval: gsm8k, math500, aime24, aime25, aime26 (AIME @ 4096 tokens)" >> "${STATUS_LOG}"

for seed in "${SEEDS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    cfg=${CFG_DIR}/${algo}_1.5b_math.yaml
    group="math-1.5b-${DATE_TAG}_${algo}"
    run_dir="runs/math3seed/${algo}_1.5b_math_seed${seed}_${DATE_TAG}"
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

echo "=== math 3-seed sweep complete $(date) ===" | tee -a "${STATUS_LOG}"
grep -c "exit=0"  "${STATUS_LOG}" | xargs -I{} echo "  successful: {}"
grep -cE "exit=[^0]" "${STATUS_LOG}" | xargs -I{} echo "  failed: {}"
echo
echo "Wandb groups: math-1.5b-${DATE_TAG}_{dapo,cispo,dr_grpo}"
