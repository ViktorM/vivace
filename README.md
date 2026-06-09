# vivace

A fast, hackable RL post-training lab for language models.
Minimal by design, extensible by layout. Built for research and fast iteration.

## What this is

A research codebase for running RL post-training experiments on
language models (and potentially vision-language models). The goals, in order:

1. **Readable.** The composable PG zoo lives in a single file you can understand in one sitting.
2. **Hackable.** Swap loss functions, advantage estimators, reward functions, benchmarks, and rollout backends freely via config switches.
3. **Multi-GPU from day one.** Two execution modes: *disaggregated* (vLLM rollout workers
   on dedicated GPUs, trainer on others) and *colocated* (rollout and training share GPUs,
   useful for debugging).
4. **Multiple benchmarks.** GSM8K, MATH (Hendrycks train), MATH-500, AIME 2024, AIME 2025 —
   behind one `Env` interface. Train on one env, eval on many in the same step.
   DeepMath-103K, Omni-MATH, and agentic/tool-use envs on the roadmap.
5. **Not a framework.** No plugin system, no registry, no abstract factories. Just code.

## What this is not

- A heavyweight framework. No plugin DSL, no abstract factory soup, no command bus.
  You can `pip install -e .`, subclass `Env`, and `register_env("my_env", MyEnv)`
  to plug a custom benchmark into a yaml-driven run — but the codebase itself stays
  small enough to read end-to-end.
- A replacement for verl, OpenRLHF, TRL, or slime. Those are excellent. This is for
  research and for experiments where those libraries get in the way.

## Install

```bash
git clone https://github.com/ViktorM/vivace.git
cd vivace
uv venv
uv pip install -e ".[dev]"

# Qwen 3.5 fast DeltaNet kernels (optional but ~3-5x faster)
uv pip install -e ".[qwen35]"
```

vLLM ≥0.22 (the version floor in pyproject) pins a matching `torch==2.11.x`.
If the default resolve breaks, force the matching torch CUDA build, e.g.:
`uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu130`.

### Upgrading an existing venv

```bash
uv pip install -e ".[dev]" --upgrade
```

Or upgrade load-bearing packages explicitly:
```bash
uv pip install --upgrade "vllm>=0.22" "transformers>=4.58" "torch>=2.11" "peft>=0.19"
```

## Quickstart

Two end-to-end recipes to verify your setup. Both train on GSM8K with LoRA
r=16 and the DAPO loss + RLOO advantages: colocated runs Qwen2.5-0.5B-Instruct
on one GPU, disaggregated runs Qwen2.5-1.5B across two.

### Colocated — single GPU (trainer + vLLM share one device)

```bash
python -m vivace.scripts.train \
    --config vivace/configs/gsm8k/dapo_0.5b_colo.yaml \
    --num-steps 200 \
    --run-dir "runs/colo_$(date +%Y%m%d_%H%M%S)"
```
Baseline 0% → ~43% accuracy in 200 steps (measured on the 2×4090 DDP variant
of this config; the single-GPU run sees half the batch per step).
Uses `weight_sync_method: ipc` (CUDA IPC zero-copy on the same GPU); vLLM
sleeps during each train phase, so rollouts get most of the GPU.

### Disaggregated — two GPUs (trainer on GPU 0, vLLM on GPU 1)

```bash
python -m vivace.scripts.train \
    --config vivace/configs/gsm8k/dapo_2x4090.yaml \
    --num-steps 200 \
    --run-dir "runs/disagg_$(date +%Y%m%d_%H%M%S)"
```
Qwen2.5-1.5B + LoRA. Uses `weight_sync_method: nccl` (direct GPU→GPU broadcast).

### Smoke test

```bash
python -m vivace.scripts.train \
    --config vivace/configs/gsm8k/dapo_0.5b_colo.yaml \
    --num-steps 5 \
    --run-dir "runs/smoke_$(date +%Y%m%d_%H%M%S)"
```

`WANDB_MODE=disabled` if you want to skip wandb. See
[docs/running_experiments.md](docs/running_experiments.md) for the full launch
pattern, sync-method overrides, and comparison commands.

## Recommended recipe — CISPO (MiniMax-M1)

vivace's validated production recipe is **CISPO + canonical IS clip + hybrid
normalization**, modeled after MiniMax-M1
([arxiv:2506.13585](https://arxiv.org/abs/2506.13585)).

| key | value | rationale |
|---|---|---|
| `loss_type` | `cispo` | clipped IS weight × `policy_logp`, all token gradients preserved (M1 eq. 5) |
| `cispo_use_token_mask` | `false` | the optional M1 eq. 7 PPO-style mask. Off by default; **either** the mask **or** hybrid normalization stabilizes ep≥2; both together adds no extra benefit |
| `cispo_normalization` | `hybrid` | average of token-mean and sequence-mean reductions — defuses long-negative-sample length imbalance (M1 pattern-collapse section) |
| `kl_coef` | `0.005` | math500 sweet spot; `kl=0` also viable per the M1 paper |
| `adam_beta2` / `adam_eps` | `0.95` / `1e-15` | the M1 Adam tuning for fast-moving RL gradient statistics |
| `optim_epochs` | `4` | per-rollout-batch reuse; `ep=2` works too, `ep≥8` collapses at fixed LR |

Scaling result with this recipe on Qwen2.5 / math env (200 steps, 4×H200 colo,
single seed):

| model | gsm8k | math500 |
|---|---|---|
| 1.5B | 75.5 | 41.2 |
| 3B   | 83.4 | 45.2 |
| 7B   | 91.4 | 50.8 (500-step) |

See [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) for the
algorithmic story (the canonical-vs-mask 2×2 ablation, the fp32-reduction fix,
the kl-coefficient sweep) and [`docs/ablation_studies.md`](docs/ablation_studies.md)
for the full result table.

## Layout

```
vivace/                    # installable package
├── algos/                 # Training methods (the math)
│   ├── types.py           # RolloutBatch, RLConfig, SFTConfig, preset factories
│   ├── policy_gradient.py # Composable PG zoo (GRPO, DAPO, GSPO, RLOO, CISPO, ...)
│   └── sft.py             # Optional SFT warmup
├── envs/                  # Benchmark wrappers behind a common Env interface
│   ├── base.py            # Env ABC + Example
│   ├── gsm8k.py           # GSM8K
│   ├── math.py            # MATH (Hendrycks) — train env
│   ├── math500.py         # MATH-500 — held-out 500-problem eval
│   ├── aime.py            # AIME 2024 + AIME 2025
│   ├── math_prompt.py     # shared math prompt + answer-extraction utilities
│   └── __init__.py        # ENV_REGISTRY + make_env
├── rewards.py             # Rule-based reward functions
├── rollout/               # Rollout backends
│   ├── hf_sampler.py      # HF generate (single-GPU dev)
│   └── vllm_worker.py     # vLLM + weight sync
├── train/
│   └── trainer.py         # The orchestrator
├── eval/
│   └── runner.py          # pass@1, maj@k, pass@k
├── utils/
│   ├── stats.py           # TrainingStats + plots
│   ├── logging.py         # wandb wrapper, console logger
│   ├── perf.py            # Timer, throughput measurement
│   ├── checkpointing.py   # save/load (LoRA-aware)
│   └── distributed.py     # init, ranks, barriers, weight-sync subgroup
├── configs/               # YAML configs (one per experiment)
└── scripts/               # CLI entry points (vivace-train; sft/eval are stubs)
tests/                     # Unit tests, weight-sync verification, vLLM↔HF probes, run comparison plotting
docs/                      # running_experiments, weight_sync_approaches, training_theory, profiling
```

## Execution modes

### Colocated

Trainer and vLLM share GPUs, time-multiplexed via `vllm.sleep()` / `wake_up()`.
Fewer GPUs needed; uses CUDA IPC for zero-copy weight sync on the same device.

```
GPU 0: [ trainer ] <-> [ vllm worker ]   (single-GPU)
```

Weight sync: set `ipc` in colocated configs (all shipped ones do). The code
default `nccl` is rejected at init for shared-GPU layouts — NCCL can't connect
two ranks on one device; `disk` works as a slower fallback.

### Disaggregated

Rollout and training on separate GPUs. Trainer pushes updated weights to vLLM
over NCCL after each optimizer step.

Two-GPU setup:
```
GPU 0: [ trainer        ] <--weight sync (NCCL)-- 
GPU 1: [ vllm worker    ] --generations-->
```

Weight sync: `nccl` (default). `disk` works as a fallback in either topology.

### Picking a config

All paths relative to `vivace/configs/`.

| where you are | use |
|---|---|
| 1 GPU, debugging | `gsm8k/dapo_0.5b_colo.yaml` (colocated, IPC, 0.5B) |
| 2 GPUs, faster | `gsm8k/dapo_2x4090.yaml` (disaggregated, NCCL, 1.5B) |
| 2 GPUs, full FT | `gsm8k/grpo_0.5b_full.yaml` (disaggregated, NCCL, 0.5B no-LoRA) |
| MATH instead of GSM8K | `math/cispo_2x4090.yaml` (disaggregated, NCCL, 1.5B) |

## Adding a new loss variant

1. Open `vivace/algos/policy_gradient.py`.
2. Add a new branch to the `if/elif` ladder in `compute_loss`. Most
   variants are 5-15 lines.
3. Add the new value to the `loss_type` enum comment in `RLConfig`
   (`vivace/algos/types.py`).
4. Optionally add a YAML preset under `vivace/configs/`.

That's it. No registry, no plugin file, no new files. Variants
compose with any `adv_type` automatically — `loss_type=gspo +
adv_type=rloo` gives you GSPO with leave-one-out advantages without
writing anything new.

## Adding a new benchmark

In-tree (most common — you're hacking on vivace):

1. Create `vivace/envs/my_bench.py` subclassing `Env` from `vivace/envs/base.py`.
2. Implement `load_split()`, `format_prompt()`, and link to a reward function.
3. Add a row to `ENV_REGISTRY` in `vivace/envs/__init__.py`.
4. Point a config at it.

Out-of-tree (you've `pip install -e .`'d vivace and want to add an env from
your own package — e.g. wrapping OpenEnv or a private benchmark):

```python
from vivace.envs import register_env, Env

class MyEnv(Env):
    name = "my_env"
    def load_split(self, split): ...
    def format_prompt(self, example): ...
    # ...

register_env("my_env", MyEnv)
# then launch training with env_name: my_env in your yaml
```

The reward function and the verifier are the contract. Everything else is up to you.

## Testing

```bash
# Low-level NCCL broadcast smoke
python -m tests.test_nccl_sync

# 3-step weight-sync end-to-end (fresh / perturb / sync) against a backend
python -m tests.test_weight_sync --config <path> --method disk
python -m tests.test_weight_sync --config <path> --method nccl

# vLLM↔HF logprob equivalence probe (e.g. when debugging RL ratios)
python -m tests.probe_vllm_hf_logprob --model Qwen/Qwen2.5-0.5B-Instruct

# Compare N runs (per-step time, throughput, cumulative wall, training reward).
# First run is treated as the baseline for the wall-clock delta numbers.
python -m tests.compare_sync_perf  --runs runs/<a> runs/<b> runs/<c> \
                                   --labels A B C --out runs/cmp.png
```

See [`docs/weight_sync_approaches.md`](docs/weight_sync_approaches.md) for
the design rationale and failure-mode diagnostics.

## Roadmap

- [x] Composable PG zoo: GRPO / DAPO / GSPO / RLOO / Dr.GRPO / CISPO (with MiniMax-M1 mask)
- [x] GSM8K env + reward functions
- [x] Single-GPU training with HF sampler
- [x] vLLM rollout backend with LoRA hot-swap (disk + NCCL + CUDA IPC paths)
- [x] Disaggregated mode (separate rollout + trainer GPUs)
- [x] DDP across multiple trainer GPUs (colocated and disaggregated)
- [x] MATH (Hendrycks train) + MATH-500 + AIME 2024/2025 envs
- [x] Multi-eval-env trainer (train on one env, eval on many per step)
- [ ] FSDP for large models
- [ ] DeepMath-103K + Omni-MATH envs
- [ ] Optional SFT warmup path
- [ ] Agentic tasks: tool-use rollout hooks (calculator, Python, search)
- [ ] RL with self-distillation
- [ ] VLM support (vision-language model training)
- [ ] Speedrun configs + leaderboard
