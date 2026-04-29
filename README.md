# vivace

A fast, hackable RL post-training lab for language models.
Minimal by design, extensible by layout. Built for research, not production.

## What this is

A research codebase for running RL post-training experiments on
language models (and potentially vision-language models). The goals, in order:

1. **Readable.** The composable PG zoo lives in a single file you can understand in one sitting.
2. **Hackable.** Swap loss functions, advantage estimators, reward functions, benchmarks, and rollout backends freely via config switches.
3. **Multi-GPU from day one.** Two execution modes: *disaggregated* (vLLM rollout workers
   on dedicated GPUs, trainer on others) and *colocated* (rollout and training share GPUs,
   useful for debugging).
4. **Multiple benchmarks.** GSM8K, MATH, AIME, DeepMath-103K, Omni-MATH,
   and agentic/tool-use environments — behind one `Env` interface.
5. **Not a framework.** No plugin system, no registry, no abstract factories. Just code.

## What this is not

- A library. Don't `pip install vivace` into other projects. Fork it, hack it, run it.
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

vLLM 0.20 hard-pins `torch==2.11.0`. If the default resolve breaks, force the
matching torch CUDA build, e.g.:
`uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu130`.

### Upgrading an existing venv

```bash
uv pip install -e ".[dev]" --upgrade
```

Or upgrade load-bearing packages explicitly:
```bash
uv pip install --upgrade "vllm>=0.20" "transformers>=4.58" "torch>=2.11" "peft>=0.19"
```

## Quickstart

Two end-to-end recipes to verify your setup. Both train Qwen2.5-0.5B-Instruct on
GSM8K with LoRA r=16, DAPO + RLOO loss.

### Colocated — single GPU (trainer + vLLM share one device)

```bash
python -m vivace.scripts.train \
    --config vivace/configs/dapo_gsm8k_qw25_0.5b_lora_colo.yaml \
    --num-steps 200 \
    --run-dir "runs/colo_$(date +%Y%m%d_%H%M%S)"
```
~440s wall-clock on a 4090, baseline 0% → 43% accuracy in 200 steps.
Uses `weight_sync_method: ipc` (CUDA IPC zero-copy on the same GPU).

### Disaggregated — two GPUs (trainer on GPU 0, vLLM on GPU 1)

```bash
python -m vivace.scripts.train \
    --config vivace/configs/dapo_gsm8k_1.5b_lowkl.yaml \
    --num-steps 200 \
    --run-dir "runs/disagg_$(date +%Y%m%d_%H%M%S)"
```
Qwen2.5-1.5B + LoRA. Uses `weight_sync_method: nccl` (direct GPU→GPU broadcast).

### Smoke test

```bash
python -m vivace.scripts.train \
    --config vivace/configs/dapo_gsm8k_qw25_0.5b_lora_colo.yaml \
    --num-steps 5 \
    --run-dir "runs/smoke_$(date +%Y%m%d_%H%M%S)"
```

`WANDB_MODE=disabled` if you want to skip wandb. See
[docs/running_experiments.md](docs/running_experiments.md) for the full launch
pattern, sync-method overrides, and comparison commands.

## Layout

```
vivace/                    # installable package
├── algos/                 # Training methods (the math)
│   ├── types.py           # RolloutBatch, RLConfig, SFTConfig, preset factories
│   ├── policy_gradient.py # Composable PG zoo (GRPO, DAPO, GSPO, RLOO, CISPO, DG, ...)
│   └── sft.py             # Optional SFT warmup
├── envs/                  # Benchmark wrappers behind a common Env interface
│   ├── base.py            # Env ABC + Example
│   └── gsm8k.py
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
└── scripts/               # CLI entry points (vivace-train, vivace-sft, vivace-eval)
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

Weight sync: `ipc` (default for colocated). NCCL doesn't work for two ranks on
the same GPU, so disk is the only fallback.

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

| where you are | use |
|---|---|
| 1 GPU, debugging | `dapo_gsm8k_qw25_0.5b_lora_colo.yaml` (IPC, 0.5B) |
| 1 GPU, more capacity | `dapo_gsm8k_qw35_0.8b_lora_colo.yaml` (IPC, 0.8B Qwen3.5) |
| 2 GPUs, faster | `dapo_gsm8k_1.5b_lowkl.yaml` (NCCL, 1.5B) |
| 2 GPUs, full FT | `grpo_gsm8k_0.5b_full_nccl.yaml` (NCCL, 0.5B no-LoRA) |

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

1. Create `vivace/envs/my_bench.py` subclassing `Env` from `vivace/envs/base.py`.
2. Implement `load_split()`, `format_prompt()`, and link to a reward function.
3. Point a config at it.

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

# Compare two runs (per-step time, throughput, cumulative wall, training reward)
python -m tests.compare_sync_perf    --a runs/<a> --b runs/<b> --out runs/cmp.png
python -m tests.compare_sync_perf_n  --runs runs/<a> runs/<b> runs/<c> \
                                     --labels A B C --out runs/cmp.png
```

See [`docs/weight_sync_approaches.md`](docs/weight_sync_approaches.md) for
the design rationale and failure-mode diagnostics.

## Roadmap

- [x] Composable PG zoo: GRPO / DAPO / GSPO / RLOO / Dr.GRPO / CISPO / DG
- [x] GSM8K env + reward functions
- [x] Single-GPU training with HF sampler
- [x] vLLM rollout backend with LoRA hot-swap (disk + NCCL + CUDA IPC paths)
- [x] Disaggregated mode (separate rollout + trainer GPUs)
- [ ] DDP / FSDP distributed training (in progress)
- [ ] MATH-500 + AIME + Omni-MATH + DeepMath-103K envs
- [ ] Optional SFT warmup path
- [ ] Agentic tasks: tool-use rollout hooks (calculator, Python, search)
- [ ] RL with self-distillation
- [ ] VLM support (vision-language model training)
- [ ] Speedrun configs + leaderboard
