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

vLLM needs a CUDA toolkit; use `uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu124`
if the default resolve breaks.

### Upgrading an existing venv

If you cloned earlier and need newer `vllm` / `transformers` (e.g. for the `qwen3_5`
architecture), run:

```bash
uv pip install -e ".[dev]" --upgrade
```

Or upgrade the load-bearing pair specifically:

```bash
uv pip install --upgrade "vllm>=0.20" "transformers>=4.58"
```

## Quickstart

```bash
# GSPO+RLOO on GSM8K with Qwen3.5-0.8B-Base, LoRA, single GPU
vivace-train --config vivace/configs/gspo_gsm8k_0.8b_lora.yaml

# Smaller model for quick iteration
vivace-train --config vivace/configs/grpo_gsm8k_0.5b_lora.yaml

# Full fine-tuning instead of LoRA
vivace-train --config vivace/configs/grpo_gsm8k_0.5b_full.yaml

# Dry run to sanity-check the full pipeline
vivace-train --config vivace/configs/grpo_gsm8k_0.5b_lora.yaml --num-steps 5
```

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
tests/                     # Unit tests + integration scripts (weight-sync verify, NCCL smoke)
docs/                      # Implementation notes
```

## Execution modes

### Colocated

Each GPU runs both training and rollout, time-multiplexed. On a rollout
step, the trainer releases activations, vLLM generates completions,
then vLLM's KV cache is released and training resumes. Slower
(no overlap between generation and training) but uses fewer GPUs.

```
GPU 0: [ trainer shard 0 ] <-> [ vllm worker 0 ]
GPU 1: [ trainer shard 1 ] <-> [ vllm worker 1 ]
```

### Disaggregated

Rollout and training run on separate GPUs. The trainer pushes updated
weights to the vLLM workers over NCCL after each optimizer step.
Rollouts and gradient updates can overlap — one batch is generating
while the previous batch is being trained on.

Simple setup (2+2):
```
GPUs 0-1: [ vllm tp=2 rollout worker  ] --generations-->
GPUs 2-3: [ FSDP trainer dp=2         ] <--weight sync--
```

Larger setup (2+4+2):
```
GPUs 0-1: [ vllm tp=2 rollout worker  ] --generations-->
GPUs 2-5: [ FSDP trainer dp=4         ] <--weight sync--
GPUs 6-7: [ eval workers (optional)   ] --accuracy, pass@k-->
```

The optional eval workers run continuous evaluation (greedy pass@1 and
sampled pass@k / maj@k) on the latest checkpoint without blocking the
training loop. They use their own vLLM instance for fast inference.

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

## Testing weight sync

Two scripts. See `docs/weight_sync_approaches.md` §Testing for details and
failure-mode diagnostics.

```bash
# 1. Low-level smoke test: scalar NCCL broadcast between trainer + vLLM worker
.venv/bin/python -m tests.test_nccl_sync

# 2. End-to-end: 3-step verify (fresh / perturb / sync) against the backend under test
.venv/bin/python -m tests.test_weight_sync --config <path> --method disk
.venv/bin/python -m tests.test_weight_sync --config <path> --method nccl
```

## Roadmap

- [x] Composable PG zoo: GRPO / DAPO / GSPO / RLOO / Dr.GRPO / CISPO / DG
- [x] GSM8K env + reward functions
- [x] Single-GPU training with HF sampler
- [ ] MATH-500 + AIME + Omni-MATH + DeepMath-103K envs
- [ ] vLLM rollout backend with LoRA hot-swap
- [ ] Disaggregated mode (separate rollout + trainer GPUs)
- [ ] DDP / FSDP distributed training
- [ ] Optional SFT warmup path
- [ ] Agentic tasks: tool-use rollout hooks (calculator, Python, search)
- [ ] RL with self-distillation
- [ ] VLM support (vision-language model training)
- [ ] Speedrun configs + leaderboard
