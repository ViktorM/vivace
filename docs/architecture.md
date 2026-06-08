# Architecture

How vivace spawns and coordinates trainer + rollout workers, and how this compares
to other open-source RL frameworks.

## Process model: single-controller + child subprocess

Each torchrun trainer rank is a top-level Python process. It spawns a vLLM
EngineCore as a child subprocess. The trainer ranks form the only `torch.distributed`
group; vLLM subprocesses are not in it. They communicate to their parent rank
via vLLM's queue-based `collective_rpc`, not through `dist.*`.

```
torchrun (--nproc_per_node=N)
├── trainer rank 0  ──fork──→  EngineCore subprocess (rank 0's vLLM)
├── trainer rank 1  ──fork──→  EngineCore subprocess (rank 1's vLLM)
└── ...
```

`WORLD_SIZE` from the torchrun env var is just the trainer-rank count.
Weight sync (NCCL or CUDA IPC) runs trainer rank N → its EngineCore subprocess only.

## How vivace spawns vLLM

In [`vllm_worker.py`](../vivace/rollout/vllm_worker.py):

```python
old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

# vLLM's LLM(...) spawns an EngineCore subprocess via Python multiprocessing.
# The child inherits the env var above and sees only the GPUs we listed.
self.llm = LLM(model=model_name, tensor_parallel_size=tp_size, ...)

if old_visible is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
```

For TP > 1, vLLM internally spawns additional worker processes within the
EngineCore and forms its own NCCL group across them — independent of the
trainer's process group.

## Comparison: how other frameworks spawn workers

| framework | architecture | rollout backend | weight sync | async generation | notes |
|---|---|---|---|---|---|
| **vivace** | single-controller, vLLM as child subprocess | vLLM | NCCL / CUDA IPC / disk | no (sync per step) | minimal, single-process trainer |
| **TRL** ([huggingface/trl](https://github.com/huggingface/trl)) | single-controller, vLLM colocated or as separate server | vLLM | direct or HTTP | partial | mainstream HF integration; supports GRPO/DPO/PPO |
| **OpenRLHF** ([OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)) | Ray actors (trainer / rollout / ref / critic separate) | vLLM | NCCL via Ray | yes (async RLHF) | Ray + DeepSpeed; mature, popular for >70B |
| **veRL / HybridFlow** ([verl-project/verl](https://github.com/verl-project/verl)) | Ray actors with named resource pools, hybrid controller | vLLM | NCCL | yes | ByteDance; powered DAPO. Resource pools per role |
| **slime** ([THUDM/slime](https://github.com/THUDM/slime)) | three independent services (trainer / SGLang / HTTP router) | SGLang | HTTP weight push (router fans out via NCCL on inference side) | yes | ByteDance; max process isolation, multi-machine friendly |
| **Miles** ([radixark/miles](https://github.com/radixark/miles), [LMSYS post](https://www.lmsys.org/blog/2025-11-19-miles/)) | fork of slime; SGLang + Megatron-LM | SGLang | HTTP-style (via slime infra) | yes | first end-to-end FP8 sampling+training; targets large MoE post-training |
| **AReaL** ([inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)) | fully async — generation streams continuously, training consumes batches as ready | vLLM | NCCL | **fully** async | Ant Group + Tsinghua; ~2.77× speedup vs sync at parity |
| **NeMo-RL** ([NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)) | Ray-orchestrated, multiple training backends (DTensor, Megatron-Core) | vLLM, Megatron | NCCL | yes | NVIDIA; replaces NeMo-Aligner. Trained NeMotron-3-Nano-30B |
| **SkyRL** ([NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)) | layered: skyrl-train + skyrl-agent + skyrl-tx (Tinker-API backend) | vLLM (built on veRL) | NCCL | yes | Berkeley Sky Computing Lab; targets long-horizon agents (SWE-Bench) |
| **MiniMax Forge** ([MiniMax-M1](https://github.com/MiniMax-AI/MiniMax-M1)) | agent-native RL with engine-agent decoupling layer | proprietary | proprietary | yes | **internal — not open-sourced**. CISPO algorithm published; M1 weights open |
| **DeepSeek-R1 stack** ([deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)) | not open-sourced (training framework kept private) | — | — | — | only weights + algorithm (GRPO) public. Reproductions: [huggingface/open-r1](https://github.com/huggingface/open-r1), [TinyDeepSeek](https://github.com/FreedomIntelligence/TinyDeepSeek) |
| **Kimi K1.5/K2** ([MoonshotAI/Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5)) | not open-sourced | — | — | — | weights/agent code public; RL framework private. Paper describes a "simplistic RL framework" — no MCTS/value functions |

## Architectural axes

The frameworks split along three orthogonal choices:

**1. Process orchestration**
- *Single-controller* (vivace, TRL): trainer is one process tree per rank; vLLM is a child subprocess. Easiest to debug, fewest moving parts. Bottlenecked by trainer's Python serial work.
- *Ray actors* (OpenRLHF, veRL, NeMo-RL, SkyRL): each component is a Ray actor; the driver is a thin coordinator. Async-friendly, multi-host friendly, costs a Ray dependency and harder failure modes.
- *Independent services* (slime, Miles): components communicate over HTTP/gRPC. Maximum isolation, best for multi-machine, pays serialization overhead.

**2. Generation timing**
- *Synchronous* (vivace, TRL default, OpenRLHF default): rollout phase blocks training. Simple gradient semantics.
- *Asynchronous* (AReaL, OpenRLHF async, veRL, slime): generation streams; training consumes batches as they're ready. More throughput, off-policy correction needed.

**3. Inference engine**
- vLLM: dominant; PagedAttention, fused QKV, CUDA graphs.
- SGLang: slime/Miles; structured generation, RadixAttention, FP8 generation.
- Custom (Megatron generation, etc.): rare; usually for matching trainer parallelism exactly.

## Where vivace sits

Single-controller + vLLM child subprocess + synchronous generation. Same family
as TRL's colocated mode. Trade-offs we accept:
- No async generation → simpler, but bounded by serial training-rollout cycle.
- No Ray → simpler dependency tree, but harder to scale to >1 node (would need
  reorchitecting around Ray or HTTP services like slime/Miles).
- Per-rank vLLM subprocess → memory cost per rank, but each rank's weight sync
  is independent (no cross-rank coordination overhead).

The natural growth path when these constraints bind: adopt the Ray-actor pattern
(verl-style) for multi-node async RL. Until then, this layout is enough for
2–8 GPU local research and Runpod-scale single-node clusters.

## Async rollout — two implementation paths (v1.1 roadmap)

Async rollout overlaps generation for batch N+1 with training of batch N. The
trainer always lags rollout by one batch — the IS-ratio in the loss already
handles this off-policy correction (CISPO's clipped IS weight bounds the
gradient magnitude even at k=1 staleness).

Two paths considered, in increasing rewiring cost:

**Path A — `concurrent.futures.Future` wrapper (recommended for v1.1)**
- ~300-500 LOC. ~1-1.5 weeks of focused work.
- Background thread inside `vllm_worker` runs the existing `LLM(...).generate()`
  in a separate task; trainer kicks off rollout N+1 BEFORE calling `train_phase(N)`.
- Trainer main loop restructure: `submit_rollout(N+1) → train_phase(N) →
  weight_sync → wait(N+1) → repeat`. The "wait" usually returns immediately
  since rollout finished in parallel with the trainer step.
- No change to vLLM API; reuses everything we have today.
- Sufficient for k=1 staleness (one-step pipeline depth). Going deeper (k>1)
  would benefit from Path B.

**Path B — `AsyncLLMEngine`**
- ~800-1200 LOC. ~2-3 weeks.
- Switch `vllm_worker` from the synchronous `LLM(...)` to vLLM's
  `AsyncLLMEngine` (continuous request submission, async generators).
- Weight-sync lifecycle changes (engine has its own event loop and request queue).
- The "right" architecture if v1.2+ wants k>1 staleness, continuous request
  submission, or one rollout engine shared across multiple trainer DDP groups.
- Overkill for v1.1.

**What does not change under either path:**
- Loss math (already off-policy-aware via IS clip)
- Reward, dataset, optimizer
- The colo IPC path — async only matters for disaggregated mode
- KL anchor (unaffected by rollout staleness)

**Open risks to validate before committing:**
1. **NCCL weight broadcast during active rollout.** Untested whether the NCCL
   weight-sync broadcast can run while vLLM is mid-`generate()` on the
   receiving GPUs. Half-day smoke test on 1.5B before sinking into the trainer
   refactor; if vLLM holds CUDA streams in a way that blocks the broadcast,
   the design needs a barrier.
2. **vLLM prefix-cache invalidation on weight updates.** Async only helps if
   the rollout doesn't re-prefill on every weight push. Verify cache lifecycle
   plays nicely with continuous LoRA weight updates.
3. **IS-ratio drift at k=1.** With staleness, more tokens hit
   `clip_cispo_high=5.0`. Add a `median_is_ratio` / `p99_is_ratio` metric
   and watch the clip-fraction. If >5% of tokens get clipped, widen the cap.
4. **Disagg vs colo accuracy gap may persist.** Half the gap we measured at
   2+2 disagg sync (~30-40% slower convergence) is from gradient-variance
   in 2-way DDP all-reduce vs 4-way colo. Async doesn't fix that — it only
   recovers the throughput. Async disagg might end up "matched throughput to
   colo, slightly worse accuracy" — acceptable for the memory-bound case
   (14B+) but not a clear win at 7B.

**Suggested v1.1 sequence:**
1. NCCL-during-generate smoke test (half day)
2. Path A implementation (~1 week)
3. Async-vs-sync correctness validation at 1.5B / 200 steps / 3 seeds (half day)
4. Disagg throughput sweep on 4×H200, 1.5B and 7B (~1 day GPU time)

The disagg throughput sweep is the writeup figure for v1.1: "async unlocks
N× rollout throughput at disagg without accuracy regression."

## Sources

- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/pdf/2405.11143)
- [veRL/HybridFlow: A Flexible and Efficient RL Post-Training Framework](https://github.com/verl-project/verl)
- [AReaL: A Large-Scale Asynchronous RL System for Language Reasoning](https://arxiv.org/html/2505.24298v2)
- [Miles intro post — LMSYS](https://www.lmsys.org/blog/2025-11-19-miles/)
- [NeMo RL Documentation](https://docs.nvidia.com/nemo/rl/latest/index.html)
- [SkyRL — UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/project/skyrl/)
- [MiniMax-M1 announcement](https://www.minimax.io/news/minimaxm1)
- [Kimi K1.5 paper](https://github.com/MoonshotAI/Kimi-k1.5)
- [HuggingFace Open-R1 reproduction](https://github.com/huggingface/open-r1)
