# Cluster support — design + provider choice

How to scale vivace beyond the local 2× 4090 to multi-node H100/B200 jobs, and
which provider to use. Companion to [`docs/architecture.md`](architecture.md)
(process model) and [`docs/weight_sync_approaches.md`](weight_sync_approaches.md).

## Why we need this

The 2× 4090 local box has worked for the GRPO/DAPO/CISPO algo zoo at
Qwen2.5-1.5B + LoRA r=16. Beyond that, we're hitting:

- **Effective batch ceiling** — 8 trajectories/gradient step is small. Most
  observed RL-instability patterns at this scale resolve with bigger batches.
  Paper recipes (MiniMax CISPO at 512 H800s, DeepSeek R1) need cluster-class
  effective batches to reproduce faithfully.
- **Model ceiling** — Qwen2.5-7B + LoRA fits one H100 but not one 4090. Full-FT
  7B and any larger model needs FSDP across multiple GPUs.
- **Seed sweeps** — single 500-step run is ~3h. 3 seeds × N variants serial
  is intractable locally. Cluster runs them in parallel for ~the same wall.

## Decision

**Primary: Runpod Secure Cloud** for the single-node H100 work and small-cluster
runs (≤8 GPUs). Reasons:
- Cheapest credible H100/B200 on-demand ($2.39–3.39/hr H100 80GB).
- Bring-your-own-Docker — vivace's existing uv venv builds into a docker image
  cleanly without rewriting the launch surface.
- Network volumes for checkpoints persist across pods at $0.07/GB/mo.
- Instant Clusters give us multi-node InfiniBand (1,600–3,200 Gbps RoCEv2) at
  3-min spin-up when we need >8 GPUs.

**Secondary: Modal** for cheap idle dev iteration + seed-sweep ergonomics. The
`modal run train.py` / `.map(seeds)` story is best-in-class. Zero idle billing
means dev cycles cost nothing. Worth setting up once vivace boots cleanly on
Runpod and we have a working H100 baseline to port.

**Nebius preemptible B200 ($3.05/hr)** is a dark-horse third option — B200
speed at H100 price, with EU-only footprint and eviction risk. Park as
"consider for long batch jobs that tolerate restart."

**Skip:** Lambda (reservations-only for clusters), Together (8-GPU minimum kills
single-H100 iteration), Together / cs336 list's reserved-only options.

## Per-axis comparison (Modal vs Runpod)

### Pricing by GPU type and count

Per-hour, on-demand (Secure tier on Runpod). Spot/preemptible variants in parentheses
where available. Numbers from public pricing pages as of 2026-05-13; verify before launch.

| GPU type | 1× | 2× | 4× | 8× | multi-node 16× |
|---|---|---|---|---|---|
| **A100 80GB SXM (Runpod)** | $1.49 | $2.98 | $5.96 | $11.92 | — |
| **A100 80GB PCIe (Runpod)** | $1.39 | $2.78 | $5.56 | $11.12 | — |
| **H100 80GB SXM (Runpod)** | $2.99 | $5.98 | $11.96 | $23.92 | $47.84 (Instant Cluster) |
| **H100 80GB (Modal)** | $3.95 | $7.90 | $15.80 | $31.60 | $63.20 (`@clustered` beta) |
| **H200 141GB (Runpod)** | $3.59 | $7.18 | $14.36 | $28.72 | $57.44 |
| **H200 141GB (Modal)** | $4.54 | $9.08 | $18.16 | $36.32 | $72.64 |
| **B200 192GB (Runpod)** | ~$5.99 | ~$11.98 | ~$23.96 | ~$47.92 | sales-only |
| **B200 192GB (Modal)** | $6.25 | $12.50 | $25.00 | $50.00 | `@clustered` beta |
| **B200 preemptible (Nebius)** | $3.05 | $6.10 | $12.20 | $24.40 | — |
| **RTX PRO 6000 96GB Blackwell** | not on Runpod yet; some smaller clouds list ~$1.50–2.00/hr (verify, 2025–26 release) | | | | |

**Choosing GPU type for vivace workloads:**

- **A100 80GB** — cheapest credible option for first end-to-end validation
  ($1.49/hr SXM, $1.39/hr PCIe). Same 80 GB VRAM as H100 80GB so configs
  written for H100 just work, just ~half the compute throughput. Use for
  initial cluster smoke tests and image-pipeline validation, then switch to
  H100/H200 for the actual experiments.
- **H100 80GB SXM** — sweet spot for our 1.5B / 7B-LoRA scale. Memory limits
  become an issue at full-FT 7B or longer contexts.
- **H200 141GB** — same compute as H100 but ~1.7× memory. Worth the +20% price when
  full-FT 7B, longer sequences (>4K), or larger effective batches matter. The big
  win in our regime: we can push group_size + batch_size further without
  gradient_checkpointing's slowdown.
- **B200 192GB** — Blackwell flagship, ~1.8–2× H100 perf + more memory. Justified
  once we move to ≥13B models or paper-scale CISPO replication. **Needs CUDA 12.8+**
  for sm_100 support; 13.x recommended.
- **RTX PRO 6000 (Blackwell, 96 GB)** — interesting price/perf for solo research:
  Blackwell uarch, more VRAM than H100 80GB at ~half the $/hr. Not on Runpod yet;
  available on smaller providers (Lambda 1-Click, FluidStack, Cudo). Worth
  revisiting in 1–2 months as availability expands.

### Other axes

| | Modal | Runpod |
|---|---|---|
| Multi-node | `@clustered` beta, ≤64 GPUs, **forces 8 GPUs/node post 2026-05-31** | Instant Clusters 2–8 nodes (≤512 via sales) |
| Network | 3,200 Gbps RoCE | 1,600–3,200 Gbps InfiniBand/RoCEv2 |
| Image | Custom `modal.Image` DSL (cached layers) | Raw Docker; closer to current vivace flow |
| Storage | `modal.Volume` (write-once/read-many) + S3 mount | Network Volumes ($0.07/GB/mo persistent) |
| Idle billing | **$0 when container exits** | Pod billed while running; stopped pods still pay storage |
| Cold start | <1s | ~30–90s pod boot, ~5 min cluster |
| Dev workflow | `modal run train.py` from laptop | SSH or `runpodctl`; prebuild + push image |
| Job fan-out | `.map`, `.spawn` — N-seed sweep trivial | Manual via REST API or `runpodctl` |

### Wall-time and cost estimates (CISPO 500-step math, 1.5B)

Local 2× 4090 baseline is ~2.5–3h wall (current config: group=8, batch=1,
grad_accum=4, max_new=768). At cluster scale we can push group_size + batch_size
much higher, reduce grad_accum, and the rollout itself runs ~2–3× faster on
H100/H200/B200. Estimates assume we tune the recipe accordingly.

| setup | est. wall | cost on Runpod | cost on Modal | typical model |
|---|---|---|---|---|
| **2× H100 SXM** (trainer + vLLM disagg) | ~30 min | ~$3 | ~$4 | 1.5B LoRA |
| **2× H200** (more memory for bigger batch) | ~25 min | ~$3 | ~$4 | 1.5B–7B LoRA |
| **2× B200** (faster + memory) | ~15–20 min | ~$3–4 | ~$4 | 1.5B–7B LoRA |
| **4× H100 SXM** (DDP×2 + disagg) | ~25 min | ~$5 | ~$7 | 1.5B–7B LoRA |
| **8× H100 SXM** (DDP×4 + disagg, or full DDP) | ~30 min | ~$12 | ~$16 | 7B–13B FT |
| **8× B200** | ~20 min | ~$16 | ~$17 | 13B+ FT |
| **2-node × 8× H100** (16 GPUs) | ~25 min | ~$20–25 | ~$26–32 | 30B+ FT or paper-scale ablations |

These are rough — actual will depend on whether we bottleneck on rollout vs
training, how much we push effective batch, and the model size. **Scaling is
not linear**: at higher GPU counts we tend to train *bigger* models, not run the
same small model faster — the 1.5B-LoRA recipe saturates on rollout long before
8 GPUs help. Use the 2× tier for current model scale; jump to 4–8 only when
moving to 7B+ or running multi-recipe ablations.

**Seed sweeps + ablations get cheap fast at cluster scale**: a 3-seed × 5-variant
matrix (15 runs) at 2× H100 is ~$45 total.

## Implementation plan

### Phase 1 — Runpod 2× H100 (week 1, ~1 day work)

We skip the 1× H100 baseline because real vivace runs are at least 2-GPU (disagg:
trainer + vLLM on separate cards), matching the current desktop topology.

1. Write `docker/Dockerfile`. Use `nvidia/cuda:13.1.0-devel-ubuntu24.04` for all
   GPU types — works for torch 2.11 (current vivace stack) via forward
   compatibility, works for torch 2.12 (future), and is required for B200
   (sm_100 compute capability). Single base image keeps H100/H200/B200 setups
   identical.
   Installs `uv`, copies repo, runs `uv sync` to build the venv (vLLM 0.20 +
   torch 2.11 + deps). Sets `WANDB_API_KEY` and `HF_HOME` via runtime env vars.
   Entrypoint: `bash` (don't auto-train; user attaches via SSH or `torchrun`
   directly).
2. Push to `ghcr.io/<user>/vivace:latest`.
3. Create a Runpod template referencing the image, attach a 500 GB network
   volume mounted at `/workspace/checkpoints` and `/workspace/wandb`.
4. Spin up a **2× H100 80GB SXM Secure** pod; `git pull`, `torchrun
   --nproc_per_node=1 ... train` with disagg (trainer on GPU 0, vLLM on
   GPU 1). Compare wall-clock + final accuracy against local 2× 4090
   numbers as a sanity check.
5. Tear down. Cost target: <$5 per smoke test, <$20 for a full 500-step run.

### Phase 2 — Runpod 4× and 8× H100 single-node (week 2, ~1 day)

Step up from 2-GPU to 4 and 8 once the 2-GPU Phase 1 baseline is reproduced.

1. **4× H100** first (`nproc_per_node=2` DDP for trainer + 2 vLLM workers, or
   `nproc_per_node=4` colocated). Test scale-up: `batch_size: 2`, `group_size:
   8`, `grad_accum_steps: 2` → 64 trajectories per gradient step (2× the
   desktop). If stable + faster, push further.
2. **8× H100** next (`nproc_per_node=4` DDP + 4 vLLM workers, or
   `nproc_per_node=8` colocated). Push to `batch_size: 4`, `group_size: 8`,
   `grad_accum_steps: 2` → 128 trajectories per gradient step (4× the
   desktop). **This is the direct test of "small batch is the cause" of the
   collapse pattern we've been hitting at 32 trajectories.**
3. Try **2× H200** as an alternative — same compute as 2× H100 but more
   memory for fitting bigger group_size without gradient_checkpointing
   (which costs ~30% step-time). Compare wall-clock at matched recipe.
4. If anything is stable that wasn't locally: run the deferred experiments
   queue (Adam/rank=32/kl=0.02/etc.) at scale to see whether they were
   variant problems or scale problems.

### Phase 3 — Modal port for seed sweep (week 3, ~2 days)

1. Convert the Dockerfile build into a `modal.Image` (`.uv_pip_install`
   layers + `.add_local_dir`).
2. Wrap the trainer entrypoint in a `@app.function(gpu="H100:8",
   timeout=14400)` decorator.
3. Use `.map([seed_42, seed_43, seed_44])` to run 3 seeds concurrently.
4. Use a `modal.Volume` for checkpoints; commit() at the end of each run.

### Phase 4 — Instant Cluster multi-node (when justified, ~1 week)

1. Runpod Instant Cluster, 2× 8× H100 nodes (16 GPUs total).
2. Test torchrun with `--nnodes=2 --node_rank=$RANK --master_addr=...`
   inside the cluster.
3. Verify NCCL all-reduce performance across the InfiniBand fabric (target:
   ≥80% of single-node throughput at 16-GPU scale).
4. First multi-node experiment: ScaleRL-style CISPO replication with
   `optim_epochs=16` and the paper's hyperparams.

## Risks & footguns

- **Runpod Community Cloud is heterogenous** — CUDA/driver version drifts
  between machines. Always use Secure Cloud for runs you'll resume.
- **Runpod Network Volume + concurrent write = corruption.** Each run
  writes to its own subdirectory. Don't share files between concurrent pods.
- **Modal's image DSL is not raw Docker.** Some pip flags don't translate.
  Plan for a 1-2 day port; don't merge until vivace's CI passes on the
  Modal image.
- **Modal `@clustered` beta + 8-GPU-per-node enforcement (post-2026-05-31).**
  No `H100:4` clusters. For 2× 4090-equivalent dev jobs, stay local.
- **vLLM subprocess model.** Our trainer forks vLLM as a child process. Need
  to verify this works under Modal's container model (Modal has historically
  preferred single-process functions; subprocess may need explicit
  permission or a different decorator).
- **Image size.** vLLM + transformers + flash-attn + cuda + torch is ~15GB
  even before model weights. Plan for a slow first push.

## Sources

- [CS336 — Stanford LLMs class GPU compute providers](https://cs336.stanford.edu/)
- [Modal pricing](https://modal.com/pricing), [Modal multi-node clusters (Beta)](https://modal.com/docs/guide/multi-node-training)
- [Runpod pricing](https://www.runpod.io/pricing), [Runpod Instant Clusters docs](https://docs.runpod.io/instant-clusters)
- Nebius B200 preemptible pricing (per cs336 list, $3.05/hr)
- Lambda Labs / Together listed but not recommended for current phase
