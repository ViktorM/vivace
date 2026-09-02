# Cluster support — design + provider choice

How to scale vivace beyond the local 2× 4090 to multi-node H100/B200 jobs, and
which provider to use. Companion to [`docs/architecture.md`](architecture.md)
(process model) and [`docs/weight_sync_approaches.md`](weight_sync_approaches.md).

## Why we need this

The 2× 4090 local box has carried the 5-algo × 3-seed v1 benchmark
(GRPO/DAPO/GSPO/CISPO/Dr.GRPO, `docs/v1_results.md`) at Qwen2.5-1.5B + LoRA
r=16. Beyond that, we're hitting:

- **Effective batch ceiling** — 32 trajectories/gradient step (bs=1 × gs=8 ×
  accum=4) is small. Most observed RL-instability patterns at this scale
  resolve with bigger batches. Paper recipes (MiniMax CISPO at 512 H800s,
  DeepSeek R1) need cluster-class effective batches to reproduce faithfully.
- **Model ceiling** — Qwen2.5-7B + LoRA fits one H100 but not one 4090. Full-FT
  7B and any larger model needs FSDP across multiple GPUs.
- **Seed sweeps** — a 200-step Math run is ~53 min; the 15-run v1 matrix took
  ~13 h serial. Cluster runs them in parallel for ~the same wall.

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
single-H100 iteration), the cs336 list's other reserved-only options.

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

Local 2× 4090 baseline (group=8, batch=1, grad_accum=4, max_new=768): ~2.5–3h
wall for 500 steps before the June 2026 perf fixes; ~53 min per 200-step run
now. At cluster scale we can push group_size + batch_size
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

### Phase 1 — Runpod 2× 80GB (done May 2026)

We skipped a 1× disagg baseline because real vivace runs are at least 2-GPU
(disagg: trainer + vLLM on separate cards), matching the desktop topology;
`math/cispo_1x80GB_colo.yaml` is the 1× colocated point for Phase 2 instead.

1. `docker/Dockerfile` — `nvidia/cuda:13.0.1-devel-ubuntu24.04` for all GPU
   types (cu13 covers B200 sm_100; one base keeps H100/H200/B200 identical);
   `uv sync --frozen` from `uv.lock` (torch 2.13.0+cu130 + vLLM 0.28.0). **Stay on
   `devel`**: vLLM/flashinfer JIT-compile sm_100 kernels via nvcc at first
   model load, which `runtime` lacks — H100 works there, B200 crashes.
   `HF_HOME=/workspace/hf_cache`, `WANDB_DIR=/workspace/wandb` are baked in;
   `WANDB_API_KEY` comes from the pod env. `docker/entrypoint.sh` starts sshd
   from Runpod's `PUBLIC_KEY`, writes the venv env to `/etc/profile.d/` (SSH
   logins don't inherit Docker `ENV`), then `sleep infinity` — training is
   launched by hand over SSH.
2. `docker/push.sh <version> [--latest] [--yes]` pushes
   `ghcr.io/<user>/vivace:<version>` + `:<git-sha>`; `:latest` only on request.
3. Runpod template + 500 GB network volume mounted at `/workspace`
   (`checkpoints/`, `wandb/`, `hf_cache/`).
4. Validated on 2× A100 80GB SXM with `math/cispo_2x80GB.yaml` (disagg, bs=8,
   gs=8, accum=1, max_new=1024, GC on): 27 s/step. Recipe in
   `docs/runpod_quickstart.md`.

### Phase 2 — 1/2/4/8-GPU scaling study (configs landed 2026-05-18; curve not yet run)

`math/cispo_{1x,2x,4x,8x}80GB_colo.yaml` keep per-rank work identical (bs=4,
gs=8, accum=2 → 64 traj/rank/step), so global batch = 64 × ranks: 64 / 128 /
256 / 512 traj/step (2×–16× the desktop's 32). `math/cispo_2x80GB.yaml` is the
disagg counterpart at 64 traj/step.

1. Run all four points on **one 8-GPU node**, subsetting GPUs — never four
   separate cloud invocations (5–15% per-host speed variance; the 8× point
   needs the real NVLink topology). H100 SXM for writeup numbers, A100 only
   as availability fallback.
2. Disagg DDP is 1:1 trainer:rollout by construction (trainer validation), so
   4× = `nproc_per_node=2` + 2 vLLM GPUs or `nproc_per_node=4` colo; 8× = 4 + 4
   or `nproc_per_node=8` colo. An asymmetric split needs a trainer change.
3. The 8× colo run (512 traj/step) is the direct test of "small batch is the
   cause" of the step-200 collapse seen at 32 trajectories and of the 20-pt
   seed spread at 128 (2026-05-13).
4. **2× H200** as an alternative — same compute as 2× H100, memory for a bigger
   group_size without gradient_checkpointing (~30% step-time).
5. Part of the deferred queue already ran as the May 2026 4×H200 colo set
   (Adam(0.95,1e-15), kl=0.02, ep=2/4/8 — `docs/ablation_studies.md`); rank=32
   and the rest remain.

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
- **vLLM subprocess model.** vLLM spawns an EngineCore child per rank. Need
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
