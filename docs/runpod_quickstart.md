# Runpod quickstart — first vivace cluster run

End-to-end recipe for getting a Runpod pod running the canonical CISPO math
recipe. Targets the Phase 1 milestone from
[`cluster_support.md`](cluster_support.md).

**Recommended first pod is 2× A100 80GB SXM** ($1.49/hr × 2 = **$2.98/hr**) for the
cheapest end-to-end validation. After the recipe reproduces there, scale to
2× H100/H200/B200 for real experiments.

## Prerequisites (one-time)

### 1. Runpod account

Sign up at [runpod.io](https://www.runpod.io/), confirm email, and add a
payment method. Load $20–50 in credits to start — enough to cover the Phase 1
smoke + first 200-step run with margin. Runpod bills by the second, so unused
credits stay on the account.

Once signed in, the **Settings** page collects the three things you'll need
below: SSH public keys, API keys (for `runpodctl`), and container-registry
auth (only if you keep your image private).

### 2. SSH key for Runpod

Runpod's SSH only authenticates *you* into the pod — it doesn't grant access to
any GitHub repo. You can reuse an existing key or generate a new one. **Read-only
vs read-write isn't a thing for SSH keys themselves** — that distinction applies
to GitHub *deploy keys* (which we don't use here; the working tree is `COPY`'d
into the image at build time).

If you don't already have an `id_ed25519` pair:

```bash
# The -C string is just a comment embedded in the key for your own bookkeeping
# (e.g. "viktor@desktop-3090", "research-machine-A") — not used for auth.
ssh-keygen -t ed25519 -C "$(whoami)@runpod" -f ~/.ssh/runpod_ed25519
# (press Enter twice for no passphrase, or set one — Runpod's SSH client
#  supports both)
```

Your **public key** is in `~/.ssh/runpod_ed25519.pub` (the `.pub` file, not the
private one). View it:

```bash
cat ~/.ssh/runpod_ed25519.pub
# ssh-ed25519 AAAA... <your-comment>
```

In Runpod: **Settings → SSH Public Keys → Add Public Key** → paste the entire
`ssh-ed25519 AAAA... <your-comment>` string. Give it a recognizable name (e.g.
`runpod-desktop`) so you can revoke it later if needed. The same Settings page
has the **API Keys** section next to it — that's where the `runpodctl` token
comes from (different from the SSH key).

If you'd rather reuse an existing key: cat `~/.ssh/id_ed25519.pub` (or
`id_rsa.pub`) and paste that. Any one key works; a per-service key is the
security-textbook recommendation but not required.

### 3. GHCR (GitHub Container Registry) setup

GHCR is free, ties to your existing GitHub account, and works well with
Runpod's container-image-from-URL flow.

**Create a Personal Access Token (PAT) with `write:packages` scope:**

1. Open https://github.com/settings/tokens (or **GitHub → Settings →
   Developer settings → Personal access tokens → Tokens (classic)**).
2. **Generate new token (classic)**. Name it `ghcr-vivace`. Set expiration to
   90 days (or however long you want).
3. **Scopes to check:**
   - `write:packages` (push images)
   - `read:packages` (pull images — Runpod will need this if the image is
     private)
   - `delete:packages` (optional, lets you `docker rmi` from the registry)
4. **Generate token**, copy the `ghp_...` string immediately (GitHub only
   shows it once).

**Log Docker into GHCR locally:**

```bash
# Save the PAT to your shell env so it's not in command history
export GITHUB_PAT='ghp_xxxxxxxxxxxxxxxxxxxxxxxx'

# Log Docker in. The username for `docker login` is case-sensitive
# (matches your GitHub handle); the image-path uses lowercase only — keep
# both as one lowercase string and you'll never trip over it.
export GH_USER=<your-github-username-lowercase>
echo "$GITHUB_PAT" | docker login ghcr.io -u "$GH_USER" --password-stdin
# Login Succeeded
```

Credentials are saved to `~/.docker/config.json` so subsequent pushes don't
need this again.

**Make the image public** (so Runpod doesn't need credentials to pull). After
the first push, visit
`https://github.com/<your-github-username>?tab=packages` → click the `vivace`
package → **Package settings** → **Change visibility** → **Public**. (Private
also works if you give Runpod the PAT — see "Pulling private images on Runpod"
below.)

### 4. (Optional) `runpodctl` CLI

```bash
# macOS
brew install runpod/runpodctl/runpodctl

# Linux
wget -q -O - https://api.github.com/repos/runpod/runpodctl/releases/latest \
    | grep "browser_download_url.*linux.*amd64" | cut -d '"' -f 4 \
    | xargs wget -O runpodctl && chmod +x runpodctl && sudo mv runpodctl /usr/local/bin/

# Authenticate (get API key from Runpod → Settings → API Keys)
runpodctl config --apiKey <key>
```

## Build + push the image

### Tagging strategy

Don't use `:latest` for everything. Tag the image so each push is
identifiable and rollback-able:

| tag | purpose |
|---|---|
| `vivace:test` | First push for sanity-checking the build itself (before any cloud spin-up). Iterate locally with this tag. |
| `vivace:v0.1.0` | Tagged release — bump for each meaningful change to dependencies, Dockerfile, or stack |
| `vivace:<git-sha>` | One per commit for full reproducibility. The pod that ran experiment X used image `vivace:abc1234` |
| `vivace:latest` | Alias for the most recent stable tag. Only updated after a `vivace:v0.x.y` proves itself |

In practice for solo work: tag with both `:vX.Y.Z` and `:<git-sha>` on every
push; reserve `:latest` for the version you'd let someone else use.

### First build (local sanity)

```bash
# From the repo root (wherever you cloned vivace)
cd /path/to/vivace

# Build with the `:test` tag first — just confirms the Dockerfile is valid
docker build -f docker/Dockerfile -t vivace:test .
# ~15 min first time (CUDA base download + uv sync of vLLM/torch).
# Subsequent rebuilds with cached layers are ~2 min if only source changed.
```

If the build fails (most likely on `uv sync` for some dependency), fix the
Dockerfile and retry — `:test` is meant for this kind of iteration.

### Tag + push to GHCR

Use `docker/push.sh` — derives `<git-sha>` from `HEAD`, lowercases the
image path automatically, and warns if your working tree has uncommitted
changes (which would make the `<git-sha>` tag misleading).

```bash
docker/push.sh v0.1.0              # push :v0.1.0 and :<git-sha>
docker/push.sh v0.1.0 --latest     # also push :latest if this is stable
docker/push.sh v0.1.0 --yes        # no dirty-tree prompt
GH_USER=foo docker/push.sh v0.1.0  # override the default username
```

The default `GH_USER` is hard-coded in the script — edit `docker/push.sh`
once to set it to your own GitHub handle (lowercase).

The first push is the slow one (~10-20 min for ~10 GB on residential upload).
Subsequent pushes only send changed layers — typically seconds to minutes
depending on how much changed.

### Verify on GHCR

Visit `https://github.com/<your-github-username>?tab=packages` and you should
see `vivace` listed with the tags you pushed. Click in to confirm tag list +
visibility.

## Create a Network Volume (one-time)

Volumes persist across pod restarts; use one for checkpoints + the HF cache so
re-launching a pod doesn't re-download 3GB of Qwen weights.

Web UI → Network Volumes → Create:
- **Size:** 500 GB
- **Datacenter:** same one you'll launch pods in (e.g. CA-OR-1 — pick one with
  H100 SXM Secure stock; check the pod creation page for current availability)
- **Name:** `vivace-shared`

Cost: 500 GB × $0.07/mo = $35/mo (billed monthly even when no pods are running;
delete + recreate if you won't use it for >2 weeks).

## Launch a pod

### First pod: 2× A100 80GB SXM (cheapest validation, ~$2.98/hr)

Use this for the first end-to-end test — same 80 GB VRAM as H100 (so the
existing 2× 80GB config runs unchanged) at ~half the compute. ~$3/hr vs ~$6/hr
at half the speed: a full run costs about the same; the saving is on smoke
tests, debugging and idle time.

**Web UI path:**

1. **Deploy** → **GPU Pod** → Filter by `A100 SXM` and **Secure Cloud**.
2. Select **2× A100 80GB SXM** ($1.49/hr each).
3. **Pod Template** → **Edit Template**:
   - **Container Image:** `ghcr.io/<your-github-username>/vivace:v0.1.0` (use
     the exact tag you pushed; avoid `:latest` unless you've promoted that tag).
   - **Container Registry Credentials:** only needed if the GHCR package is
     private — see "Pulling private images" below.
   - **Container Disk:** 50 GB (image expands here; ~15 GB image needs ~30 GB
     unpacked).
   - **Volume:** attach the `vivace-shared` Network Volume; mount path
     `/workspace`.
   - **Expose:** TCP 22 (SSH). Some templates also expose 8888 for Jupyter —
     not needed here.
   - **Environment Variables:** add via the UI, or via `--env` if using CLI:
     - `WANDB_API_KEY=<your wandb key>` — wandb sync
     - `HF_TOKEN=<your hf token>` — only needed for gated models
4. **Deploy On-Demand**. Secure pod boots in ~60–90 sec.

**CLI path** (after `runpodctl config`):

```bash
runpodctl create pod \
    --name vivace-2xa100 \
    --image ghcr.io/<your-github-username>/vivace:v0.1.0 \
    --gpu-type "NVIDIA A100 80GB PCIe" --gpu-count 2 \
    --container-disk-in-gb 50 \
    --network-volume-id <volume-id-from-web> \
    --env WANDB_API_KEY=$WANDB_API_KEY
```

(GPU-type string for SXM in `runpodctl` is currently spelled `"NVIDIA A100-SXM4-80GB"` —
double-check the latest in `runpodctl get gpu-types`.)

### Real-experiment pod: 2× H100 80GB SXM (~$5.98/hr)

Once A100 validation passes, swap to H100 SXM for the actual research runs.
Same template, just change the GPU type:

- Web UI: filter `H100 SXM` + `Secure Cloud`, select 2× H100 80GB SXM.
- CLI: `--gpu-type "NVIDIA H100 80GB HBM3" --gpu-count 2`.

### Pulling private images on Runpod

If you kept the GHCR package private, Runpod needs your PAT to pull:

1. Runpod **Settings → Container Registry Auth → Add Registry**.
2. Registry URL: `ghcr.io`, Username: `<your-github-username>`, Password: paste
   your PAT.
3. In the pod template, select this credential under "Container Registry
   Credentials".

Easier: just make the package public (one-time setting on the GHCR side, no
ongoing credential management).

## Connect + run

Once the pod is "Running", grab its SSH command from the web UI (Connect → SSH
over exposed port). You'll get something like:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

Inside the pod (you're now in `/opt/vivace` with the venv activated):

```bash
# Pull the latest code (image was built at push time; the running repo
# inside the container may be behind by a few commits)
git pull
uv sync --no-dev --frozen  # only re-runs if uv.lock changed

# Sanity smoke — 5-step run to confirm everything wires up
torchrun --nproc_per_node=1 -m vivace.scripts.train \
    --config vivace/configs/math/cispo_2x80GB.yaml \
    --num-steps 5 \
    --run-dir /workspace/checkpoints/smoke_$(date +%Y%m%d_%H%M%S)

```

### Colocated vs disaggregated on the same 2 GPUs

Both modes consume the same hardware (2 GPUs of equal class) but lay the work
out differently. Disaggregated pins the trainer to GPU 0 and the vLLM rollout
engine to GPU 1 — they communicate via a tensor weight push each step.
Colocated runs trainer + vLLM together on every rank (vLLM sleeps during
backward, wakes for the next rollout) and uses DDP all-reduce across ranks.

|                       | Disaggregated                       | Colocated                                |
|---|---|---|
| Config                | `math/cispo_2x80GB.yaml`            | `math/cispo_2x80GB_colo.yaml`            |
| `--nproc_per_node`    | `1`                                 | `2`                                      |
| GPU 0                 | trainer                             | trainer + vLLM (sleep/wake)              |
| GPU 1                 | vLLM rollout                        | trainer + vLLM (sleep/wake)              |
| Trajectories / step   | `bs × gs × accum = 8×8×1 = 64`      | `bs × gs × accum × ranks = 4×8×2×2 = 128`|
| Cross-device comms    | weight push trainer → vLLM per step | DDP grad all-reduce per step             |

Replace `<hw>` with your actual hardware tag (e.g. `2xa100`, `2xh100`,
`2xrtx6000pro`).

```bash
# Disaggregated — 1 trainer + 1 vLLM, --nproc_per_node=1
torchrun --nproc_per_node=1 -m vivace.scripts.train \
    --config vivace/configs/math/cispo_2x80GB.yaml \
    --wandb-group cispo_math_qw25_1.5b_ep2_lr2e5_disagg_<hw> \
    --run-dir /workspace/checkpoints/cispo_math_disagg_<hw>_$(date +%Y%m%d_%H%M%S)

# Colocated — DDP across ranks, trainer + vLLM on every rank, --nproc_per_node=2
torchrun --nproc_per_node=2 -m vivace.scripts.train \
    --config vivace/configs/math/cispo_2x80GB_colo.yaml \
    --wandb-group cispo_math_qw25_1.5b_ep2_lr2e5_colo_<hw> \
    --run-dir /workspace/checkpoints/cispo_math_colo_<hw>_$(date +%Y%m%d_%H%M%S)
```

The two configs ship **different global batches by default** (64 vs 128
trajectories/step). That is intentional — each layout has its own
memory/throughput sweet spot, and the goal of the comparison is to find the
best operating point per mode, not to force a batch-matched ablation. The same
logic applies as we scale up: on 8-GPU runs the GPU split itself becomes a
tuning knob (e.g. 2 rollout + 6 trainer; needs a trainer change — disagg is
1:1 today), with different batch sizes on each side. Compare best-of-mode vs
best-of-mode, not matched-batch curves.

## Verify GPU memory utilization

After launching a training run, check that both GPUs are well-subscribed.
Headroom means you can bump batch size further for faster training; pinned
memory means the config is at its limit (don't push further or you'll OOM).

### Live `nvidia-smi` watch

In a second SSH session into the pod:

```bash
nvidia-smi -l 2  # refreshes every 2 sec
```

Look at the **Memory-Usage** column for both GPUs during a step:

| | what you want to see | what's bad |
|---|---|---|
| **GPU 0 (trainer)** | 50–70 GB used (out of 80) during the train phase peak; drops to ~30 GB between steps | Pinned at 78–80 GB → OOM risk, dial back; under 30 GB → under-utilized, bump batch_size |
| **GPU 1 (vLLM rollout)** | 64 GB used continuously (= 0.8 × 80 GB pre-allocated by vLLM); the **actual KV usage** isn't visible here — only the reservation | The 64 GB is always shown — that's the reservation, not the active KV; see below to check the real KV utilization |

The trainer-side GPU is the one to watch in nvidia-smi. vLLM pre-allocates its
budget once at startup, so the nvidia-smi number is misleading there.

### Check trainer-side peak per step (in the training log)

vivace logs the trainer-side allocator state on every log line:

```
[cispo/rloo LoRA r=16 ...] Step 0050 | ... | 1525 tok/s 19s/step | 1395s
    alive=100.0% spread_mean=2.475 ... mem_alloc=4.89G mem_res=4.95G mem_peak=12.50G
```

- `mem_alloc` — current PyTorch tensors in use (~5 GB at idle for 1.5B)
- `mem_res` — PyTorch's reserved allocator pool
- `mem_peak` — peak since last reset (between this step and the previous);
  this is the number to watch

**`mem_peak` should land in the 50–70 GB range during the forward+backward**
on 80 GB. `cispo_2x80GB.yaml` (bs=8, gs=8, max_new=1024) peaks ~52 GB with GC
on, ~78 GB without. If you see 25–35 GB consistently → trainer is
under-subscribed, bump `batch_size` or `group_size`. If you see 75+ GB or hit
OOM → reduce `batch_size` (GC is already on in the disagg config; the colo
configs run GC off — flip it there first).

### Check vLLM-side actual KV utilization

vLLM logs a line at engine init with the computed KV cache budget:

```
INFO ... [worker.py] Maximum concurrency for 1280 tokens per request: 92.5x
INFO ... [executor_base.py] # cuda blocks: 24576, # CPU blocks: 0
```

`Maximum concurrency for 1280 tokens` means vLLM can hold 92 concurrent
sequences at that context length within the pre-allocated budget. If that
number is much higher than `vllm_max_num_seqs` (256), KV is over-provisioned
relative to what we ask for — fine for headroom, but we won't actually use it
unless we push concurrency higher.

To see real KV usage during a run, vLLM emits periodic engine stats:

```
INFO ... [metrics.py] Engine 000: Avg prompt throughput: 1234 tok/s, Avg
generation throughput: 5678 tok/s, Running: 64 reqs, Swapped: 0 reqs,
Pending: 0 reqs, GPU KV cache usage: 18.2%
```

**`GPU KV cache usage:` is the one to watch.** If it stays below ~50% during
the rollout phase, vLLM has plenty of room and we can either:
- Bump `vllm_max_num_seqs` (more concurrency at the same context length)
- Bump `vllm_max_model_len` (longer sequences for math/AIME)
- Bump trainer-side `group_size` / `batch_size` (more rollout demand)

If it pins at 95%+ vLLM is the bottleneck — bigger `gpu_memory_utilization`
is the easiest fix (currently 0.8; can push to 0.9 on a dedicated rollout
GPU since nothing else is using that card).

### Rule of thumb for the first run

Run a 10-step smoke, watch both `mem_peak` (trainer) and the vLLM engine
stats once they print. Decide from those numbers whether the next run should
bump batch_size further. The current configs (64 traj/step disagg, 128 colo)
are conservative starting points assuming ~50% utilization; you'll likely have
headroom on both sides to push to 192 or 256.

## Monitor + tear down

- **wandb** — runs auto-sync if `WANDB_API_KEY` is set. Group is the
  `--wandb-group` above: `cispo_math_qw25_1.5b_ep2_lr2e5_{disagg,colo}_<hw>`.
- **Live metrics in pod** — trainer logs to stdout only, so launch inside tmux
  (`tmux_quickstart.md`) and reattach, or `nvidia-smi -l 2`.
- **Stop the pod** when training finishes (web UI → Stop). Stopped pods cost
  $0 for compute; the network volume keeps billing at ~$1.20/day.
- **Delete the pod** if you won't use it for >24h; the container disk
  ($0.10/GB/mo) keeps billing until deletion.

## Cost rough budget for Phase 1

| activity | 2× A100 SXM (~$3/hr) | 2× H100 SXM (~$6/hr) |
|---|---|---|
| First image push (one-time) | $0 (your bandwidth) | $0 |
| Smoke test (~5 min) | <$0.30 | <$0.50 |
| CISPO 200-step run (27 s/step measured on 2× A100 → ~90 min; H100 ~half) | ~$4.50 | ~$4.50 |
| Network volume per month (idle) | $35 | $35 |
| 3-seed sweep × full run | ~$13.50 | ~$13.50 |

Suggested order: do the smoke + first 200-step run on 2× A100 (~$5 total) to
validate the image and config. Then switch to 2× H100 for the actual research
runs and seed sweeps.

For Phase 2 (4× / 8×), bump `--gpu-count` and use
`math/cispo_{4x,8x}80GB_colo.yaml` (`--nproc_per_node=4`/`8`, 256/512 traj/step);
run all scaling points on one 8-GPU node, subsetting GPUs (`cluster_support.md`).

## Troubleshooting

- **`ImportError: libcuda.so.1`** — pod doesn't have GPUs attached; verify
  the template requests GPUs.
- **Slow first download of Qwen weights** — point `HF_HOME=/workspace/hf_cache`
  (already set in the Dockerfile) so subsequent pods reuse the volume.
- **vLLM sleep/wake assertion (`freed_bytes >= 0`)** — colocated only: the
  trainer's allocator pool grows over steps and squeezes vLLM's; the trainer
  calls `empty_cache()` around sleep/wake to hold it off. Never set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` in colo — it breaks vLLM sleep.
- **NCCL timeout on weight sync** — check that both `trainer_gpus` and
  `rollout_gpus` are on the same node. Multi-node NCCL needs additional
  setup (Phase 4).
