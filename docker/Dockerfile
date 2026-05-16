# vivace container image for cloud GPU pods (Runpod, Modal, etc.)
#
# Stack chosen for compatibility with cloud GPU host drivers as of mid-2026:
#   - Base: CUDA 12.8 (driver >= 545 — universal on cloud A100/H100/H200).
#     cu13 needs driver >= 580 which only B200 hosts have; vivace's torch
#     pin would refuse to start on A100/H100 nodes.
#   - torch 2.10 + cu128 wheels — last torch series with a cu128 build that
#     vllm's flash-attention links against. torch 2.11 ships cu13 wheels.
#   - vllm 0.19.x — its compiled extension is cu12.8. vllm 0.20 ships cu13
#     and triggers "CUDA driver version is insufficient" on A100/H100.
# Local pyproject.toml stays on torch 2.11 / vllm 0.20 / cu13 (desktop has
# a newer driver). docker/pyproject.cu128.toml is the docker-only manifest.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MAX_JOBS=8 \
    HF_HOME=/workspace/hf_cache \
    WANDB_DIR=/workspace/wandb
# MAX_JOBS caps parallel nvcc jobs during the flash-attn compile inside `uv sync`.

# OS deps. python3.12 ships with Ubuntu 24.04; we install via apt to skip
# pyenv churn. git/openssh for repo sync, build-essential for the rare case
# a wheel needs to compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        git openssh-client curl ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
# python3.12-dev gives us Python.h, which triton needs to JIT-compile its
# cuda_utils C shim at first vLLM model load.

# uv for fast venv + lock-aware install.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /opt/vivace

# Copy dependency manifests first so the heavy install layer survives source
# edits. `--no-install-project` builds the dep tree (torch, vLLM, etc.) without
# trying to install the local `vivace` package — that would require README.md
# and the source tree, which we don't have yet.
#
# We use docker/pyproject.cu128.toml + docker/uv.cu128.lock (renamed in the
# image to pyproject.toml + uv.lock) so the Docker build pins torch + CUDA
# libs to cu128 wheels (runs on cloud driver 545+). The local project's
# pyproject.toml stays on cu13 wheels for the desktop dev environment.
COPY docker/pyproject.cu128.toml ./pyproject.toml
COPY docker/uv.cu128.lock ./uv.lock

RUN uv venv --python python3.12 && \
    uv sync --no-dev --frozen --no-install-project

# Belt-and-braces: force-install cuDNN explicitly. uv sometimes prunes
# nvidia-cudnn-cu12 when cuda-toolkit is in the dependency tree, even when
# nvidia-cudnn-cu12 is locked as a direct dep — leaves torch unable to
# dlopen libcudnn.so.9 at runtime. This RUN guarantees it's there.
RUN uv pip install --python /opt/vivace/.venv/bin/python nvidia-cudnn-cu12==9.19.0.56

# Repo source (excludes via .dockerignore: runs/, .venv, wandb, etc.)
COPY . .

# Install the vivace package itself now that source + README are present.
# Deps are already cached above, so this is a fast pip-install-style step.
RUN uv sync --no-dev --frozen

# torchvision pulled by transformers is built against cu13 even when torch
# is cu12.8. Its CUDA-version sanity check refuses to load alongside our cu12.8
# torch, breaking transformers' image_utils import (and therefore vllm config
# load). vivace doesn't use vision — drop it. transformers gracefully skips
# the torchvision.io import when the package isn't installed.
RUN uv pip uninstall --python /opt/vivace/.venv/bin/python torchvision

# Register the bundled NVIDIA libs (libcudnn / libcublas / libnccl / etc.)
# with the dynamic loader. The cu12 torch wheels ship these under
# .venv/lib/.../nvidia/<lib>/lib, which dlopen doesn't search by default,
# so `import torch` fails with "libcudnn.so.9: cannot open shared object
# file" without this step.
RUN find /opt/vivace/.venv/lib/python*/site-packages/nvidia -maxdepth 2 -name "lib" -type d \
        > /etc/ld.so.conf.d/nvidia-venv.conf && \
    ldconfig

# /workspace is where Runpod's network volume mounts; checkpoint + wandb
# subdirs live there so they persist across pod restarts.
RUN mkdir -p /workspace/hf_cache /workspace/wandb /workspace/checkpoints

# Keep the container alive so the cluster SSH proxy (Runpod et al.) can exec
# into it. A bare `bash` would exit immediately without a TTY; `sleep infinity`
# is the standard idle-keepalive that costs no CPU. Training is launched
# interactively after SSH'ing in, not at container start.
WORKDIR /opt/vivace
# torch/lib must be on LD_LIBRARY_PATH so vllm._C can dlopen libtorch.so,
# libtorch_cuda.so, libc10*.so at runtime. torch's __init__.py loads these
# at Python level for its own use, but vllm's compiled extension goes through
# the OS loader which doesn't know about Python-level paths.
ENV PATH="/opt/vivace/.venv/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/vivace/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

CMD ["sleep", "infinity"]
