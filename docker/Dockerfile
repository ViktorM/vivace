# vivace container image for modern cloud GPU pods (Runpod, Modal, etc.)
#
# Stack: torch 2.11 + vllm 0.20 + cu13. Requires NVIDIA driver >= 580.
# Targets recent H100 / H200 / B200 hosts. Older pods on driver 545-575
# will fail with "CUDA driver version is insufficient" — that is intentional;
# vivace v0.4.3+ only supports drivers shipping CUDA 13 runtime.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

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
        build-essential tmux \
    && rm -rf /var/lib/apt/lists/*
# python3.12-dev gives us Python.h, which triton needs to JIT-compile its
# cuda_utils C shim at first vLLM model load.

# uv for fast venv + lock-aware install.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /opt/vivace

# Copy the root pyproject + lock first so the heavy install layer caches
# across source edits. --no-install-project skips the local vivace package
# itself until the source is copied below.
COPY pyproject.toml uv.lock ./

RUN uv venv --python python3.12 && \
    uv sync --no-dev --frozen --no-install-project

# Belt-and-braces: force-install cuDNN explicitly. uv sometimes prunes
# nvidia-cudnn-cu13 when cuda-toolkit is in the dependency tree, even when
# it's locked as a direct dep — leaves torch unable to dlopen libcudnn.so.9
# at runtime. This RUN guarantees it's there.
RUN uv pip install --python /opt/vivace/.venv/bin/python nvidia-cudnn-cu13==9.19.0.56

# Repo source (excludes via .dockerignore: runs/, .venv, wandb, etc.)
COPY . .

# Install the vivace package itself now that source + README are present.
# Deps are already cached above, so this is a fast pip-install-style step.
RUN uv sync --no-dev --frozen

# Register the bundled NVIDIA libs (libcudnn / libcublas / libnccl / etc.)
# with the dynamic loader. The cu13 torch wheels ship these under
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
