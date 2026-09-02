#!/usr/bin/env bash
# Runpod injects PUBLIC_KEY into the container env; start sshd with it so the
# direct IP:port SSH mapping works (our base image has no Runpod start script).
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd
fi

# SSH sessions get a fresh login env — Docker ENV doesn't reach them. Persist
# the vars sshd sessions need (PATH incl. venv, loader path, caches, wandb).
{
    echo "export PATH=\"$PATH\""
    echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH:-}\""
    echo "export HF_HOME=\"${HF_HOME:-/workspace/hf_cache}\""
    echo "export WANDB_DIR=\"${WANDB_DIR:-/workspace/wandb}\""
    [ -n "${WANDB_API_KEY:-}" ] && echo "export WANDB_API_KEY=\"$WANDB_API_KEY\""
    [ -n "${VLLM_USE_FASTOKENS:-}" ] && echo "export VLLM_USE_FASTOKENS=\"$VLLM_USE_FASTOKENS\""
} > /etc/profile.d/vivace-env.sh

exec "$@"
