"""Smoke test for NCCL weight sync via StatelessProcessGroup + PyNcclCommunicator.

Isolates the Pattern A plumbing (StatelessProcessGroup + PyNcclCommunicator)
behind trainer.py's weight_sync_method=nccl. If this passes, any sync bug is in
higher-level code (spec building, name mapping, etc.).

What it does:
  1. Build a minimal vLLM worker on the second GPU
  2. Rendezvous trainer (rank 0) + worker (rank 1) via StatelessProcessGroup
  3. Wrap in PyNcclCommunicator on each side
  4. Broadcast a scalar tensor — worker echoes the received value back
  5. Assert the value matches

Usage:
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m tests.test_nccl_sync

Expected output on success:
    [trainer] sending: [42.0, 3.14]
    [worker] received: [42.0, 3.14]
    OK — NCCL comm works

If it hangs at "rendezvous at localhost:...": the worker isn't joining.
Likely the `collective_rpc` call errored and was swallowed by the thread
wrapper. The script sets VLLM_ALLOW_INSECURE_SERIALIZATION=1 up front
because vLLM's default pickling rejects user-defined callables.

If it errors on PyNcclCommunicator: vLLM version mismatch — check
`from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator`
imports cleanly with your installed vLLM.
"""

from __future__ import annotations

import argparse
import os

# vLLM 0.19 refuses to pickle arbitrary callables through collective_rpc
# without this env var. Must be set BEFORE importing vllm so it propagates
# to the EngineCore subprocess. Needed for Pattern A weight sync.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import socket
import sys
import threading

import torch
from vllm import LLM
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# -----------------------------------------------------------------------------
# Worker-side functions (run inside vLLM subprocess via collective_rpc)
# -----------------------------------------------------------------------------
# These are defined at module scope so collective_rpc can pickle them.
# `self` inside a collective_rpc target refers to the vLLM worker instance.


def _worker_init(self, master_addr: str, master_port: int, my_rank: int, world_size: int):
    """Establish worker-side NCCL comm. Stashed on model_runner for later calls."""
    import torch
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pg = StatelessProcessGroup.create(
        host=master_addr, port=master_port,
        rank=my_rank, world_size=world_size,
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    comm = PyNcclCommunicator(group=pg, device=device)
    # Stash on a well-known attribute so subsequent RPCs can find it.
    self.model_runner._smoke_test_comm = comm
    return f"worker ready on {device}"


def _worker_receive(self, numel: int, src_rank: int):
    """Broadcast-receive a float32 tensor, return as Python list for the caller."""
    import torch

    comm = self.model_runner._smoke_test_comm
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    buf = torch.zeros(numel, dtype=torch.float32, device=device)
    comm.broadcast(buf, src=src_rank)
    return buf.cpu().tolist()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                   help="HF model id (kept small — this is only for spawning a vLLM process)")
    p.add_argument("--trainer-gpu", type=int, default=0)
    p.add_argument("--rollout-gpu", type=int, default=1)
    args = p.parse_args()

    # Set trainer's current device BEFORE creating any CUDA tensors on it.
    torch.cuda.set_device(args.trainer_gpu)

    print(f"[trainer] spawning vLLM on GPU {args.rollout_gpu}...")
    # Mirror the VLLMRolloutWorker pattern: pin vLLM to its GPU via CUDA_VISIBLE_DEVICES.
    import os
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.rollout_gpu)
    llm = LLM(
        model=args.model, tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        disable_log_stats=True,
    )
    if old_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
    else:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    # ------- Rendezvous -------
    host, port = "localhost", _find_free_port()
    world_size = 2
    print(f"[trainer] rendezvous at {host}:{port}, world_size={world_size}")

    # Kick off worker init on a thread. collective_rpc blocks the caller
    # until the target function returns on the worker — we need the trainer
    # to be rendezvousing at the same time, hence the thread.
    worker_init_result = []
    worker_init_error = []

    def _trigger_worker_init():
        try:
            r = llm.collective_rpc(
                _worker_init, args=(host, port, 1, world_size),
            )
            worker_init_result.append(r)
        except Exception as e:
            worker_init_error.append(e)
            # Print immediately — don't wait for trainer's TCPStore timeout to surface us.
            print(f"[worker thread] ERROR: {type(e).__name__}: {e}", file=sys.stderr)

    t = threading.Thread(target=_trigger_worker_init, daemon=True)
    t.start()

    # Trainer-side init — concurrent with the worker's StatelessProcessGroup.create.
    # Short timeout (30s) so we fail fast if the worker never joined; the
    # worker's collective_rpc should be ready within a few seconds.
    try:
        trainer_pg = StatelessProcessGroup.create(
            host=host, port=port, rank=0, world_size=world_size,
            store_timeout=30,
        )
    except Exception as e:
        if worker_init_error:
            raise RuntimeError(
                f"rendezvous failed because worker errored: {worker_init_error[0]}"
            ) from worker_init_error[0]
        raise
    trainer_device = torch.device(f"cuda:{args.trainer_gpu}")
    trainer_comm = PyNcclCommunicator(group=trainer_pg, device=trainer_device)

    t.join(timeout=60)
    if worker_init_error:
        raise RuntimeError(f"worker init failed: {worker_init_error[0]}") from worker_init_error[0]
    if not worker_init_result:
        raise RuntimeError("worker init timed out after 60s")
    print(f"[trainer] worker init result: {worker_init_result[0]}")

    # ------- Scalar broadcast -------
    test_values = [42.0, 3.14, -8.125, 1e-6]
    test_tensor = torch.tensor(test_values, dtype=torch.float32, device=trainer_device)
    print(f"[trainer] sending: {test_values}")

    # Worker receives on a thread, while trainer broadcasts on the main thread
    recv_result = []
    recv_error = []

    def _trigger_recv():
        try:
            r = llm.collective_rpc(_worker_receive, args=(len(test_values), 0))
            recv_result.append(r)
        except Exception as e:
            recv_error.append(e)

    t = threading.Thread(target=_trigger_recv, daemon=True)
    t.start()

    trainer_comm.broadcast(test_tensor, src=0)
    torch.cuda.synchronize()

    t.join(timeout=30)
    if recv_error:
        raise RuntimeError(f"worker recv failed: {recv_error[0]}") from recv_error[0]
    if not recv_result:
        raise RuntimeError("worker recv timed out after 30s")

    received = recv_result[0]
    # collective_rpc returns a list with one result per worker; grab the driver's.
    received_values = received[0] if isinstance(received, list) else received
    print(f"[worker] received: {received_values}")

    # NCCL moves bytes exactly, so same-dtype comparison is bit-identical; allclose
    # only so swapped test values needn't mind float32 rounding. Corruption >> atol.
    received_tensor = torch.tensor(received_values, dtype=torch.float32)
    if not torch.allclose(received_tensor, test_tensor.cpu(), rtol=1e-5, atol=1e-6):
        print(f"FAIL — sent {test_values}, received {received_values}", file=sys.stderr)
        sys.exit(1)
    print("OK — NCCL comm works")


if __name__ == "__main__":
    main()
