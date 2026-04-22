"""vLLM rollout worker: fast generation + weight sync.

Two modes:
  - DISAGGREGATED: worker runs on dedicated GPUs; trainer pushes weights over NCCL.
  - COLOCATED: worker and trainer share GPUs; sleep/wake_up frees KV cache between phases.
"""

from __future__ import annotations

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.utils import StatelessProcessGroup
from vivace.utils.weight_sync import receiver_broadcast_loop, BufferPool


def _init_nccl_on_vllm_worker(worker_self, master_addr, master_port, rank_offset, world_size):
    """Build Pattern A NCCL comm inside the vLLM worker subprocess.

    Invoked via `collective_rpc`. Module-scope (not nested) so cloudpickle
    doesn't capture the enclosing VLLMRolloutWorker, whose `llm` field holds
    a non-picklable `_queue.SimpleQueue`.
    """
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    import torch

    # TODO: for TP>1, derive the worker's tp_rank from vLLM's distributed env
    # instead of hard-coding 0.
    tp_rank = 0
    my_rank = rank_offset + tp_rank

    pg = StatelessProcessGroup.create(
        host=master_addr, port=master_port,
        rank=my_rank, world_size=world_size,
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    comm = PyNcclCommunicator(group=pg, device=device)

    worker_self.model_runner._weight_sync_comm = comm
    worker_self.model_runner._weight_sync_pg = pg
    return True


def _receive_nccl_on_vllm_worker(worker_self, specs):
    """Run the per-step NCCL receive loop inside the vLLM worker subprocess.

    Uses the comm stashed by `_init_nccl_on_vllm_worker`. Must run concurrently
    with the trainer's `sender_broadcast_loop` (they rendezvous per broadcast).
    """
    import torch
    from vivace.utils.weight_sync import receiver_broadcast_loop, BufferPool, ParamSpec

    mr = worker_self.model_runner
    comm = mr._weight_sync_comm
    assert comm is not None, "init_weight_sync must be called before update_weights"

    # vLLM's RPC serializer may downgrade our ParamSpec dataclass to a plain
    # dict when crossing the process boundary. Coerce back here.
    specs = [s if isinstance(s, ParamSpec) else ParamSpec(**s) for s in specs]

    # TODO: verify `mr.model` path if upgrading vLLM past 0.19.
    vllm_named = dict(mr.model.named_parameters())

    if not hasattr(mr, "_weight_sync_buffers"):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        mr._weight_sync_buffers = BufferPool(device)

    receiver_broadcast_loop(
        vllm_named_params=vllm_named,
        specs=specs,
        comm=comm,
        src_rank=0,
        buffer_pool=mr._weight_sync_buffers,
    )
    return True


class VLLMRolloutWorker:
    """vLLM-backed rollout worker. Owns one `vllm.LLM` instance for its lifetime."""

    def __init__(
        self,
        model_name: str,
        gpu_ids: list[int] | None = None,
        gpu_memory_utilization: float = 0.4,
        dtype: str = "bfloat16",
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        colocated: bool = True,
        enforce_eager: bool = False,
    ):
        """Build the `vllm.LLM` once. `enforce_eager=True` disables CUDA graphs —
        slower but sometimes needed during dev when graphs and hot weight updates
        conflict. Trainer and rollout must share the same dtype (bf16 default).
        """
        # vLLM spawns its own subprocess (EngineCore) that inherits
        # CUDA_VISIBLE_DEVICES. Set it before LLM() and restore after so the
        # trainer process still sees all GPUs.
        import os
        gpu_ids = gpu_ids or [0]
        tp_size = len(gpu_ids)

        old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        self.llm = LLM(model=model_name, tensor_parallel_size=tp_size,
                       gpu_memory_utilization=gpu_memory_utilization, dtype=dtype,
                       enable_lora=enable_lora, max_lora_rank=max_lora_rank,
                       enforce_eager=enforce_eager,
                       disable_log_stats=True)

        if old_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        self.colocated = colocated
        self.gpu_ids = gpu_ids
        self._lora_counter = 0
        self._current_lora = None
        # Populated by init_weight_sync(). None until then.
        self._weight_sync_group = None
        self._weight_sync_initialized = False

    def init_weight_sync(
        self,
        master_addr: str,
        master_port: int,
        trainer_rank: int = 0,
        worker_rank_offset: int = 1,
        world_size: int = 2,
    ) -> None:
        """Build the trainer↔worker NCCL comm (Pattern A). Run once at startup.

        The trainer side must be calling `StatelessProcessGroup.create` with the
        same host/port concurrently (from another thread) — otherwise the TCP
        rendezvous deadlocks. See `Trainer.__init__` for the threading.

        `worker_rank_offset` = first vLLM worker's rank in the shared world.
        For 1 trainer + TP=1 vLLM: offset=1, world_size=2.
        """
        if self._weight_sync_initialized:
            return
        self.llm.collective_rpc(
            _init_nccl_on_vllm_worker,
            args=(master_addr, master_port, worker_rank_offset, world_size),
        )
        self._weight_sync_initialized = True

    def generate(
        self,
        prompts: list[str] | None = None,
        prompt_token_ids: list[list[int]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        max_tokens: int = 1024,
        n: int = 8,
        logprobs: int | None = None,
        seed: int | None = None,
    ) -> tuple[list, list[list[str]]]:
        """Generate n completions per prompt. Returns (raw_outputs, response_texts).

        Args:
            prompts: text prompts (mutually exclusive with prompt_token_ids)
            prompt_token_ids: pre-tokenized prompts as list[list[int]]
                              (avoids re-tokenization boundary issues)
            temperature, top_p, top_k, max_tokens, n: standard sampling params
            logprobs: if set, return top-k log-probabilities per generated token.
                      Needed for verify_weights_match and off-policy corrections.
                      None = no logprobs returned (default, saves compute).
            seed: RNG seed for reproducibility. None = vLLM's default.

        Returns:
            raw_outputs: list of vLLM RequestOutput objects (for extracting
                         prompt_token_ids and completion token_ids)
            response_texts: nested list [B][G] of response strings
        """
        sp = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_tokens, n=n, logprobs=logprobs, seed=seed,
        )
        input_arg = prompt_token_ids if prompt_token_ids is not None else prompts
        outputs = self.llm.generate(
            input_arg, sp,
            lora_request=self._current_lora,
            use_tqdm=False,
        )
        texts = [[o.text for o in req.outputs] for req in outputs]
        return outputs, texts

    def update_weights(self, specs) -> None:
        """Per-step trainer→worker NCCL weight sync. Blocks until receive completes.

        Dispatches to `_receive_nccl_on_vllm_worker` in the vLLM subprocess,
        which runs `receiver_broadcast_loop` against the comm stashed by
        `init_weight_sync`. Must be called concurrently with the trainer's
        `sender_broadcast_loop` (see `Trainer._sync_weights_nccl`).
        """
        self.llm.collective_rpc(_receive_nccl_on_vllm_worker, args=(specs,))

    def update_lora(self, adapter_path: str) -> None:
        """Register a new LoRA adapter for subsequent `generate` calls.

        vLLM caches adapters by `lora_int_id`; we increment on every call so
        a freshly-saved adapter on disk actually gets re-loaded instead of
        serving the cached previous version.
        """
        self._lora_counter += 1
        self._current_lora = LoRARequest(
            lora_name=f"adapter-{self._lora_counter}",
            lora_int_id=self._lora_counter,
            lora_path=adapter_path,
        )

    def sleep(self) -> None:
        """Release vLLM's KV cache (colocated mode). Pair with wake_up()."""
        self.llm.sleep()
        torch.cuda.empty_cache()

    def wake_up(self) -> None:
        """Rebuild vLLM's KV cache. Inverse of sleep()."""
        self.llm.wake_up()
