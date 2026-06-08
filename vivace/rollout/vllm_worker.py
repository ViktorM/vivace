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


def _init_ipc_on_vllm_worker(worker_self, handles, specs):
    """Open trainer-side CUDA IPC handles inside the vLLM worker subprocess.

    Called once at init via `collective_rpc`. Caches the resulting aliased
    tensors on the model_runner so per-step `_apply_ipc_on_vllm_worker` can
    just do `vllm_param.copy_(aliased)` without re-opening handles.

    Module-scope to avoid cloudpickle capturing VLLMRolloutWorker (whose `llm`
    holds non-picklable internals).
    """
    from vivace.utils.ipc_sync import open_ipc_handles_to_aliased_tensors
    from vivace.utils.weight_sync import ParamSpec

    # vLLM RPC may downgrade ParamSpec dataclass to dict — coerce back.
    specs = [s if isinstance(s, ParamSpec) else ParamSpec(**s) for s in specs]

    mr = worker_self.model_runner
    mr._ipc_aliased = open_ipc_handles_to_aliased_tensors(handles)
    mr._ipc_specs = specs
    return True


def _apply_ipc_on_vllm_worker(worker_self):
    """Per-step: copy aliased trainer tensors into vLLM's params.

    Trainer must have already filled fused buffers + done cuda.synchronize()
    before invoking this RPC (see Trainer._sync_weights_ipc).
    """
    import torch
    from vivace.utils.ipc_sync import receiver_copy_loop

    mr = worker_self.model_runner
    aliased = getattr(mr, "_ipc_aliased", None)
    specs = getattr(mr, "_ipc_specs", None)
    assert aliased is not None and specs is not None, (
        "init_ipc_sync must be called before update_weights_via_ipc"
    )

    vllm_named = dict(mr.model.named_parameters())
    receiver_copy_loop(aliased, vllm_named, specs)
    # Same-device memcpy may launch async; ensure all copies committed before
    # the trainer's unmerge_adapter() runs (which mutates the source storage).
    torch.cuda.synchronize()
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
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
    ):
        """Build the `vllm.LLM` once. `enforce_eager=True` disables CUDA graphs —
        slower but sometimes needed during dev when graphs and hot weight updates
        conflict. Trainer and rollout must share the same dtype (bf16 default).
        """
        # vLLM spawns its own subprocess (EngineCore) that inherits
        # CUDA_VISIBLE_DEVICES. Set it before LLM() and restore after so the
        # trainer process still sees all GPUs.
        #
        # Composition with an outer CUDA_VISIBLE_DEVICES (set by torchrun launcher
        # or the user when multiplexing several runs on one node): the child
        # subprocess inherits whatever we put in os.environ and interprets it as
        # **absolute** physical device indices — the outer mapping does NOT carry
        # across the fork. So if outer = "4,5,6,7" and we naively set inner = "0",
        # the child sees physical GPU 0, not physical 4. Translate gpu_ids
        # (relative to the parent's visible set) into physical indices first.
        import os
        gpu_ids = gpu_ids or [0]
        tp_size = len(gpu_ids)

        old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if old_visible:
            outer_list = [int(x) for x in old_visible.split(",") if x.strip()]
            physical_ids = [outer_list[g] for g in gpu_ids]
        else:
            physical_ids = list(gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(p) for p in physical_ids)

        # Under torchrun, env vars like MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/
        # LOCAL_RANK get set on the trainer process. The vLLM EngineCore
        # subprocess inherits them and gets confused — it tries to bind/connect
        # to a TCPStore at MASTER_ADDR:MASTER_PORT for its own internal setup,
        # which has nothing to do with torchrun's rendezvous. Unset them while
        # we spawn LLM(...), restore after.
        _torchrun_env_keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE",
                              "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
                              "TORCHELASTIC_USE_AGENT_STORE",
                              "TORCHELASTIC_RUN_ID", "TORCHELASTIC_MAX_RESTARTS",
                              "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME")
        saved_torchrun_env = {k: os.environ.pop(k) for k in _torchrun_env_keys
                              if k in os.environ}

        # Silence per-step INFO chatter from EngineCore (sleep/wake_up/CuMemAllocator).
        # In colocated mode these fire every step and drown out training logs.
        # `setdefault` lets the user override with VLLM_LOGGING_LEVEL=INFO when debugging.
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        # Rust BPE tokenizer (~9x faster encode at gsm8k length, ~30x at MATH/AIME
        # context). Opt out with VLLM_USE_FASTOKENS=0 if a model isn't supported.
        os.environ.setdefault("VLLM_USE_FASTOKENS", "1")

        # logprobs_mode="processed_logprobs": return log_softmax(logits/T) AFTER
        # temperature + top_p/top_k. vLLM v1's default is "raw_logprobs" which
        # ignores temperature — that breaks (3) since our compute_token_logprobs
        # uses logits/T. With "processed_logprobs" the sampled-token logprob from
        # vLLM matches our HF recompute within bf16 floor.
        # max_model_len=None lets vLLM use the model's max_position_embeddings,
        # which on Qwen3.5 / long-context models is huge and balloons KV cache size.
        # Set it explicitly (prompt + max_new_tokens) to size KV cache appropriately.
        llm_kwargs = dict(
            model=model_name, tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization, dtype=dtype,
            enable_lora=enable_lora, max_lora_rank=max_lora_rank,
            enforce_eager=enforce_eager,
            logprobs_mode="processed_logprobs",
            disable_log_stats=True,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        # Qwen3.5 DeltaNet uses Mamba cache blocks (one per concurrent decode seq).
        # vLLM's default max_num_seqs=256 needs 256 blocks, which doesn't fit at
        # gpu_memory_utilization<=0.5. Cap to actual concurrency.
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        self.llm = LLM(**llm_kwargs)

        if old_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        # Restore torchrun env vars in the trainer process
        os.environ.update(saved_torchrun_env)

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

    def init_ipc_sync(self, handles, specs) -> None:
        """One-shot: open trainer's CUDA IPC handles in the vLLM worker subprocess.

        Persistent: handles are kept on the model_runner and re-used by every
        `update_weights_via_ipc` call. Trainer must keep the underlying storages
        alive (peft merge/unmerge is in-place; fused buffers are preallocated —
        both invariants met by the trainer-side caller).
        """
        self.llm.collective_rpc(_init_ipc_on_vllm_worker, args=(handles, specs))

    def update_weights_via_ipc(self) -> None:
        """Per-step: copy aliased trainer tensors into vLLM's params (same-GPU).

        Trainer must have filled fused buffers and done `torch.cuda.synchronize()`
        before this call so that the bytes vLLM reads are the post-merge values.
        """
        self.llm.collective_rpc(_apply_ipc_on_vllm_worker)

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
