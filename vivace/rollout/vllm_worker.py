"""vLLM rollout worker with weight synchronization.

==============================================================================
TWO RESPONSIBILITIES, TWO MODES
==============================================================================

This module is the hardest file in the repo. It does two things:

  1. Run fast generation via vLLM given a list of prompts.
  2. Accept updated policy weights from the trainer and hot-swap them
     into the running vLLM engine WITHOUT rebuilding it.

Two execution modes are supported:

  - DISAGGREGATED: this worker runs in its own process(es) on dedicated GPUs.
    The trainer broadcasts updated weights over NCCL after each optimizer step.

  - COLOCATED: this worker and the trainer share GPUs. Between training and
    rollout phases the trainer frees activations / empties the CUDA cache,
    then we call vLLM for generation, then release vLLM's KV cache before
    resuming training. Slower but debuggable on 2x4090.

==============================================================================
SUGGESTED IMPLEMENTATION ORDER
==============================================================================

  1. __init__ + generate + sleep/wake_up   — get rollout-only working
  2. update_lora                            — disk-based hot-swap (simple)
  3. update_weights                         — in-place NCCL (hard)

This way you have a runnable rollout backend after step 1, and a runnable
LoRA training loop after step 2. Step 3 unlocks full fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1               # -1 = disabled. Set to e.g. 50 for top-k sampling.
    max_tokens: int = 1024
    n: int = 8                    # group size (completions per prompt)
    seed: int | None = None


class VLLMRolloutWorker:
    """vLLM-backed rollout worker.

    Constructor builds a `vllm.LLM` once. The methods below operate on
    that single instance. Pair sleep/wake_up calls in colocated mode to
    free KV cache between phases.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.4,
        dtype: str = "bfloat16",
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        colocated: bool = True,
    ):
        """Build the vllm.LLM.

        THEORY
        ------
        vLLM's `LLM` constructor is heavy: it allocates KV cache, builds
        CUDA graphs (unless enforce_eager), loads weights, and starts a
        scheduler. Do this ONCE per process, never inside the training
        loop.

        Memory budgeting in colocated mode is the trickiest part. On 2x4090
        with Qwen2.5-0.5B + LoRA, you have ~24GB per GPU. Rough budget:
            ~ 2 GB  base model bf16
            ~ 0.5 GB LoRA params + grads + optimizer state
            ~ 4 GB  activation memory during forward+backward
            ~ 8 GB  vLLM KV cache (configurable via gpu_memory_utilization)
            ~ 9 GB  headroom (cuda overhead, fragmentation, etc.)
        Set `gpu_memory_utilization=0.35` ish for vLLM in colocated; the
        trainer gets the rest implicitly. In disaggregated mode you can crank
        vLLM up to 0.85 because nothing else is on the rollout GPUs.

        GOTCHAS
        -------
        - `enforce_eager=True` is sometimes needed during dev: CUDA graphs
          and hot weight updates don't always cooperate. Eager is slower
          but simpler. Drop it once update_weights is solid.
        - `dtype="bfloat16"` everywhere — never mix dtypes between trainer
          and rollout, or the in-place weight copy will fail with a dtype
          mismatch.
        - Build vLLM AFTER init_distributed() and AFTER torch.cuda.set_device(),
          otherwise vLLM grabs cuda:0 on every rank.
        - `enable_lora=True` + `max_lora_rank` cost a little extra memory
          at init (vLLM pre-allocates LoRA buffers). Set max_lora_rank to
          the largest rank you'll use across all training runs.

        HINTS
        -----
        - from vllm import LLM, SamplingParams
        - self.llm = LLM(model=model_name, tensor_parallel_size=...,
                         gpu_memory_utilization=..., dtype=dtype,
                         enable_lora=enable_lora, max_lora_rank=max_lora_rank,
                         enforce_eager=True)  # remove enforce_eager once stable
        - self.colocated = colocated
        - self._lora_counter = 0   # used by update_lora to bump lora_int_id

        REFERENCES
        ----------
        - vLLM docs: https://docs.vllm.ai/en/latest/
        - OpenRLHF rollout worker: openrlhf/trainer/ppo_utils/experience_maker.py
        - verl: verl/workers/rollout/vllm_rollout/
        - slime (ByteDance): slime/backends/vllm/  - very clean reference
        """
        # TODO: implement.
        raise NotImplementedError

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[list[str]]:
        """Generate `sampling.n` completions per prompt. Returns [B][G] list of strings.

        THEORY
        ------
        vLLM's SamplingParams supports `n=group_size` natively — it does
        sibling sampling efficiently inside the engine (shared prefix KV
        cache, batched continuation). DON'T sample sequentially in a Python
        loop; that throws away vLLM's main perf advantage.

        GOTCHAS
        -------
        - The `RequestOutput` returned by `llm.generate` has shape
          [num_prompts] outer, each with [n] sibling completions inside
          `output.outputs`. Flatten to [B][G] list of strings.
        - Stop tokens vs `max_tokens`: setting both is fine — whichever
          fires first wins. For reasoning tasks, set `stop=["</answer>"]`
          to terminate as soon as the answer block closes.
        - Keep `prompt_token_ids` from the output if you need them later
          for log-prob recomputation alignment (left-padding gotcha
          carries over).

        HINTS
        -----
        - sp = SamplingParams(temperature=sampling.temperature, top_p=sampling.top_p,
                              top_k=sampling.top_k, max_tokens=sampling.max_tokens,
                              n=sampling.n, seed=sampling.seed)
        - outputs = self.llm.generate(prompts, sp)
        - return [[o.text for o in req.outputs] for req in outputs]
        """
        # TODO: implement.
        raise NotImplementedError

    def update_weights(self, named_tensors) -> None:
        """In-place parameter update over a NCCL subgroup.

        THEORY
        ------
        The trainer holds the freshly-stepped policy weights. The vLLM
        worker holds stale weights. We need to copy the trainer's weights
        into vLLM's underlying HF model, in-place, without rebuilding the
        engine. The hand-off is over a process group that spans BOTH
        trainer ranks AND vllm worker ranks (built once at startup by
        `vivace/utils/distributed.py::make_weight_sync_group`).

        Per-parameter dance:
            trainer rank 0: dist.broadcast(param.data, src=0, group=sync_group)
            vllm  rank N:   dist.broadcast(buffer,    src=0, group=sync_group)
                            target_param.data.copy_(buffer)

        FSDP twist
        ----------
        If the trainer is using FSDP, parameters are SHARDED across ranks.
        You must gather full parameters before broadcasting. Two options:
          - `with FSDP.summon_full_params(model, writeback=False):`
          - `with state_dict_type(FULL_STATE_DICT, offload_to_cpu=False):`
        Then iterate `named_parameters()` and broadcast the unsharded view.

        GOTCHAS
        -------
        - Parameter name mismatch: DDP-wrapped models have a `module.`
          prefix, vLLM's underlying model does not. Strip `module.` before
          name-matching.
        - The `param.data.copy_(received)` on the vLLM side IS safe even
          though vLLM wraps the HF model in its own scheduler — vLLM holds
          tensor REFERENCES, not copies, so an in-place update propagates.
        - Dtype must match exactly. bf16 trainer, bf16 vLLM, no fp32 in
          between (or you'll get a dtype mismatch error mid-broadcast).
        - On the receiving side, allocate the buffer once and reuse it
          for every parameter — don't allocate inside the loop.

        HINTS
        -----
        - Reach into vllm internals: vllm_model = (
              self.llm.llm_engine.model_executor.driver_worker.model_runner.model)
        - Build {name: tensor} dict for vllm_model.named_parameters() once.
        - For each (name, tensor) in named_tensors:
              vllm_param = vllm_named[strip_prefix(name)]
              dist.broadcast(buffer.copy_(tensor), src=0, group=sync_group)
              vllm_param.data.copy_(buffer)

        IMPLEMENT THIS LAST. update_lora gives you a working LoRA training
        loop with much less complexity.

        REFERENCES
        ----------
        - vLLM: vllm/worker/worker.py update_weights_from_external
        - OpenRLHF + verl + slime patterns mentioned in __init__ refs
        """
        # TODO: implement.
        raise NotImplementedError

    def update_lora(self, adapter_path: str) -> None:
        """Hot-swap a LoRA adapter from disk.

        THEORY
        ------
        For LoRA training you don't need to sync base model weights —
        they're frozen and never change. Only the LoRA A/B matrices update.
        The simplest sync path is:
            trainer:    peft_model.save_pretrained(adapter_path)
            worker:     self.llm.add_lora(LoRARequest(name, lora_int_id, adapter_path))
                        (then use that lora_int_id in the next generate call)

        Disk-based sync is slower than NCCL but MUCH simpler and ALWAYS
        works. Good enough for 2x4090 dev. Graduate to in-place NCCL when
        you outgrow it.

        GOTCHAS
        -------
        - vLLM caches loaded adapters by `lora_int_id`. You MUST increment
          this counter every time you load a new version, otherwise vLLM
          serves the OLD adapter and you wonder why training is broken.
        - Save adapter on rank 0 only, then `barrier()` so all vllm workers
          see a complete file before loading.
        - `LoRARequest` is the object you pass to `generate(..., lora_request=...)`.
          Cache the latest one as self._current_lora and pass it on every
          generate call.

        HINTS
        -----
        - from vllm.lora.request import LoRARequest
        - self._lora_counter += 1
        - self._current_lora = LoRARequest(
              lora_name=f"adapter-{self._lora_counter}",
              lora_int_id=self._lora_counter,
              lora_local_path=adapter_path,
          )
        - In generate(), pass lora_request=self._current_lora when set.

        IMPLEMENT THIS BEFORE update_weights — it gives you a working
        LoRA training loop with ~10 lines of code.
        """
        # TODO: implement.
        raise NotImplementedError

    def sleep(self) -> None:
        """Release KV cache for colocated mode.

        THEORY
        ------
        In colocated mode, vLLM's KV cache is the biggest single VRAM
        consumer during the rollout phase. After rollout, you free it so
        the trainer can use that memory for activations + grads.

        Recent vLLM (>=0.6.x): self.llm.sleep() exists directly.
        Older vLLM: self.llm.llm_engine.model_executor.driver_worker.cache_engine = None
                    + torch.cuda.empty_cache()

        Always pair sleep/wake_up. Easy to forget one and OOM on the next step.

        HINTS
        -----
        - Try the new API first; fall back to the old one with hasattr.
        - Always follow with torch.cuda.empty_cache() to actually return
          memory to the allocator pool.
        """
        # TODO: implement.
        raise NotImplementedError

    def wake_up(self) -> None:
        """Rebuild KV cache for colocated mode. Inverse of sleep()."""
        # TODO: implement.
        raise NotImplementedError
