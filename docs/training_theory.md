# Training theory: the WHY behind vivace's choices

Minimal docstrings in the code state *what* a function does. This doc
collects *why* — the theory, gotchas, and references that used to live in
module docstrings. Read before making non-obvious changes to `Trainer`,
`VLLMRolloutWorker`, or `weight_sync`.

## Contents
1. [Wrap order: LoRA + DDP](#wrap-order-lora--ddp)
2. [Reference model handling](#reference-model-handling)
3. [Tokenizer gotchas](#tokenizer-gotchas)
4. [Colocated memory budget](#colocated-memory-budget)
5. [vLLM construction gotchas](#vllm-construction-gotchas)
6. [LoRA hot-swap via disk](#lora-hot-swap-via-disk)
7. [KV cache sleep/wake in colocated mode](#kv-cache-sleepwake-in-colocated-mode)
8. [torch.distributed primer](#torchdistributed-primer)
9. [DDP vs FSDP](#ddp-vs-fsdp)
10. [References](#references)

---

## Wrap order: LoRA + DDP

Setting up a LoRA + DDP model must happen in this order:

1. Load base model (HF, bf16, `low_cpu_mem_usage=True`)
2. `peft.get_peft_model(base, lora_cfg)` — LoRA wrap
3. `.to(device)`
4. `DDP(model, device_ids=[local_rank])` — DDP wrap

Other orders appear to work but silently break checkpointing or gradient
sync. The peft wrap must happen before DDP so DDP sees the LoRA modules as
part of the param list; `.to(device)` must happen before DDP because DDP
reads device info at wrap time.

For full FT (no LoRA): skip step 2.

## Reference model handling

RL uses a reference model for the KL penalty. Two strategies:

**With LoRA** (preferred): there is no separate ref model. Use
`with model.disable_adapter():` inside `compute_token_logprobs` to forward
through the base weights. Zero extra VRAM; the base is already loaded.

**Without LoRA**: need a frozen deepcopy.

```python
ref_model = copy.deepcopy(base_model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)
```

~2× model VRAM. Fine for ≤1.5B models; painful for 7B+.

## Tokenizer gotchas

- `padding_side="left"` — required for generation (right-padding shifts the
  generated tokens to weird positions).
- `pad_token = eos_token` — HF default has no pad token for some models;
  generation errors if unset.
- Set both BEFORE any `generate` call. Forgetting these is the most common
  generation bug.
- `model.config.use_cache = False` during training, or gradient checkpointing
  complains.

## Colocated memory budget

On 2×4090 (24 GB each) with Qwen2.5-0.5B + LoRA:

```
~ 2.0 GB   base model (bf16)
~ 0.5 GB   LoRA params + grads + optimizer state (Adam 4× params in fp32)
~ 4.0 GB   activation memory during forward + backward
~ 8.0 GB   vLLM KV cache (tunable via gpu_memory_utilization)
~ 9.0 GB   headroom (CUDA overhead, fragmentation)
```

Set `gpu_memory_utilization ≈ 0.35` for vLLM in colocated mode — trainer
gets the rest implicitly. In disaggregated mode, crank to 0.85 since the
rollout GPU has nothing else on it.

For larger models (Qwen2.5-1.5B+), colocated gets tight on 24 GB. Use
disaggregated mode (trainer on one GPU, vLLM on another).

## vLLM construction gotchas

- `enforce_eager=True` during dev. CUDA graphs and hot weight updates don't
  always cooperate; eager is slower but simpler. Drop once `update_weights`
  is solid.
- Match dtypes exactly across trainer and vLLM (bf16 default). Mixed dtype
  makes in-place weight copy fail mid-broadcast.
- Build `LLM` AFTER `init_distributed()` and AFTER `torch.cuda.set_device()`,
  otherwise vLLM grabs cuda:0 on every rank.
- `enable_lora=True` + `max_lora_rank` reserve a little extra memory at
  init (vLLM pre-allocates LoRA buffers). Set `max_lora_rank` to the
  largest rank you'll use across all training runs.
- vLLM spawns an `EngineCore` subprocess that inherits `CUDA_VISIBLE_DEVICES`
  at spawn time. `torch.cuda.set_device()` does NOT propagate to child
  processes. Pin vLLM with env var mutation before `LLM()`, restore after.

## LoRA hot-swap via disk

Minimal sync path when NCCL is overkill or not yet working:

```python
# trainer:
peft_model.save_pretrained(adapter_path)
# worker:
self.llm.add_lora(LoRARequest(name, lora_int_id, adapter_path))
```

Then pass that `LoRARequest` on every `generate` call. Key gotcha: vLLM
caches adapters by `lora_int_id`. You MUST increment on each sync, or vLLM
silently serves the stale adapter and training looks broken for no reason.

Works always; slow (~300-500 ms/sync). Graduate to in-place NCCL when perf
matters.

## KV cache sleep/wake in colocated mode

In colocated mode, vLLM's KV cache is the biggest VRAM consumer during
rollout. After rollout ends, free it so the trainer can use that memory
for activations + grads:

```python
self.llm.sleep()         # recent vLLM (≥0.6.x)
torch.cuda.empty_cache()  # actually return to allocator pool
```

Pair with `wake_up()` before the next rollout. Forget one and OOM on the
next step.

Older vLLM (<0.6): set `driver_worker.cache_engine = None` manually. Feature
detect with `hasattr`.

## torch.distributed primer

PyTorch's distributed package uses one of three backends:

- **NCCL** — NVIDIA's collective comms. GPU-to-GPU. Fast. Default for training.
- **GLOO** — CPU fallback. Works without CUDA. Slow.
- **MPI** — rarely used in PyTorch land.

Every process needs four env vars:

| Var | Meaning |
|---|---|
| `RANK` | global process ID (0..world_size-1) |
| `LOCAL_RANK` | GPU index on this node (0..gpus_per_node-1) |
| `WORLD_SIZE` | total processes across all nodes |
| `MASTER_ADDR` + `MASTER_PORT` | rendezvous TCP/file store |

`torchrun --nproc_per_node=N` sets all four. For SLURM, read from
`SLURM_PROCID`, `SLURM_LOCALID`, etc. and export as env vars.

`dist.init_process_group(backend="nccl")` opens NCCL communicators between
all ranks in the WORLD. After this call, `dist.broadcast`, `dist.all_reduce`,
`dist.barrier` start working. Each NCCL communicator costs ~hundreds of MB
of GPU memory per device — don't sprinkle subgroups for no reason.

**Weight sync uses a separate `StatelessProcessGroup`** (Pattern A), not the
default `torch.distributed` group. See `weight_sync_approaches.md` for why.

## DDP vs FSDP

**DDP (DistributedDataParallel)**: every rank keeps a full model copy.
During backward, autograd hooks all-reduce gradients across ranks. By the
time backward returns, all ranks have the same (averaged) gradients. Cost:
N NCCL all-reduces per backward pass (bucketed for efficiency).

Use DDP when: model + optimizer state fits on one GPU, you want max
throughput per step.

**FSDP (FullyShardedDataParallel)**: shards the model itself across ranks.
Each rank holds 1/N of parameters and gathers the full param only when
needed for forward/backward. Cost: extra all-gather per layer.

Use FSDP when: model + optimizer state exceeds one GPU (7B+ typically).

vivace targets DDP first (Qwen2.5 0.5B–1.5B fit easily) and graduates to
FSDP for 7B+.

### FSDP twist for weight sync

If you broadcast to vLLM from an FSDP trainer, the params are sharded. Must
gather first:

```python
with FSDP.summon_full_params(model, writeback=False):
    # named_parameters() now yields unsharded views
    sender_broadcast_loop(...)
```

Not needed for DDP (each rank has the full param).

## References

- [PyTorch distributed](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [vLLM docs](https://docs.vllm.ai/en/latest/)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray-actor RLHF,
  reference for weight sync and scheduler
- [verl](https://github.com/volcengine/verl) — HybridEngine, sophisticated
  resharding between training and generation layouts
- [slime](https://github.com/THUDM/slime) — ByteDance, clean
  trainer + SGLang separation over HTTP
- [TRL vLLM integration](https://huggingface.co/docs/trl/main/en/vllm_integration) — HuggingFace's client/server-mode pattern
- [vLLM RFC #31848: Native weight syncing APIs](https://github.com/vllm-project/vllm/issues/31848) — future stable API
