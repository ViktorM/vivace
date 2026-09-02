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

`Trainer.__init__` builds the model in this order:

1. Load base model (HF, bf16, `low_cpu_mem_usage=True`), `.to(device)`
2. `peft.get_peft_model(base, lora_cfg)` — LoRA wrap
3. `gradient_checkpointing_enable(use_reentrant=False)` if enabled
4. `DDP(model, device_ids=[self.device.index])` — DDP wrap

peft before DDP so DDP registers the LoRA params for gradient sync; on-device
before DDP because DDP reads device info at wrap time. `device_ids` is the
model's device, not `local_rank` (`trainer_gpus=[2,3]` under no outer mask).

For full FT (no LoRA): skip step 2.

## Reference model handling

RL uses a reference model for the KL penalty. Two strategies:

**With LoRA** (preferred): there is no separate ref model. `rollout_phase`
wraps the ref `compute_token_logprobs` call in
`with self._inner_model.disable_adapter():` to forward through the base
weights. Zero extra VRAM. Skipped entirely when `kl_coef == 0`.

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

Colocated mode time-shares the card: vLLM is built with
`enable_sleep_mode=True` (without it `sleep()` frees zero bytes), level-1
sleeps right after generate and wakes before sync. The trainer gets the whole
card during train; what binds is the awake phase, when trainer weights + LoRA
+ Adam state sit next to vLLM's `gpu_memory_utilization`.

Shipped values: 0.7 for 0.5B colo; 0.65 for 1.5B DDP-colo on 2×4090 (0.7
OOMs in `merge_adapter` once the desktop eats 2.4 GB); 0.5 for 3B DDP-colo
(5.8 GB of weights in both trainer and vLLM during sync); 0.8 for every
disaggregated yaml. Never set `expandable_segments` in colocated mode — it
breaks sleep mode's CUDA memory pools (PyTorch #147851). 1.5B MATH on 24 GB
also needs `gradient_checkpointing: true`.

## vLLM construction gotchas

- `enforce_eager` defaults to False (CUDA graphs on): level-1 sleep and
  in-place IPC/NCCL sync keep virtual addresses stable, so graphs survive
  both.
- Match dtypes exactly across trainer and vLLM (bf16 default). Mixed dtype
  makes in-place weight copy fail mid-broadcast.
- vLLM spawns an `EngineCore` subprocess that inherits `CUDA_VISIBLE_DEVICES`
  at spawn time; `torch.cuda.set_device()` does NOT propagate. The worker
  sets the env var (physical ids) around `LLM()` and pops torchrun's
  `RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*` for the same window — EngineCore
  otherwise tries torchrun's TCPStore.
- `enable_lora=True` only on the disk-sync path: NCCL/IPC ship merged base
  weights, and `enable_lora` re-parents vLLM's linears under
  `.base_layer.weight`, breaking the broadcast-spec name matching.
- `logprobs_mode="processed_logprobs"`: v1's default `raw_logprobs` ignores
  temperature; `compute_token_logprobs` uses `logits/T`.

## LoRA hot-swap via disk

Minimal sync path (`weight_sync_method: disk`), LoRA only:

```python
# trainer (rank 0, then barrier); adapter_path defaults to /dev/shm
self._inner_model.save_pretrained(adapter_path)
# worker: bump lora_int_id; pass the request on every generate()
self._lora_counter += 1
self._current_lora = LoRARequest(f"adapter-{n}", lora_int_id=n, lora_path=adapter_path)
```

Key gotcha: vLLM caches adapters by `lora_int_id`. You MUST increment on each
sync, or vLLM silently serves the stale adapter and training looks broken for
no reason.

Works always; slow (~300-500 ms/sync). Graduate to IPC / NCCL when perf
matters.

## KV cache sleep/wake in colocated mode

vLLM's KV cache is the biggest VRAM consumer during rollout. `rollout_phase`
sleeps vLLM right after `generate` — the reward / advantage / logprob
recompute that follows is trainer-side and needs the memory:

```python
gc.collect(); torch.cuda.empty_cache()   # stable baseline for vLLM's freed_bytes >= 0 check
self.llm.sleep(level=1)                  # weights → CPU, KV discarded; needs enable_sleep_mode=True
```

`wake_up()` runs after `train_phase` and BEFORE `sync_weights()`: a sync into
a sleeping engine writes to unmapped pages and crashes the worker.

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
- [slime](https://github.com/THUDM/slime) — THUDM / Z.ai (GLM post-training),
  clean trainer + SGLang separation over HTTP
- [TRL vLLM integration](https://huggingface.co/docs/trl/main/en/vllm_integration) — HuggingFace's client/server-mode pattern
- [vLLM RFC #31848: Native weight syncing APIs](https://github.com/vllm-project/vllm/issues/31848) — future stable API
