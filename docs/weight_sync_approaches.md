# Weight Sync Approaches Across RL Frameworks

A survey of how major RL-for-LLM frameworks handle the trainer ↔ rollout engine weight synchronization problem. Background for the choice in "What vivace landed on" below.

## The problem

After each optimizer step, the freshly-updated policy weights live in the trainer. The inference engine (vLLM, SGLang, etc.) is running stale weights. Before the next rollout phase you must get the new weights into the inference engine — ideally in < 1% of step time and without rebuilding the engine.

Three axes of design variation:

1. **Process architecture** — single-process, multi-process managed by torchrun, or Ray-orchestrated actors
2. **Transport** — NCCL direct GPU-to-GPU, CUDA IPC, HTTP serialization, or shared filesystem
3. **Timing model** — synchronous (trainer waits for sync every step), asynchronous (weights pushed to a version queue, rollout uses latest available), or fully decoupled (separate training/rollout pools)

---

## slime (ByteDance, Megatron-LM + SGLang)

**Architecture**: clean three-service separation. Training (Megatron), inference (SGLang), and a router sit as independent processes. Even with `--colocate` they stay as distinct processes talking over HTTP through a router.

**Transport**: HTTP to the router's `/update_weights_from_tensor` endpoint. Underneath, SGLang's implementation uses NCCL on the inference side to scatter across its TP workers.

**Timing**: synchronous per training step.

**Name mapping**: uses `mbridge` to translate Megatron's sharded layout → HuggingFace-style naming that SGLang expects.

**Pros**: fault isolation (a crash in inference doesn't take down the trainer), easy to place on different hardware.
**Cons**: HTTP serialization + deserialization is overhead vs. raw NCCL.

---

## verl (ByteDance, HybridEngine)

**Architecture**: Ray-orchestrated. A single `ActorRolloutRefWorker` class can host actor + rollout + reference roles in one class or any subset — so actor and rollout can literally share the same GPU and Python process. Ray actors for fault isolation and placement.

**Transport**: NCCL + CUDA IPC, orchestrated by a "HybridEngine" that does in-place tensor resharding between training layout (FSDP DP×TP) and generation layout (vLLM TP). Zero-copy when co-located.

**Timing**: synchronous. The HybridEngine "transforms single model in-place" via sharding managers — achieves near-zero memory overhead because the training-time and generation-time tensors are the same underlying storage, just reshaped.

**Name mapping**: custom resharding code that handles FSDP unshard → TP reshard → name translation for vLLM.

**Async variant**: `AsyncServer` mode pulls generation out into asynchronous per-conversation servers, supports multi-turn with tools without GPU idle. Reports typical sync under 300 ms.

**Pros**: most advanced resharding in the ecosystem, scales to huge models.
**Cons**: heavy abstraction. Hard to understand what's actually happening. Ray dependency.

---

## OpenRLHF (original Ray-based RLHF framework)

**Architecture**: Ray actors, one per role (Actor, Critic, Reward, Reference). Actor uses vLLM for generation; Critic/Reward/Reference use HF/DeepSpeed forward passes.

**Transport**: NCCL or CUDA IPC. Rank 0 of the trainer forms a process group with all ranks of the inference engine and broadcasts each named parameter.

**Timing**: synchronous per step.

**This is the pattern vLLM's docs describe as the reference**: "having rank 0 of the trainer form a process group with all ranks of the inference engine". If you read the vLLM RFC #31848, it's codifying the OpenRLHF/slime pattern into a first-class API.

---

## TRL (HuggingFace)

**Two modes:**

**Colocate mode** (default): vLLM runs inside the trainer process, shares GPU memory with the training model. `vllm_mode="colocate"` in `GRPOConfig`. Simple, uses same device. Risk of memory contention.

**Server mode**: vLLM runs as a separate process on dedicated GPUs. `trl vllm-serve --model ...` spins up an HTTP server. The trainer talks to it via `vllm_client.generate(...)` for rollouts and `vllm_client.update_named_param(name, param.data)` for weight sync.

**Transport**: server mode uses NCCL under the covers for the actual weight transfer (via vLLM's `collective_rpc`), with HTTP just for coordination. For FSDP, TRL has a `sync_fsdp_params_to_vllm` helper that traverses modules post-order, unshards via a context manager, and calls `update_named_param` per parameter.

**LoRA handling**: if PEFT is enabled, TRL merges LoRA into the base weights, calls `update_named_param`, then un-merges the adapter. This is **worse than necessary** — it transfers the full base model every step even though only the adapter changed. But it's simple and correct.

**Constraint**: trainer and vLLM server must be on different CUDA devices. Starting from TRL after v0.19.1, it raises an explicit error if they collide.

---

## Unsloth

**Architecture**: single-process, colocated. vLLM runs inside the training process via `fast_inference=True` flag.

**Transport**: **shared buffers**. This is the interesting innovation: Unsloth shares the *same* physical memory between the training model and the vLLM model. When you train, you modify those tensors in place, and vLLM sees the updated weights on the next forward pass with zero transfer. FP8 weights live in a single buffer referenced by both the trainer and vLLM.

**Timing**: implicit — no transfer needed because it's the same memory.

**Pros**: literally zero weight-sync overhead. Can't be beaten on a single node.
**Cons**: requires tight coupling between training dtype and inference dtype, and patches vLLM's loader to share buffers. Doesn't generalize to multi-node or tensor-parallel.

---

## vLLM Native Weight Syncing API (RFC #31848, 2026)

vLLM is codifying the pattern. New proposed API:

```python
# Trainer side: establish process group with inference workers
model_update_group = stateless_init_process_group(
    master_address, master_port,
    rank=0, world_size=1 + n_vllm_workers,
    device=torch.device("cuda:0"),
)

# Tell vLLM to prepare its side
handle = llm.init_weight_transfer.remote(
    WeightTransferInitRequest(init_info=NCCLInitInfo(
        master_address=master_address,
        master_port=master_port,
        rank_offset=1,           # vLLM workers start at rank 1
        world_size=1 + n_vllm_workers,
    ))
)
ray.get(handle)

# Per-step: send metadata, then broadcast
handle = llm.update_weights.remote(WeightUpdateRequest(
    update_info=NCCLUpdateInfo(
        names=[...], dtype_names=[...], shapes=[...],
    )
))
for name, p in train_model.named_parameters():
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
ray.get(handle)
```

The "ask then broadcast" pattern: trainer instructs workers what to expect, then broadcasts while they receive. Workers loop over names in the declared order and write into their local copies.

When this API lands in stable vLLM, it's what everyone will use. For now, most frameworks roll their own on top of `collective_rpc`.

---

## Concurrency model (why we need a background thread)

For our Pattern A implementation (trainer process + vLLM subprocess sharing a NCCL group), the sender and receiver of a broadcast **must be running at the same time** — NCCL collective ops are synchronous rendezvous points. If only one side has entered `dist.broadcast(...)`, it blocks until the other side arrives.

In our architecture:
- The trainer is the main Python process. After `optimizer.step()`, it calls `sync_weights()`.
- The vLLM EngineCore subprocess is where the receiver must run. It doesn't check for messages on its own — the only way to run code on it is via `self.llm.collective_rpc(fn, args)`, which is **synchronous**: `collective_rpc` blocks the caller until the function returns on the worker.

Naive sequential code:

```python
# trainer main thread:
self.rollout_worker.update_weights(specs)   # blocks in collective_rpc...
sender_broadcast_loop(...)                  # never reached
```

The worker enters `receiver_broadcast_loop` inside `collective_rpc`, calls `dist.broadcast(...)` — and blocks waiting for the trainer to send. The trainer is blocked in `collective_rpc` waiting for the worker to finish. **Deadlock.**

### The threading solution

Run the receiver trigger on a background thread so the main thread can concurrently run the sender:

```python
def _trigger_receiver():
    self.rollout_worker.update_weights(specs)   # blocks this thread in collective_rpc

receiver_thread = threading.Thread(target=_trigger_receiver, daemon=True)
receiver_thread.start()

# main thread runs the sender — rendezvouses with the worker per broadcast
sender_broadcast_loop(self.model, specs, group=..., src_rank=0)

receiver_thread.join()   # worker returned from collective_rpc by now
```

Timeline:

```
trainer main thread:           trainer receiver thread:              vLLM worker subprocess:
─────────────────────────      ─────────────────────────────        ────────────────────────────
thread.start() ─────────────▶  (thread begins)
                               collective_rpc(receiver_fn) ────▶    enters receiver_broadcast_loop
sender_broadcast_loop:                                              dist.broadcast #1 (blocks)
  dist.broadcast #1 ◀══════════════ NCCL rendezvous ══════════════▶ dist.broadcast #1 (completes)
  dist.broadcast #2 ◀══════════════ NCCL rendezvous ══════════════▶ dist.broadcast #2
  ...                                                               ...
  dist.broadcast #N ◀══════════════ NCCL rendezvous ══════════════▶ dist.broadcast #N
                                                                    returns from receiver_broadcast_loop
                               collective_rpc returns ◀─────────    (worker back in idle loop)
thread.join() ←────────────────(thread ends)
```

### Alternative patterns (not used here)

- **Async collective_rpc** — `LLM.collective_rpc` is sync-only through vLLM 0.28; the coroutine variant exists only on `AsyncLLM`. Fewer moving parts than threading, but only reachable if we switch engines (architecture.md, Path B).
- **Persistent listener** — start a thread inside the vLLM worker at init time that loops on a queue or pipe, triggering `receiver_broadcast_loop` when a sync is requested. This is what slime and some OpenRLHF variants do. Avoids the per-sync `collective_rpc` overhead (~ms) but is more invasive: you must inject a long-running thread into the worker, manage its lifecycle, and handle cleanup on shutdown.
- **Split sender across threads** — when the sender has a lot of CPU-side work per broadcast (e.g. building fused tensors via `torch.cat`), you can overlap that work with in-flight broadcasts. Marginal returns at this scale. Skip until profiling says otherwise.

### GIL note

Python's GIL does not hurt us here because `dist.broadcast` and `collective_rpc` both release the GIL while waiting on I/O / NCCL. The main thread can make real progress in `sender_broadcast_loop` while the receiver thread is parked in `collective_rpc`.

### Error handling trap

If the receiver thread crashes (e.g. vLLM raises an exception inside the RPC'd function), the main thread happily continues broadcasting into the void, then hangs forever on the N+1th broadcast waiting for a receiver that's no longer there. For robustness, capture the receiver thread's exception and re-raise on join:

```python
receiver_exc = []
def _trigger_receiver():
    try:
        self.rollout_worker.update_weights(specs)
    except Exception as e:
        receiver_exc.append(e)
...
receiver_thread.join()
if receiver_exc:
    raise receiver_exc[0]
```

Done for the init rendezvous in `Trainer.__init__`; `_sync_weights_nccl`'s per-step receiver thread still doesn't.

## How generation is sequenced (sync vs async)

Separate from how weights flow, there's **when** generation happens relative to training.

**Synchronous (most common, incl. vivace today)**: step t = rollout → train → sync. Trainer idle during rollout, rollout idle during train. Simple to reason about. Good enough for most workloads.

**Async request-level (verl AsyncServer, slime)**: each conversation runs as an independent inference request. Training proceeds on whatever rollouts are finished. Matters a lot for multi-turn RL with tools — otherwise the GPU is idle while waiting for a tool call.

**Fully decoupled (AReaL)**: separate training and rollout GPU pools running continuously. Training pool reads whatever rollouts landed since last step; rollout pool uses whatever weights were last pushed. Needs off-policy corrections because rollouts are always stale. Highest utilization, most complexity.

**Pipelined (Checkpoint-Engine, MoonshotAI)**: for very large models, stages the sync itself — `H2D → broadcast → reload` run in pipeline. Reports updating a 1T-parameter model in ~21s across thousands of GPUs.

---

## What vivace landed on

Three methods, picked by `weight_sync_method` in YAML / `--weight-sync-method`:

- **`nccl`** — disaggregated only (trainer and vLLM on separate GPUs); the code
  default. Direct GPU→GPU broadcast via vLLM `collective_rpc` + Pattern A
  (StatelessProcessGroup + PyNcclCommunicator). NCCL refuses two ranks on the
  same device, so init rejects it for colocated.
- **`ipc`** — what the colocated yamls set; init rejects it elsewhere. Trainer
  takes CUDA IPC handles for its fused/base buffers once at init; vLLM's worker
  opens them and copies via same-device memcpy each step. ~26% wall-clock faster
  than disk at 200 steps on Qwen2.5-0.5B + LoRA. See `vivace/utils/ipc_sync.py`.
- **`disk`** — fallback, LoRA only. Saves the adapter to `/dev/shm/vivace_sync_<tag>`
  by default; vLLM reloads via `update_lora`. Works in any topology, slowest.

Not used: Ray orchestration (verl/OpenRLHF), HTTP weight push (slime), separate
trainer/rollout pools. Single-process trainer with vLLM subprocess is enough
for 2×4090 and Runpod-scale clusters.

Async generation is a separate axis, not in scope. Revisit for multi-turn /
tool-using RL.

---

## Testing

Validating NCCL weight sync has two parts: a **low-level smoke test** (plumbing works) and a **high-level verify test** (weights actually matched after sync). Always run the smoke test first — it rules out infrastructure bugs before you debug higher-level logic.

### 1. Smoke test — NCCL plumbing

`tests/test_nccl_sync.py` builds a minimal vLLM worker on GPU 1, establishes a `StatelessProcessGroup` + `PyNcclCommunicator` between the trainer (rank 0) and the worker (rank 1), then broadcasts a scalar and checks the received value. No training, no model weights — just the comm channel.

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m tests.test_nccl_sync
```

Optional flags: `--trainer-gpu 0 --rollout-gpu 1 --model Qwen/Qwen2.5-0.5B`.

Expected output on success:
```
[trainer] spawning vLLM on GPU 1...
[trainer] rendezvous at localhost:NNNNN, world_size=2
[trainer] worker init result: ['worker ready on cuda:0']
[trainer] sending: [42.0, 3.14]
[worker] received: [42.0, 3.14]
OK — NCCL comm works
```

The worker prints its device as `cuda:0` because of `CUDA_VISIBLE_DEVICES` remapping in the vLLM subprocess — physically it's GPU 1.

#### Required environment

The script sets `VLLM_ALLOW_INSECURE_SERIALIZATION=1` automatically. **Required** on vLLM 0.19–0.28 (default off in `vllm/envs.py`): the default serializer refuses to pickle user-defined callables through `collective_rpc`. Without it, the worker thread errors silently and the trainer hangs at the TCP rendezvous.

`train.py::_maybe_enable_vllm_callable_rpc` sets it for `nccl`/`ipc` before `Trainer` builds the vLLM subprocess. Any other script shipping callables through `collective_rpc` must set it before vLLM spawns, or export it in the shell.

#### Failure modes

- **`Timed out ... 1/2 clients joined`** — the worker never reached its side of the rendezvous. Usually a pickling error on the worker side. Check the `[worker thread] ERROR: ...` line that the script prints. If missing, the RPC is blocked somewhere before error reporting — check `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is set.
- **`OSError: [Errno 9] Bad file descriptor`** during `create_tcp_store` — a symptom of the above; the trainer's socket was cleaned up after timeout.
- **`ImportError: ... PyNcclCommunicator`** — your vLLM version doesn't expose it at `vllm.distributed.device_communicators.pynccl`. Check the import path for your version or upgrade vLLM.
- **Values mismatch** (received tensor doesn't equal sent) — NCCL path is broken on your hardware. Extremely rare; usually indicates a driver or topology issue.

### 2. verify_weights_match — end-to-end check against a sync backend

Once the smoke test passes, validate the full sync with disk, then NCCL. `tests/test_weight_sync.py` (`--method disk|nccl`, disaggregated only; `ipc` has no harness) runs a 3-step protocol:

1. Fresh trainer + fresh vLLM — expect AGREE (same checkpoint loaded both sides).
2. Perturb trainer's trainable params with gaussian noise — expect DISAGREE.
3. Call `trainer.sync_weights()` — expect AGREE again.

Step 1 passing proves the comparator works on identical weights. Step 2 passing (i.e., detecting disagreement) proves the comparator has teeth. Step 3 passing proves the sync backend actually propagated weights.

**First validate against disk (known-working):**

```bash
.venv/bin/python -m tests.test_weight_sync \
    --config vivace/configs/experiments/dapo_gsm8k_1.5b_profiling.yaml \
    --method disk
```

If all three steps pass, the test harness is trustworthy.

- Step 1 fails → `verify_weights_match` has a bug, or HF↔vLLM baseline noise flipped the top-1 (the gate is top-1 match AND top-5 overlap ≥ 0.6; `atol` is unused). Try another `--test-prompt`.
- Step 2 fails → perturbation too small to detect. Increase `--perturb-scale`.
- Step 3 passes iff `max_logprob_diff ≤ max(3 × step 1's, 0.5)` — no top-1 gate, since peft's LoraLinear computes `base@x + B@(A@x)` and vLLM the merged `(base + B@A)@x`, equal in math, not in bf16. Failing is NOT necessarily a sync bug: HF and vLLM use different attention, rotary and norm kernels, and on a perturbed (ill-conditioned) model these amplify into large logprob gaps even with bit-identical weights. Try `--perturb-scale 0.001` to confirm. The default (0.01) keeps the post-sync model well-conditioned enough that implementation noise stays bounded.

### 3. NCCL end-to-end

Only after (1) and (2) pass with disk, switch to NCCL:

```bash
.venv/bin/python -m tests.test_weight_sync \
    --config vivace/configs/experiments/dapo_gsm8k_1.5b_profiling.yaml \
    --method nccl
```

Same expected output. If step 3 fails with NCCL but passed with disk, the bug is in the NCCL path — most likely a name-mapping gap (`canonical_named_parameters`, `strip_wrapper_prefixes`, or `FUSION_GROUPS`). Spec order can't diverge: the receiver iterates the same `specs` list the sender ships over RPC, and `receiver_broadcast_loop` asserts shape/dtype per tensor. The script's diagnostics print top-k agreement and max logprob diff for each step, which narrows it further:

- `agreement ≈ 0` → broadcast wrote to wrong tensors (name mismatch).
- `agreement ≈ 0.5` → partial sync (some params landed, others didn't).
- `agreement ≈ 0.8 + small diff` → likely just bf16 noise, may actually be passing.

## Sources

- [vLLM RFC #31848: Native Weight Syncing APIs](https://github.com/vllm-project/vllm/issues/31848)
- [vLLM RFC #11399: Flexible Weight Sync for vLLM Workers](https://github.com/vllm-project/vllm/issues/11399)
- [Anatomy of RL Frameworks — Hanif Leoputera](https://www.hanifleo.com/anatomy-of-rl-frameworks/)
- [TRL vLLM Integration Docs](https://huggingface.co/docs/trl/main/en/vllm_integration)
- [vLLM New Weight Syncing Example](https://docs.vllm.ai/en/latest/examples/offline_inference/new_weight_syncing/)
- [Accelerating RLHF with vLLM — vLLM Blog (OpenRLHF post)](https://blog.vllm.ai/2025/04/23/openrlhf-vllm.html)
- [Unsloth Vision RL post](https://www.unsloth.ai/blog/vision-rl)
- [verl HybridFlow Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
