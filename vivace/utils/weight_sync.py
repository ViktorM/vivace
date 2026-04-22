"""NCCL-based weight synchronization between trainer and vLLM worker.

Direct GPU→GPU broadcast that replaces disk-based LoRA adapter loading.
See docs/weight_sync_approaches.md for the library survey and design notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP / torch.compile / peft wrappers. vLLM's model has none of these,
    so trainer-side names must match after stripping.
    """
    base = model
    for _ in range(4):  # safety: max 4 layers of wrapping
        if hasattr(base, "_orig_mod"):            # torch.compile
            base = base._orig_mod
        elif hasattr(base, "module") and not isinstance(base, nn.ModuleList):
            base = base.module                    # DDP / FSDP
        elif hasattr(base, "base_model") and hasattr(base.base_model, "model"):
            base = base.base_model.model          # peft
        else:
            break
    return base


def strip_wrapper_prefixes(name: str) -> str:
    """Remove DDP / torch.compile / peft prefixes from a parameter name."""
    prefixes = ("_orig_mod.", "module.", "base_model.model.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                changed = True
    return name


class BufferPool:
    """Reusable CUDA buffers keyed by (shape, dtype). Reuse across broadcasts —
    allocating per-param per-step would be slower than the disk-sync baseline.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._pool: dict[tuple, torch.Tensor] = {}

    def get(self, shape: torch.Size | tuple, dtype: torch.dtype) -> torch.Tensor:
        key = (tuple(shape), dtype)
        buf = self._pool.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._pool[key] = buf
        return buf

    def clear(self) -> None:
        self._pool.clear()


@dataclass
class ParamSpec:
    """One parameter in the broadcast schedule. Sender + receiver iterate the same list."""
    name: str          # canonical (post-unwrap) name
    shape: tuple[int, ...]
    dtype: torch.dtype


def build_param_specs(model: nn.Module, filter_fn=None) -> list[ParamSpec]:
    """Build a sorted list of ParamSpec. `filter_fn(name, p) -> bool` to keep
    only a subset (e.g. LoRA). Sort is what guarantees sender/receiver order match.
    """
    base = unwrap_model(model)
    specs = []
    for name, p in base.named_parameters():
        if filter_fn is not None and not filter_fn(name, p):
            continue
        specs.append(ParamSpec(name=name, shape=tuple(p.shape), dtype=p.dtype))
    specs.sort(key=lambda s: s.name)
    return specs


def allocate_fused_buffers(model, specs, device):
    """Pre-allocate persistent buffers for QKV/gate_up fusion. Call once at startup.

    TODO: fusion logic (sender-side cat into these buffers) not wired yet;
    needed for full FT sync. LoRA-only sync doesn't use this.
    """
    fused = {}
    for spec in specs:
        if "qkv_proj" in spec.name or "gate_up_proj" in spec.name:
            fused[spec.name] = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    return fused


def is_lora_param(name: str, _p: nn.Parameter) -> bool:
    """Filter for build_param_specs: LoRA adapter params only."""
    return "lora_" in name


def sender_broadcast_loop(
    model: nn.Module,
    specs: list[ParamSpec],
    comm,
    src_rank: int = 0,
) -> None:
    """Trainer-side NCCL broadcast over `comm` (PyNcclCommunicator).

    Must run concurrently with the receiver's matching loop — `comm.broadcast`
    is a rendezvous. `specs` is the contract: sender + receiver iterate in the
    same order, or they deadlock / misroute tensors.
    """
    base = unwrap_model(model)
    named = dict(base.named_parameters())

    for spec in specs:
        tensor = named[spec.name]
        assert tensor.shape == spec.shape, f"{spec.name}: shape mismatch"
        assert tensor.dtype == spec.dtype, f"{spec.name}: dtype mismatch"
        comm.broadcast(tensor.data, src=src_rank)

    torch.cuda.synchronize()


def receiver_broadcast_loop(
    vllm_named_params: dict[str, torch.Tensor],
    specs: list[ParamSpec],
    comm,
    src_rank: int,
    buffer_pool: BufferPool,
) -> None:
    """vLLM worker side: receive over NCCL, copy into vLLM's model tensors.

    Runs inside the worker subprocess via `collective_rpc`. Buffers come from
    a pre-allocated pool — per-step allocation would regress below disk-sync.
    """
    for spec in specs:
        buf = buffer_pool.get(spec.shape, spec.dtype)
        comm.broadcast(buf, src=src_rank)
        canonical_name = strip_wrapper_prefixes(spec.name)
        target = vllm_named_params[canonical_name]
        assert target.shape == buf.shape, f"{canonical_name}: vllm={tuple(target.shape)} sender={tuple(buf.shape)}"
        assert target.dtype == buf.dtype, f"{canonical_name}: dtype mismatch vllm={target.dtype} sender={buf.dtype}"
        target.data.copy_(buf)


def verify_weights_match(
    trainer_model: nn.Module,
    vllm_worker,
    tokenizer,
    test_prompt: str = "The capital of France is",
    atol: float = 5e-2,
    topk: int = 10,
) -> dict:
    """Check that trainer and vLLM produce matching next-token distributions.

    Tokenizes on the trainer side and passes `prompt_token_ids` to vLLM to
    rule out tokenization differences — any remaining disagreement is about
    weights, not BOS/EOS handling.

    `ok` gates only on top-1 match + top-5 overlap ≥ 0.6 (rank-based signals,
    scale-invariant). `max_logprob_diff` is kept as a diagnostic — it's noisy
    (0.1-0.5 even with identical weights) because bf16 forward noise grows
    with distribution peakedness.

    Interpretation:
      top_1_match=False     → real sync bug (argmax shifted)
      top_5_agreement < 0.6 → partial sync or extreme numerical drift
    """
    device = next(trainer_model.parameters()).device

    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    prompt_token_ids = input_ids[0].tolist()

    trainer_model.eval()
    logits = trainer_model(input_ids).logits.float()   # (B, T, V)
    trainer_logits = logits[0, -1]                      # (V,)
    trainer_topk = torch.topk(trainer_logits, topk).indices.tolist()
    trainer_logp = torch.log_softmax(trainer_logits, dim=-1)
    trainer_model.train()

    # Pass prompt_token_ids so vLLM uses the same tokens as the trainer's forward pass.
    outputs, _ = vllm_worker.generate(
        prompt_token_ids=[prompt_token_ids],
        temperature=0.0, top_k=1, max_tokens=1, n=1,
        logprobs=topk,
    )
    vllm_logp_dict = outputs[0].outputs[0].logprobs[0]   # dict[int, Logprob]
    vllm_topk = sorted(vllm_logp_dict.keys(),
                       key=lambda t: vllm_logp_dict[t].logprob, reverse=True)[:topk]

    common = set(trainer_topk) & set(vllm_topk)
    top_k_agreement = len(common) / topk
    top_5_agreement = len(set(trainer_topk[:5]) & set(vllm_topk[:5])) / 5
    top_1_match = trainer_topk[0] == vllm_topk[0]

    diffs = [abs(trainer_logp[t].item() - vllm_logp_dict[t].logprob) for t in common]
    max_diff = max(diffs) if diffs else float("inf")

    ok = top_1_match and top_5_agreement >= 0.6

    return {
        "ok": ok,
        "top_1_match": top_1_match,
        "top_5_agreement": top_5_agreement,
        "top_k_agreement": top_k_agreement,     # diagnostic
        "max_logprob_diff": max_diff,
        "trainer_topk": trainer_topk,
        "vllm_topk": vllm_topk,
    }
