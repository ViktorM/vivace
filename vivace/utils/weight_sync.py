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


# peft LoRA adapter names (A/B for linear, A/B for embedding, DoRA magnitude).
# When merge_adapter() has folded these into base_layer.weight, we don't broadcast
# them — they'd have no matching tensor on vLLM's enable_lora=False side.
_LORA_ADAPTER_TAGS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


def _is_lora_adapter_param(name: str) -> bool:
    return any(tag in name for tag in _LORA_ADAPTER_TAGS)


def canonical_named_parameters(model: nn.Module):
    """Yield (canonical_name, parameter) from a possibly-peft-wrapped model.

    peft applies LoRA by replacing target modules with LoraLinear wrappers, so
    `q_proj` becomes a wrapper holding `base_layer` (the original Linear) plus
    `lora_A`/`lora_B`. `named_parameters()` then yields names like
    `...q_proj.base_layer.weight` for the base, plus the adapter params.

    For NCCL sync after `merge_adapter()`:
      - Strip `.base_layer.` so the name matches vLLM's plain HF naming.
      - Drop adapter params — their effect is in base_layer.weight already.
    """
    base = unwrap_model(model)
    for name, p in base.named_parameters():
        if _is_lora_adapter_param(name):
            continue
        yield name.replace(".base_layer.", "."), p


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

# Fusion rules for trainer→vLLM name mapping. Suffix-based so they match
# any layer prefix (model.layers.{i}.self_attn.q_proj, etc.).
FUSION_GROUPS = [
    # (component_suffixes, fused_suffix)
    (("q_proj", "k_proj", "v_proj"), "qkv_proj"),
    (("gate_proj", "up_proj"),       "gate_up_proj"),
]


def build_param_specs(model: nn.Module, filter_fn=None, fuse=True) -> list[ParamSpec]:
    """Build a sorted list of ParamSpec. `filter_fn(name, p) -> bool` to keep
    only a subset (e.g. LoRA). Sort is what guarantees sender/receiver order match.

    Names are canonical HF names (peft `.base_layer.` segments stripped, LoRA
    adapter params skipped). For NCCL sync of LoRA models, callers must call
    `merge_adapter()` before broadcasting so that the canonical-named tensors
    contain the merged base+LoRA values.
    """
    named = dict(canonical_named_parameters(model))

    specs = []
    fusion_map = {}
    if fuse:
        # Fusion is applied to trainable tensors unconditionally — filter_fn
        # is deliberately NOT consulted here to keep groups all-or-nothing.
        # (If you later want filtered fusion, build a filter-aware version.)
        consumed = set()
        for comps, fused in FUSION_GROUPS:
            # Iterate both suffixes so qkv_proj.bias gets fused when present
            # (Qwen2.5 has bias on q/k/v). gate/up have bias=False in the same
            # family, so the `all-members-present` check below skips them.
            for suffix in ("weight", "bias"):
                for name in named:
                    if not name.endswith(f".{comps[0]}.{suffix}"):
                        continue

                    prefix = name[: -len(f".{comps[0]}.{suffix}")]
                    member_names = [f"{prefix}.{c}.{suffix}" for c in comps]

                    if not all(n in named for n in member_names):
                        continue  # partial group (e.g. no bias) — skip cleanly

                    fused_name = f"{prefix}.{fused}.{suffix}"
                    shapes = [named[n].shape for n in member_names]
                    # dim 0 is out-features for .weight ([h_out, h_in]) and the
                    # only dim for .bias ([h_out]); concat along dim 0 in both cases.
                    fused_shape = (sum(s[0] for s in shapes), *shapes[0][1:])
                    specs.append(ParamSpec(name=fused_name, shape=fused_shape,
                                            dtype=named[member_names[0]].dtype))
                    fusion_map[fused_name] = member_names
                    consumed.update(member_names)

        # Everything that wasn't consumed goes through as-is.
        for name, p in named.items():
            if filter_fn is not None and not filter_fn(name, p):
                continue

            if name not in consumed:
                specs.append(ParamSpec(name=name, shape=tuple(p.shape), dtype=p.dtype))
    else:
        for name, p in named.items():
            if filter_fn is not None and not filter_fn(name, p):
                continue
            specs.append(ParamSpec(name=name, shape=tuple(p.shape), dtype=p.dtype))
    specs.sort(key=lambda s: s.name)

    return specs, fusion_map


def allocate_fused_buffers(model, specs, device):
    """Pre-allocate persistent buffers for QKV/gate_up fusion. Call once at startup.
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
    fusion_map=None,
    fused_buffers=None
) -> None:
    """Trainer-side NCCL broadcast over `comm` (PyNcclCommunicator).

    Must run concurrently with the receiver's matching loop — `comm.broadcast`
    is a rendezvous. `specs` is the contract: sender + receiver iterate in the
    same order, or they deadlock / misroute tensors.
    """
    # Canonical names match what's in `specs` and `fusion_map` (peft `.base_layer.`
    # stripped, LoRA adapter params dropped). For LoRA + NCCL, the caller has
    # already merge_adapter()'d so the canonical-named base tensors hold the
    # merged base+LoRA values.
    named = dict(canonical_named_parameters(model))

    for spec in specs:
        if fusion_map and spec.name in fusion_map:
            # Fused spec: cat components into preallocated buffer.
            # Use .data views — trainer params have requires_grad=True, and
            # torch.cat(out=...) refuses autograd-tracked inputs.
            components = [named[n].data for n in fusion_map[spec.name]]
            torch.cat(components, dim=0, out=fused_buffers[spec.name])
            comm.broadcast(fused_buffers[spec.name], src=src_rank)
        else:
            comm.broadcast(named[spec.name].data, src=src_rank)

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
