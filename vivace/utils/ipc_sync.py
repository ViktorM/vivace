"""CUDA IPC-based weight sync for same-GPU trainer + vLLM (colocated mode).

NCCL doesn't work between two ranks on the same GPU. Disk-tmpfs is fast (~150ms
saved per step at LoRA r=16 vs real disk) but still goes GPU → CPU → tmpfs → CPU
→ GPU. CUDA IPC lets vLLM's worker subprocess directly alias the trainer's GPU
buffers, so per-step sync is a single same-device memcpy.

Invariant: trainer-side tensor pointers must be stable across the run.
  - peft `merge_adapter()` / `unmerge_adapter()` mutate base.weight.data in-place
    via `+=` / `-=`, so the storage is the same object across steps.
  - Fused buffers (qkv, gate_up) are preallocated once via `allocate_fused_buffers`
    and reused; pointers stable.
  - Live LoRA matrices (A, B) and full-FT live params likewise have stable storage.

Therefore IPC handles are taken once at init and reused for every per-step copy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vivace.utils.weight_sync import (
    canonical_named_parameters,
    strip_wrapper_prefixes,
    ParamSpec,
)


def pack_ipc_handles(
    model: nn.Module,
    specs: list[ParamSpec],
    fusion_map: dict | None,
    fused_buffers: dict | None,
    receiver_device_ordinal: int = 0,
) -> list[dict]:
    """Trainer side: build a list of IPC-handle dicts aligned with `specs`.

    For fused specs (qkv_proj, gate_up_proj), the source tensor is the
    preallocated fused buffer (must be filled before each sync — see
    `fill_fused_buffers`). For non-fused, the source is the live param `.data`.

    `receiver_device_ordinal` is what the vLLM EngineCore subprocess sees its
    GPU as. Since we set `CUDA_VISIBLE_DEVICES=<single_gpu>` for the EngineCore,
    it always sees its assigned GPU as cuda:0 — so default 0. The trainer's
    own ordinal can be anything (cuda:N for N=local_rank under torchrun); we
    rewrite the IPC handle's device field so the receiver opens on the correct
    (relative) ordinal, not the sender's absolute one.

    Each dict carries enough info for the receiver to recreate a tensor that
    aliases this exact storage. Sent via vLLM's `collective_rpc` once at init.
    """
    named = dict(canonical_named_parameters(model))
    handles: list[dict] = []
    for spec in specs:
        if fusion_map and spec.name in fusion_map:
            assert fused_buffers is not None and spec.name in fused_buffers, (
                f"fused spec {spec.name!r} has no preallocated buffer"
            )
            tensor = fused_buffers[spec.name]
        else:
            assert spec.name in named, f"spec {spec.name!r} not in canonical named_parameters"
            tensor = named[spec.name].data

        storage = tensor.untyped_storage()
        # `_share_cuda_()` returns an 8-tuple of CUDA IPC primitives that's
        # picklable across processes:
        # (device, handle, size_bytes, offset, ref_counter_handle,
        #  ref_counter_offset, event_handle, event_sync_required)
        info = list(storage._share_cuda_())
        info[0] = receiver_device_ordinal   # rewrite to receiver's relative ordinal
        handles.append({
            "name": spec.name,
            "ipc_info": tuple(info),
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "stride": tuple(tensor.stride()),
            "tensor_offset": tensor.storage_offset(),
        })
    return handles


def open_ipc_handles_to_aliased_tensors(handles: list[dict]) -> dict[str, torch.Tensor]:
    """vLLM-worker side: open each IPC handle once.

    Returns a dict {spec.name: tensor} where each tensor aliases trainer's
    GPU memory — reads/writes are visible across the process boundary without
    any explicit copy.
    """
    aliased: dict[str, torch.Tensor] = {}
    cur_device = torch.cuda.current_device()
    for h in handles:
        # vLLM's RPC may downgrade tuple→list during serialization. Coerce.
        info = tuple(h["ipc_info"])
        storage = torch.UntypedStorage._new_shared_cuda(*info)
        # Build a typed tensor view on the storage with the original layout.
        t = torch.empty(0, dtype=h["dtype"], device=f"cuda:{cur_device}")
        t.set_(storage, h["tensor_offset"], h["shape"], h["stride"])
        aliased[h["name"]] = t
    return aliased


def fill_fused_buffers(
    model: nn.Module,
    specs: list[ParamSpec],
    fusion_map: dict | None,
    fused_buffers: dict | None,
) -> None:
    """Trainer side: cat fused-spec components into preallocated buffers.

    Same logic as sender_broadcast_loop's fused branch, just without the
    broadcast — vLLM reads via IPC alias instead.
    """
    if not fusion_map or not fused_buffers:
        return
    named = dict(canonical_named_parameters(model))
    for spec in specs:
        if spec.name in fusion_map:
            components = [named[n].data for n in fusion_map[spec.name]]
            torch.cat(components, dim=0, out=fused_buffers[spec.name])


def receiver_copy_loop(
    aliased_tensors: dict[str, torch.Tensor],
    vllm_named_params: dict[str, torch.Tensor],
    specs: list[ParamSpec],
) -> None:
    """vLLM-worker side: copy each aliased trainer tensor into vLLM's matching param.

    Same-device GPU memcpy per spec. No NCCL, no wire transfer, no allocator
    pressure beyond what `target.copy_(src)` does (zero — both are existing
    allocations).
    """
    for spec in specs:
        canonical = strip_wrapper_prefixes(spec.name)
        target = vllm_named_params[canonical]
        src = aliased_tensors[spec.name]
        assert target.shape == src.shape, (
            f"{canonical}: vllm shape {tuple(target.shape)} != src {tuple(src.shape)}"
        )
        assert target.dtype == src.dtype, (
            f"{canonical}: vllm dtype {target.dtype} != src {src.dtype}"
        )
        target.data.copy_(src)
