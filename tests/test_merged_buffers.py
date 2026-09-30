"""Merged-buffer weight sync keeps the frozen base bit-identical (CPU).

peft's merge_adapter()/unmerge_adapter() do `w += delta; w -= delta` in bf16; the
two roundings leave a growing subset of base weights one ULP off. The sync path
now fills separate buffers from an untouched base instead.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from vivace.utils.weight_sync import (
    allocate_merged_buffers,
    build_param_specs,
    fill_merged_buffers,
    lora_layers,
    sync_named_tensors,
)


class _Block(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, d, bias=False)   # not a LoRA target

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) + self.up_proj(x)


class _Model(nn.Module):           # a parent so names carry a dotted prefix, as in HF models
    def __init__(self):
        super().__init__()
        self.attn = _Block()

    def forward(self, x):
        return self.attn(x)


def _lora_model(seed=0):
    torch.manual_seed(seed)
    base = _Model().to(torch.bfloat16)
    pm = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0))
    return pm


def _randomize_adapter(pm, scale=0.02):
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.normal_(0, scale)


def test_base_is_bit_identical_after_many_syncs():
    pm = _lora_model()
    base_before = {n: p.detach().clone() for n, p in pm.named_parameters() if n.endswith("base_layer.weight")}
    merged = allocate_merged_buffers(pm)
    assert set(merged) == {n for n, _ in lora_layers(pm)} == {f"attn.{t}.weight" for t in ("q_proj", "k_proj", "v_proj", "o_proj")}
    for _ in range(200):
        _randomize_adapter(pm)                     # a new adapter state every step, like training
        fill_merged_buffers(pm, merged)
    for n, p in pm.named_parameters():
        if n.endswith("base_layer.weight"):
            assert torch.equal(p, base_before[n]), n


def test_merged_buffer_matches_fp32_reference_and_peft_merge_within_one_ulp():
    pm = _lora_model(); _randomize_adapter(pm)
    merged = allocate_merged_buffers(pm)
    fill_merged_buffers(pm, merged)
    for name, mod in lora_layers(pm):
        w = mod.base_layer.weight.float()
        a, b = mod.lora_A["default"].weight.float(), mod.lora_B["default"].weight.float()
        ref = (w + mod.scaling["default"] * (b @ a)).to(torch.bfloat16)
        assert torch.equal(merged[name], ref), name                       # exactly the fp32-then-round value
    # peft's in-place merge rounds twice; it must agree to within one bf16 ULP
    snapshot = {n: p.detach().clone() for n, p in pm.named_parameters() if n.endswith("base_layer.weight")}
    pm.merge_adapter()
    for name, mod in lora_layers(pm):
        diff = (mod.base_layer.weight.float() - merged[name].float()).abs()
        ulp = torch.finfo(torch.bfloat16).eps * merged[name].float().abs().clamp_min(1e-8)
        assert (diff <= ulp + 1e-9).all(), name
    pm.unmerge_adapter()
    # ... and the in-place round trip is exactly the drift the buffers avoid
    assert any(not torch.equal(p, snapshot[n]) for n, p in pm.named_parameters() if n.endswith("base_layer.weight"))


def test_sync_names_resolve_targets_to_buffers_and_the_rest_to_params():
    pm = _lora_model(); _randomize_adapter(pm)
    merged = allocate_merged_buffers(pm); fill_merged_buffers(pm, merged)
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    specs, fusion_map = build_param_specs(pm, filter_fn=lambda n, _p: any(n.endswith(f".{t}.weight") for t in targets), fuse=True)
    named = sync_named_tensors(pm, merged)
    assert fusion_map == {"attn.qkv_proj.weight": ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"]}
    assert {s.name for s in specs} == {"attn.qkv_proj.weight", "attn.o_proj.weight"}
    for comp in fusion_map["attn.qkv_proj.weight"] + ["attn.o_proj.weight"]:
        assert named[comp].data_ptr() == merged[comp].data_ptr(), comp     # targets ship from the buffers ...
    assert named["attn.up_proj.weight"].data_ptr() == pm.base_model.model.attn.up_proj.weight.data_ptr()  # ... the rest from the live param
