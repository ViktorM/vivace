import pytest
import torch

from vivace.algos.policy_gradient import compute_kl, compute_loss
from vivace.algos.types import RLConfig, validate_rl_config


def test_cispo_keeps_high_ratio_positive_tokens_by_default():
    cfg = RLConfig(
        loss_type="cispo",
        clip_cispo_high=2.0,
        cispo_use_token_mask=False,
        cispo_normalization="token",
    )
    policy_logp = torch.tensor([[0.0]], requires_grad=True)
    old_logp = policy_logp.detach() - torch.log(torch.tensor([[10.0]]))
    adv = torch.tensor([1.0])
    mask = torch.ones_like(policy_logp)
    token_count = mask.sum(dim=1)

    loss = compute_loss(cfg, policy_logp, old_logp, adv, mask, token_count)
    loss.backward()

    assert policy_logp.grad is not None
    assert policy_logp.grad.item() == pytest.approx(-2.0)


def test_cispo_token_mask_opt_in_drops_outward_high_ratio_tokens():
    cfg = RLConfig(
        loss_type="cispo",
        clip_cispo_high=2.0,
        cispo_use_token_mask=True,
        cispo_normalization="token",
    )
    policy_logp = torch.tensor([[0.0]], requires_grad=True)
    old_logp = policy_logp.detach() - torch.log(torch.tensor([[10.0]]))
    adv = torch.tensor([1.0])
    mask = torch.ones_like(policy_logp)
    token_count = mask.sum(dim=1)

    loss = compute_loss(cfg, policy_logp, old_logp, adv, mask, token_count)
    loss.backward()

    assert policy_logp.grad is not None
    assert policy_logp.grad.item() == pytest.approx(0.0)


def test_cispo_hybrid_normalization_blends_token_and_sequence_losses():
    policy_logp = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-4.0, 0.0, 0.0, 0.0]])
    old_logp = policy_logp.clone()
    adv = torch.tensor([1.0, 1.0])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    token_count = mask.sum(dim=1)

    token_cfg = RLConfig(loss_type="cispo", cispo_normalization="token")
    sequence_cfg = RLConfig(loss_type="cispo", cispo_normalization="sequence")
    hybrid_cfg = RLConfig(loss_type="cispo", cispo_normalization="hybrid")

    token_loss = compute_loss(token_cfg, policy_logp, old_logp, adv, mask, token_count)
    sequence_loss = compute_loss(sequence_cfg, policy_logp, old_logp, adv, mask, token_count)
    hybrid_loss = compute_loss(hybrid_cfg, policy_logp, old_logp, adv, mask, token_count)

    expected = 0.5 * (token_loss.item() + sequence_loss.item())
    assert hybrid_loss.item() == pytest.approx(expected)
    assert token_loss.item() != pytest.approx(sequence_loss.item())


def test_kl_reduces_in_float32_from_bfloat16_inputs():
    policy_logp = torch.tensor([[-1.0, -2.0]], dtype=torch.bfloat16)
    ref_logp = torch.tensor([[-1.1, -1.9]], dtype=torch.bfloat16)
    mask = torch.ones((1, 2), dtype=torch.bfloat16)

    kl = compute_kl(policy_logp, ref_logp, mask)

    assert kl.dtype == torch.float32


def test_cispo_config_validation_rejects_inverted_clip_range():
    cfg = RLConfig(loss_type="cispo", clip_cispo_low=3.0, clip_cispo_high=2.0)

    with pytest.raises(ValueError, match="clip_cispo_low"):
        validate_rl_config(cfg)


def _random_microbatches(seed: int, token_counts=(6, 2), S: int = 6):
    """Two same-width micro-batches with unequal response-token counts."""
    g = torch.Generator().manual_seed(seed)
    mbs = []
    for n_tok in token_counts:
        policy = torch.randn(2, S, generator=g, requires_grad=True)
        old = policy.detach() + 0.1 * torch.randn(2, S, generator=g)
        mask = torch.zeros(2, S)
        mask[:, :n_tok] = 1.0
        adv = torch.randn(2, generator=g)
        mbs.append((policy, old, adv, mask))
    return mbs


@pytest.mark.parametrize("cfg", [
    RLConfig(loss_type="dapo"),
    RLConfig(loss_type="cispo", cispo_normalization="token"),
    RLConfig(loss_type="cispo", cispo_normalization="hybrid"),
])
def test_token_norm_grad_accum_matches_single_batch(cfg):
    """sum(loss_mb / n) with token_norm == loss of the concatenated batch.

    Without token_norm, per-micro-batch token-mean upweights short-response
    micro-batches — the bug class this pins (mean-of-means != global mean).
    """
    mbs = _random_microbatches(seed=0)
    total_tokens = sum(m[3].sum() for m in mbs)
    n = len(mbs)
    token_norm = total_tokens / n

    accum = sum(
        compute_loss(cfg, p, o, a, m, m.sum(dim=1).clamp(min=1.0), token_norm=token_norm)
        for p, o, a, m in mbs
    ) / n

    cat = [torch.cat([mbs[0][i], mbs[1][i]]) for i in range(4)]
    single = compute_loss(cfg, cat[0], cat[1], cat[2], cat[3],
                          cat[3].sum(dim=1).clamp(min=1.0))

    assert accum.item() == pytest.approx(single.item(), rel=1e-6)

    # Gradients must match too, not just the scalar.
    accum.backward()
    grads_accum = [m[0].grad.clone() for m in mbs]
    single_inputs = _random_microbatches(seed=0)
    cat_p = torch.cat([single_inputs[0][0], single_inputs[1][0]])
    # rebuild graph on concatenated leaf tensors
    cat_loss = compute_loss(
        cfg,
        cat_p,
        torch.cat([single_inputs[0][1], single_inputs[1][1]]),
        torch.cat([single_inputs[0][2], single_inputs[1][2]]),
        torch.cat([single_inputs[0][3], single_inputs[1][3]]),
        torch.cat([single_inputs[0][3], single_inputs[1][3]]).sum(dim=1).clamp(min=1.0),
    )
    cat_loss.backward()
    grad_cat = torch.cat(grads_accum)
    assert torch.allclose(grad_cat[:2], single_inputs[0][0].grad, atol=1e-7)
    assert torch.allclose(grad_cat[2:], single_inputs[1][0].grad, atol=1e-7)


def test_token_norm_default_none_keeps_local_normalization():
    """token_norm=None (whole batch in one piece) is the paper objective."""
    cfg = RLConfig(loss_type="dapo")
    mbs = _random_microbatches(seed=1, token_counts=(4,))
    p, o, a, m = mbs[0]
    with_none = compute_loss(cfg, p, o, a, m, m.sum(dim=1).clamp(min=1.0))
    explicit = compute_loss(cfg, p, o, a, m, m.sum(dim=1).clamp(min=1.0),
                            token_norm=m.sum())
    assert with_none.item() == pytest.approx(explicit.item(), rel=1e-7)


def test_build_response_mask_matches_compute_token_logprobs():
    """The trainer now builds the loss mask without a forward — it must equal
    the mask compute_token_logprobs derives from the same response_lengths."""
    from vivace.algos.policy_gradient import build_response_mask, compute_token_logprobs

    class _StubModel(torch.nn.Module):
        def __init__(self, vocab=23):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, vocab)

        def forward(self, ids, attention_mask=None, position_ids=None):
            class _Out: ...
            out = _Out()
            out.logits = self.emb(ids)
            return out

    torch.manual_seed(0)
    B, S, plen = 3, 10, 4
    full_ids = torch.randint(1, 23, (B, S))
    resp_lens = torch.tensor([6, 2, 4])  # last row right-padded

    _, mask_ref, _ = compute_token_logprobs(
        _StubModel(), full_ids, plen, pad_token_id=0, response_lengths=resp_lens)
    mask = build_response_mask(plen, S, resp_lens)
    assert torch.equal(mask, mask_ref)
    assert mask.sum(dim=1).tolist() == [6.0, 2.0, 4.0]


def test_rl_step_fills_old_logp_and_ratio_is_one_on_first_epoch():
    """old_logp is no longer precomputed: epoch 1 must set it from the policy
    forward (detached), making the epoch-1 ratio exactly 1 (clip_frac == 0)."""
    from vivace.algos.policy_gradient import rl_step

    class _StubModel(torch.nn.Module):
        def __init__(self, vocab=23):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, vocab)

        def forward(self, ids, attention_mask=None, position_ids=None):
            class _Out: ...
            out = _Out()
            out.logits = self.emb(ids)
            return out

    torch.manual_seed(1)
    model = _StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    cfg = RLConfig(loss_type="grpo", adv_type="grpo", group_size=2,
                   optim_epochs=2, kl_coef=0.0, adaptive_sampling=False)
    B, S, plen = 2, 8, 3
    mbs = []
    for _ in range(2):
        full_ids = torch.randint(1, 23, (B, S))
        resp_lens = torch.tensor([5, 3])
        from vivace.algos.policy_gradient import build_response_mask
        mask = build_response_mask(plen, S, resp_lens)
        mbs.append({
            "full_ids": full_ids, "plen": plen,
            "adv": torch.tensor([0.5, -0.5]),
            "old_logp": None, "ref_logp": None,   # filled by rl_step / kl skipped
            "mask": mask, "token_count": mask.sum(dim=1).clamp(min=1.0),
            "responses": ["<answer>1</answer>", "x"], "rewards": torch.tensor([1.0, 0.0]),
            "pad_token_id": 0, "response_lengths": resp_lens,
        })

    metrics, _ = rl_step(cfg, mbs, model, None, opt, None, step=0, kl_ema=0.0)

    assert all(mb["old_logp"] is not None for mb in mbs)
    assert not mbs[0]["old_logp"].requires_grad
    assert metrics["kl"] == 0.0          # ref skipped at kl_coef=0
    # epoch-1 ratio ≡ 1; only epoch 2 (after one SGD step) can clip — with
    # lr=1e-3 on this stub the ratio stays inside 1±0.2, so clip_frac == 0.
    assert metrics["clip_frac"] == 0.0


def test_build_param_specs_lora_filter_keeps_only_target_weights():
    """LoRA sync filter: fused qkv weight + o_proj weight survive; biases,
    embeddings, and the (untargeted) MLP group are dropped on both paths."""
    import torch.nn as nn
    from vivace.utils.weight_sync import build_param_specs

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=True)
            self.k_proj = nn.Linear(8, 4, bias=True)
            self.v_proj = nn.Linear(8, 4, bias=True)
            self.o_proj = nn.Linear(8, 8, bias=False)

    class _Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)
            self.up_proj = nn.Linear(8, 16, bias=False)
            self.down_proj = nn.Linear(16, 8, bias=False)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attn()
            self.mlp = _Mlp()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(11, 8)
            self.layers = nn.ModuleList([_Layer()])

    model = _Model()
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")

    def lora_filter(name, _p):
        return any(name.endswith(f".{t}.weight") for t in targets)

    specs, fusion_map = build_param_specs(model, filter_fn=lora_filter, fuse=True)
    names = sorted(s.name for s in specs)
    assert names == ["layers.0.self_attn.o_proj.weight",
                     "layers.0.self_attn.qkv_proj.weight"]
    assert "layers.0.self_attn.qkv_proj.weight" in fusion_map
    # unfiltered still ships everything (full-FT path unchanged)
    specs_all, _ = build_param_specs(model, fuse=True)
    assert len(specs_all) > 2


def test_math_correct_batch_matches_singles_and_preserves_order():
    from vivace.rewards import _math_correct, math_correct_batch
    gts = ["72", "\\frac{1}{2}", "", "\\sqrt{12}", "5"]
    preds = ["72.0", "0.5", "1", "2\\sqrt{3}", "6"]
    batch = math_correct_batch(gts, preds)
    singles = [_math_correct(g, p) for g, p in zip(gts, preds)]
    assert batch == singles == [True, True, False, True, False]


def test_compute_advantages_hand_values_bound_and_rloo_identity():
    from vivace.algos.policy_gradient import compute_advantages
    G = 8
    r = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0], [1] * 8], dtype=torch.float32)

    def adv(t):
        return compute_advantages(r, RLConfig(adv_type=t, group_size=G)).view(-1, G)

    rloo, drg, grpo = adv("rloo"), adv("dr_grpo"), adv("grpo")
    # group 0 (one correct of 8): dr_grpo = [7/8, -1/8 x7], rloo = [1, -1/7 x7]
    assert torch.allclose(drg[0], torch.tensor([7 / 8] + [-1 / 8] * 7))
    assert torch.allclose(rloo[0], torch.tensor([1.0] + [-1 / 7] * 7))
    assert torch.allclose(rloo, drg * G / (G - 1))                 # same estimator up to G/(G-1)
    assert (grpo.abs() <= (G - 1) ** 0.5 + 1e-6).all()             # population z-score bound
    assert abs(grpo[0].abs().max().item() - (G - 1) ** 0.5) < 2e-3  # hit by a one-of-G group
    assert (rloo[2] == 0).all() and (drg[2] == 0).all() and (grpo[2] == 0).all()
    # z-scoring lifts a 1e-3 jitter group to unit scale; centered advantages keep the raw spread
    jitter = torch.tensor([[0.001] + [0.0] * 7])
    assert compute_advantages(jitter, RLConfig(adv_type="grpo", group_size=G)).abs().max() > 1.0
    assert compute_advantages(jitter, RLConfig(adv_type="rloo", group_size=G)).abs().max() < 0.01


def test_validate_rl_config_rejects_group_size_1_and_unknown_switches():
    for bad in (RLConfig(group_size=1), RLConfig(loss_type="dappo"), RLConfig(adv_type="rlo")):
        with pytest.raises(ValueError):
            validate_rl_config(bad)


def test_grad_clip_zero_disables_clipping_instead_of_zeroing_grads():
    from vivace.algos.policy_gradient import build_response_mask, rl_step

    class _StubModel(torch.nn.Module):
        def __init__(self, vocab=23):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, vocab)

        def forward(self, ids, attention_mask=None, position_ids=None):
            class _Out: ...
            out = _Out()
            out.logits = self.emb(ids)
            return out

    torch.manual_seed(2)
    model = _StubModel()
    before = model.emb.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    cfg = RLConfig(loss_type="grpo", adv_type="grpo", group_size=2, optim_epochs=1,
                   kl_coef=0.0, adaptive_sampling=False, grad_clip=0.0)
    B, S, plen = 2, 8, 3
    full_ids = torch.randint(1, 23, (B, S))
    resp_lens = torch.tensor([5, 3])
    mask = build_response_mask(plen, S, resp_lens)
    mb = {"full_ids": full_ids, "plen": plen, "adv": torch.tensor([0.5, -0.5]),
          "old_logp": None, "ref_logp": None, "mask": mask,
          "token_count": mask.sum(dim=1).clamp(min=1.0),
          "responses": ["<answer>1</answer>", "x"], "rewards": torch.tensor([1.0, 0.0]),
          "pad_token_id": 0, "response_lengths": resp_lens}
    metrics, _ = rl_step(cfg, [mb], model, None, opt, None, step=0, kl_ema=0.0)
    assert not torch.equal(model.emb.weight.detach(), before)   # grad_clip=0 means "off", not "zero every grad"
    assert metrics["grad_norm"] > 0
