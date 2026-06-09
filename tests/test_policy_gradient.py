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
