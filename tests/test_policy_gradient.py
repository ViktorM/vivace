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
