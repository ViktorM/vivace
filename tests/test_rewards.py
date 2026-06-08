"""Reward function tests."""

from vivace.rewards import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    math_reward_batch,
    math_reward_single,
    overlong_penalty_reward,
)


def test_overlong_penalty_disabled_by_default():
    """Default RewardConfig has overlong_penalty=0.0 — penalty is always 0."""
    assert overlong_penalty_reward([0, 500, 1024, 2048], 1024) == [0.0, 0.0, 0.0, 0.0]


def test_overlong_penalty_zero_in_safe_zone():
    """Tokens at or below max_new_tokens - buffer get no penalty."""
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=256)
    # safe zone for max=1024 is [0, 768]
    assert overlong_penalty_reward([0, 100, 500, 768], 1024, cfg) == [0.0, 0.0, 0.0, 0.0]


def test_overlong_penalty_linear_in_buffer():
    """Linear ramp from 0 at safe edge to -1 at max."""
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=256)
    # midpoint of buffer (n=896) → penalty = -0.5
    assert overlong_penalty_reward([896], 1024, cfg) == [-0.5]
    # quarter into buffer (n=832) → -0.25
    assert overlong_penalty_reward([832], 1024, cfg) == [-0.25]


def test_overlong_penalty_capped_at_max():
    """At or above max_new_tokens, penalty is -overlong_penalty (clipped at the floor)."""
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=256)
    assert overlong_penalty_reward([1024, 2000], 1024, cfg) == [-1.0, -1.0]


def test_overlong_penalty_scales_with_weight():
    """overlong_penalty acts as a multiplier on the [-1, 0] base shape."""
    cfg = RewardConfig(overlong_penalty=0.5, overlong_buffer_tokens=256)
    assert overlong_penalty_reward([1024], 1024, cfg) == [-0.5]
    assert overlong_penalty_reward([896], 1024, cfg) == [-0.25]


def test_math_reward_batch_backward_compatible():
    """Existing call sites without token kwargs continue to work."""
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=256)
    base = math_reward_batch([r"\boxed{42}"], ["42"], DEFAULT_REWARD_CONFIG)
    # Same result when overlong cfg is on but no token counts provided
    with_cfg_no_tokens = math_reward_batch([r"\boxed{42}"], ["42"], cfg)
    assert base == with_cfg_no_tokens


def test_math_reward_batch_applies_penalty_when_enabled():
    """When cfg.overlong_penalty>0 AND token counts given, penalty is subtracted."""
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=256)
    no_penalty = math_reward_batch(
        [r"\boxed{42}"], ["42"], cfg,
        response_token_counts=[500], max_new_tokens=1024,
    )[0]
    with_penalty = math_reward_batch(
        [r"\boxed{42}"], ["42"], cfg,
        response_token_counts=[1024], max_new_tokens=1024,
    )[0]
    # At capped length, total reward is base - 1.0
    assert abs((no_penalty - with_penalty) - 1.0) < 1e-6


def test_math_reward_single_threads_through_kwargs():
    """The single-response variant accepts and uses token kwargs."""

    class _Ex:
        answer = "42"

    # With default cfg, kwargs are inert (overlong_penalty=0).
    assert math_reward_single(r"\boxed{42}", _Ex()) == \
        math_reward_single(r"\boxed{42}", _Ex(), response_token_count=1024, max_new_tokens=1024)
