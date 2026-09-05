"""Reward function tests."""

from vivace.rewards import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    gsm8k_reward_batch,
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


def test_gsm8k_reward_batch_components_sum_to_total():
    """return_components=True must give a breakdown whose sum equals the total."""
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=64)
    responses = [
        "<think>\nx\n</think>\n<answer>\n42\n</answer>\n",   # correct + format + int
        "guess 99",                                          # wrong, no format
    ]
    totals, comps = gsm8k_reward_batch(
        responses, ["42", "42"], cfg,
        response_token_counts=[20, 200], max_new_tokens=192,
        return_components=True,
    )
    assert set(comps) == {"correct", "int", "format_strict", "format_soft", "xmlcount", "overlong"}
    for i in range(2):
        assert abs(sum(comps[k][i] for k in comps) - totals[i]) < 1e-9


def test_math_reward_batch_components_sum_to_total():
    """Math variant has no `int` component; everything else mirrors gsm8k."""
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=64)
    responses = ["<think>\nx\n</think>\n<answer>\n42\n</answer>\n"]
    totals, comps = math_reward_batch(
        responses, ["42"], cfg,
        response_token_counts=[20], max_new_tokens=192,
        return_components=True,
    )
    assert "int" not in comps
    assert set(comps) == {"correct", "format_strict", "format_soft", "xmlcount", "overlong"}
    assert abs(sum(c[0] for c in comps.values()) - totals[0]) < 1e-9


def test_gsm8k_reward_batch_backward_compatible():
    """Default call (return_components=False) returns a flat list[float]."""
    out = gsm8k_reward_batch(["x"], ["1"])
    assert isinstance(out, list) and isinstance(out[0], float)


def test_to_float_strips_valid_thousands_grouping():
    from vivace.rewards import to_float
    assert to_float("1,200") == 1200.0
    assert to_float("+12,345.67") == 12345.67
    assert to_float("-1,000,000") == -1000000.0
    assert to_float(" 1,200 ") == 1200.0
    # Invalid groupings must NOT silently collapse ("1,2" is not 12).
    assert to_float("1,2") is None
    assert to_float("12,34") is None
    assert to_float("2, 3") is None
    # Non-string and plain paths unchanged.
    assert to_float(72) == 72.0
    assert to_float("72.5") == 72.5
    assert to_float("inf") is None
    assert to_float(None) is None


def test_answer_match_accepts_comma_formatted_prediction():
    from vivace.rewards import answer_match
    assert answer_match("1200", "1,200")
    assert answer_match("18000", "18,000.00")
    assert not answer_match("12", "1,2")


def test_gsm8k_correctness_rewards_comma_formatted_answer():
    """int_format_reward grants the int bonus for comma-grouped answers;
    the correctness component must agree they're correct."""
    _, comps = gsm8k_reward_batch(
        ["<think>\nx\n</think>\n<answer>\n1,200\n</answer>\n"], ["1200"],
        return_components=True,
    )
    assert comps["correct"][0] == DEFAULT_REWARD_CONFIG.correct_bonus
    assert comps["int"][0] == DEFAULT_REWARD_CONFIG.int_bonus


def test_overlong_penalty_rejects_buffer_wider_than_budget():
    import pytest
    from vivace.rewards import RewardConfig, overlong_penalty_reward
    cfg = RewardConfig(overlong_penalty=1.0, overlong_buffer_tokens=256)
    with pytest.raises(ValueError):
        overlong_penalty_reward([10], max_new_tokens=192, cfg=cfg)   # safe zone would be empty
    assert overlong_penalty_reward([100, 1024], max_new_tokens=1024, cfg=cfg) == [0.0, -1.0]


def test_xmlcount_zero_weight_disables_junk_penalty():
    from vivace.rewards import RewardConfig, xmlcount_reward
    resp = "<think>\nx\n</think>\n<answer>\n4\n</answer>\n" + "junk " * 100
    assert xmlcount_reward([resp])[0] < 0.25                                # default: trailing junk is penalised
    assert xmlcount_reward([resp], RewardConfig(xmlcount_max=0.0)) == [0.0]  # switched off: nothing, not a penalty


def test_strict_format_fires_without_trailing_newline():
    from vivace.rewards import strict_format_reward
    shaped = "<think>\nx\n</think>\n<answer>\n4\n</answer>"
    assert strict_format_reward([shaped, shaped + "\n"]) == [0.5, 0.5]        # EOS strips the final \n
    assert strict_format_reward(["<think>x</think><answer>4</answer>"]) == [0.0]


def test_soft_format_is_not_anchored_to_position_zero():
    from vivace.rewards import soft_format_reward
    assert soft_format_reward(["Sure.\n<think>x</think>\n<answer>4</answer>"]) == [0.25]
    assert soft_format_reward(["<think>x</think> no answer tag"]) == [0.0]


def test_xmlcount_junk_penalty_is_capped_and_needs_the_closing_tag():
    from vivace.rewards import xmlcount_reward
    perfect = "<think>\nx\n</think>\n<answer>\n4\n</answer>"
    assert xmlcount_reward([perfect]) == [0.25]                                 # no phantom -0.044
    junk = xmlcount_reward([perfect + "\n" + "junk " * 2000])[0]
    assert junk == 0.25 - 0.0625                                                # capped at one tag's credit
    capped = "<think>\nx\n</think>\n<answer>\n" + "reasoning " * 800             # generation cap hit, no </answer>
    assert xmlcount_reward([capped]) == [0.1875]                                # three tags, no length charge


def test_gsm8k_is_correct_boxed_first():
    from vivace.envs.base import Example
    from vivace.envs.gsm8k import GSM8KEnv
    env, ex = GSM8KEnv(), Example(problem="p", answer="42", metadata={})
    assert env.is_correct("<think>x</think>\n<answer>\n42\n</answer>", ex)                       # gsm8k style
    assert env.is_correct("<think>x</think>\n<answer>\nSo the total is \\boxed{42}.\n</answer>", ex)  # MATH style
    assert env.is_correct("<answer>\nThus \\boxed{\\$42} dollars.\n</answer>", ex)                  # LaTeX in the box
    assert not env.is_correct("<answer>\nFirst \\boxed{42}, finally \\boxed{41}.\n</answer>", ex)   # last box counts
    assert not env.is_correct("<answer>\nThe answer is 42 dollars.\n</answer>", ex)                 # prose still misses
    assert env.is_correct("<answer>\nSo \\boxed{\\frac{84}{2}}.\n</answer>", ex)                       # nested braces
    assert not env.is_correct("<answer>\n\\boxed{42} then \\boxed{\\frac{82}{2}}.\n</answer>", ex)     # nested last box wins
    assert env.is_correct("<answer>\n\\boxed{42}\n</answer>", Example(problem="p", answer=42.0, metadata={}))  # float GT (old dumps)
