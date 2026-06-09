"""Rule-based reward functions.

Flat file (not a folder) until 3+ envs need their own families. Then
this becomes `vivace/rewards/` with one file per family.

The format-style rewards (`strict_format`, `soft_format`, `xmlcount`)
are dataset-agnostic and reusable across math/code envs that share the
`<think>...</think><answer>...</answer>` convention. The
`correctness` and `int_format` rewards are GSM8K-flavored numeric
checks; copy them when adding a new math env that needs different
parsing.

`gsm8k_reward_batch` is the top-level entry point that sums all five
reward components for a batch of responses.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    """Per-component reward weights."""

    correct_bonus: float = 2.0
    wrong_penalty: float = 0.0
    int_bonus: float = 0.25
    strict_format_bonus: float = 0.5
    soft_format_bonus: float = 0.25
    xmlcount_max: float = 0.25
    # DAPO §3.4 overlong soft penalty. Disabled by default (overlong_penalty=0.0);
    # opt-in per recipe. buffer = the linear ramp zone before max_new_tokens.
    overlong_penalty: float = 0.0
    overlong_buffer_tokens: int = 256


DEFAULT_REWARD_CONFIG = RewardConfig()


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_answer(text: str) -> str:
    """Pull the contents of the last <answer>...</answer> block."""
    if "<answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def to_float(x) -> Optional[float]:
    """Best-effort float coercion. Returns None on NaN/inf/garbage."""
    try:
        val = float(x)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def answer_match(gt, pred, tol: float = 1e-3) -> bool:
    """Numeric equality with tolerance. Falsy on either-side parse failure."""
    if gt is None or pred is None:
        return False
    try:
        return abs(float(gt) - float(pred)) < tol
    except (ValueError, TypeError):
        return False


def correctness_reward(
    responses: list[str], answers: list[str], cfg: RewardConfig = DEFAULT_REWARD_CONFIG
) -> list[float]:
    """correct_bonus if extracted answer numerically matches GT, else wrong_penalty."""
    extracted = [extract_answer(r) for r in responses]
    return [
        cfg.correct_bonus if answer_match(a, e) else cfg.wrong_penalty
        for e, a in zip(extracted, answers)
    ]


def int_format_reward(
    responses: list[str], cfg: RewardConfig = DEFAULT_REWARD_CONFIG
) -> list[float]:
    """Bonus for answers that are pure integers (signed, comma-separator allowed)."""
    return [
        cfg.int_bonus
        if extract_answer(r).lstrip("-").replace(",", "").isdigit()
        else 0.0
        for r in responses
    ]


def strict_format_reward(
    responses: list[str], cfg: RewardConfig = DEFAULT_REWARD_CONFIG
) -> list[float]:
    """Bonus for the exact <think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>\\n shape."""
    pat = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    return [
        cfg.strict_format_bonus if re.match(pat, r, re.DOTALL) else 0.0 for r in responses
    ]


def soft_format_reward(
    responses: list[str], cfg: RewardConfig = DEFAULT_REWARD_CONFIG
) -> list[float]:
    """Bonus for any well-formed think+answer pair (whitespace tolerant)."""
    pat = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [
        cfg.soft_format_bonus if re.match(pat, r, re.DOTALL) else 0.0 for r in responses
    ]


def xmlcount_reward(
    responses: list[str], cfg: RewardConfig = DEFAULT_REWARD_CONFIG
) -> list[float]:
    """Per-tag partial credit for well-placed open/close tags. Penalises trailing junk."""
    per_tag = cfg.xmlcount_max / 4.0

    def _score(t: str) -> float:
        s = 0.0
        if t.count("<think>\n") == 1:
            s += per_tag
        if t.count("\n</think>\n") == 1:
            s += per_tag
        if t.count("\n<answer>\n") == 1:
            s += per_tag
            s -= len(t.split("\n</answer>\n")[-1]) * 0.001
        if t.count("\n</answer>") == 1:
            s += per_tag
            s -= (len(t.split("\n</answer>")[-1]) - 1) * 0.001
        return s

    return [_score(r) for r in responses]


def overlong_penalty_reward(
    response_token_counts: list[int],
    max_new_tokens: int,
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> list[float]:
    """DAPO §3.4 soft length penalty. Linear ramp from 0 in the safe zone to
    -cfg.overlong_penalty at max_new_tokens. Returns one penalty per response."""
    if cfg.overlong_penalty <= 0.0:
        return [0.0] * len(response_token_counts)
    safe_zone = max_new_tokens - cfg.overlong_buffer_tokens
    out = []
    for n in response_token_counts:
        if n <= safe_zone:
            r = 0.0
        elif n >= max_new_tokens:
            r = -1.0
        else:
            r = -(n - safe_zone) / cfg.overlong_buffer_tokens
        out.append(cfg.overlong_penalty * r)
    return out


def gsm8k_reward_batch(
    responses: list[str],
    answers: list[str],
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
    response_token_counts: list[int] | None = None,
    max_new_tokens: int | None = None,
    return_components: bool = False,
) -> list[float] | tuple[list[float], dict[str, list[float]]]:
    """Sum of all reward components. Top-level GSM8K reward.

    If `cfg.overlong_penalty > 0` AND token counts + max_new_tokens are provided,
    the DAPO §3.4 soft length penalty is added as an extra component.

    With `return_components=True`, also returns a {name: per_response_list} dict
    for wandb logging.
    """
    components: dict[str, list[float]] = {
        "correct":       correctness_reward(responses, answers, cfg),
        "int":           int_format_reward(responses, cfg),
        "format_strict": strict_format_reward(responses, cfg),
        "format_soft":   soft_format_reward(responses, cfg),
        "xmlcount":      xmlcount_reward(responses, cfg),
    }
    if cfg.overlong_penalty > 0.0 and response_token_counts is not None and max_new_tokens is not None:
        components["overlong"] = overlong_penalty_reward(response_token_counts, max_new_tokens, cfg)
    totals = [sum(c[i] for c in components.values()) for i in range(len(responses))]
    return (totals, components) if return_components else totals


def gsm8k_reward_single(
    response: str,
    example,
    response_token_count: int | None = None,
    max_new_tokens: int | None = None,
) -> float:
    """Single-example variant. Useful from `Env.reward_fn`.

    Takes the full Example so the reward function has access to the problem
    statement, metadata, etc. Currently only uses example.answer for
    correctness checking, but having the full Example available enables
    future rewards like:
      - bonus for attempting math operations with numbers from the problem
      - penalty for answers that ignore stated constraints
      - difficulty-weighted scoring via example.difficulty
    """
    tokens = [response_token_count] if response_token_count is not None else None
    return gsm8k_reward_batch(
        [response], [example.answer], DEFAULT_REWARD_CONFIG,
        response_token_counts=tokens, max_new_tokens=max_new_tokens,
    )[0]


# =============================================================================
# Math (Hendrycks MATH / MATH-500 / AIME) — LaTeX-aware verifier
# =============================================================================
# `math_verify` (HF) is sympy-backed: ~50–500 ms per LaTeX comparison. A pure
# numeric fast-path keeps AIME-style integer answers sub-millisecond and saves
# real wall-clock at eval time over 30–500 problems.

_PURE_NUM_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _math_correct(gt: str, pred: str, tol: float = 1e-6) -> bool:
    """LaTeX-aware equivalence with a numeric short-circuit.

    Numeric short-circuit: when both sides parse cleanly as decimals, compare
    with tolerance — covers AIME's integer answers and any GSM8K-style numeric
    response without paying for sympy. Falls through to math_verify for LaTeX
    expressions (fractions, radicals, intervals, etc.) where symbolic
    equivalence is the only honest check.
    """
    if not gt or not pred:
        return False
    gt_s, pred_s = gt.strip(), pred.strip()
    if _PURE_NUM_RE.match(gt_s) and _PURE_NUM_RE.match(pred_s):
        try:
            return abs(float(gt_s) - float(pred_s)) < tol
        except ValueError:
            pass
    try:
        from math_verify import parse, verify

        def _parse(s: str):
            # math_verify's extraction expects delimited math ($...$, \boxed{}),
            # but our inputs (dataset ground truths, <answer> contents) are BARE
            # LaTeX, where extraction misfires: \sqrt{2} -> [] and 2\sqrt{3} ->
            # [2] (leading-number grab). Wrap in $...$ first; fall back to a
            # bare parse for input that already carries its own delimiters.
            parsed = parse(f"${s}$")
            return parsed if parsed else parse(s)

        return bool(verify(_parse(gt_s), _parse(pred_s)))
    except Exception:
        # math_verify throws on adversarial LaTeX or hangs sympy on rare
        # inputs; treat any exception as "not equivalent" rather than
        # poisoning the reward signal.
        return False


def math_correctness_reward(
    responses: list[str], answers: list[str], cfg: RewardConfig = DEFAULT_REWARD_CONFIG
) -> list[float]:
    """correct_bonus on math_verify equivalence; wrong_penalty otherwise."""
    extracted = [extract_answer(r) for r in responses]
    return [
        cfg.correct_bonus if _math_correct(a, e) else cfg.wrong_penalty
        for a, e in zip(answers, extracted)
    ]


def math_reward_batch(
    responses: list[str],
    answers: list[str],
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
    response_token_counts: list[int] | None = None,
    max_new_tokens: int | None = None,
    return_components: bool = False,
) -> list[float] | tuple[list[float], dict[str, list[float]]]:
    """Sum of correctness + format components. Skips int_format (GSM8K-specific).

    If `cfg.overlong_penalty > 0` AND token counts + max_new_tokens are provided,
    the DAPO §3.4 soft length penalty is added as an extra component.

    With `return_components=True`, also returns a {name: per_response_list} dict
    for wandb logging.
    """
    components: dict[str, list[float]] = {
        "correct":       math_correctness_reward(responses, answers, cfg),
        "format_strict": strict_format_reward(responses, cfg),
        "format_soft":   soft_format_reward(responses, cfg),
        "xmlcount":      xmlcount_reward(responses, cfg),
    }
    if cfg.overlong_penalty > 0.0 and response_token_counts is not None and max_new_tokens is not None:
        components["overlong"] = overlong_penalty_reward(response_token_counts, max_new_tokens, cfg)
    totals = [sum(c[i] for c in components.values()) for i in range(len(responses))]
    return (totals, components) if return_components else totals


def math_reward_single(
    response: str,
    example,
    response_token_count: int | None = None,
    max_new_tokens: int | None = None,
) -> float:
    """Single-example variant for `Env.reward_fn`. Hendrycks MATH / MATH-500 / AIME."""
    tokens = [response_token_count] if response_token_count is not None else None
    return math_reward_batch(
        [response], [example.answer], DEFAULT_REWARD_CONFIG,
        response_token_counts=tokens, max_new_tokens=max_new_tokens,
    )[0]
