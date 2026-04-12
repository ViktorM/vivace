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


def gsm8k_reward_batch(
    responses: list[str],
    answers: list[str],
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> list[float]:
    """Sum of all five reward components. Top-level GSM8K reward."""
    components = [
        correctness_reward(responses, answers, cfg),
        int_format_reward(responses, cfg),
        strict_format_reward(responses, cfg),
        soft_format_reward(responses, cfg),
        xmlcount_reward(responses, cfg),
    ]
    return [sum(c[i] for c in components) for i in range(len(responses))]


def gsm8k_reward_single(response: str, example) -> float:
    """Single-example variant. Useful from `Env.reward_fn`.

    Takes the full Example so the reward function has access to the problem
    statement, metadata, etc. Currently only uses example.answer for
    correctness checking, but having the full Example available enables
    future rewards like:
      - bonus for attempting math operations with numbers from the problem
      - penalty for answers that ignore stated constraints
      - difficulty-weighted scoring via example.difficulty
    """
    return gsm8k_reward_batch([response], [example.answer])[0]
