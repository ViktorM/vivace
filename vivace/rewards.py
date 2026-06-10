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
import os
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
# Validly-grouped thousands numeral: 1,200 / +12,345.67 — NOT 1,2 or 12,34.
_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def extract_answer(text: str) -> str:
    """Pull the contents of the last <answer>...</answer> block."""
    if "<answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def to_float(x) -> Optional[float]:
    """Best-effort float coercion. Returns None on NaN/inf/garbage.

    Strips commas from validly-grouped thousands numerals ("1,200" → 1200.0):
    models write large GSM8K answers that way (int_format_reward even rewards
    the format). Invalid groupings ("1,2") stay unparseable rather than
    silently collapsing to 12.
    """
    try:
        if isinstance(x, str):
            x = x.strip()
            if _THOUSANDS_RE.match(x):
                x = x.replace(",", "")
        val = float(x)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def answer_match(gt, pred, tol: float = 1e-3) -> bool:
    """Numeric equality with tolerance. False on either-side parse failure.

    Coerces through `to_float` so comma-grouped numerals compare equal —
    keeps the reward path, Env.is_correct, and maj@k voting consistent.
    """
    gt_f, pred_f = to_float(gt), to_float(pred)
    if gt_f is None or pred_f is None:
        return False
    return abs(gt_f - pred_f) < tol


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


# --- Parallel LaTeX verification ---------------------------------------------
# math_verify is sympy-backed (~50-500 ms per LaTeX comparison); its own
# parse/verify carry built-in 5 s signal timeouts, so a rare sympy hang is
# already bounded per call. Batch paths fan the LaTeX comparisons out to a
# small persistent process pool purely for parallelism; numeric pairs never
# leave the calling process.

_VERIFY_POOL = None


def _warm_worker():
    # Pre-import the sympy stack so a worker's first task doesn't pay for it.
    import math_verify  # noqa: F401


def _make_pool():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # fork: cheap workers that inherit the loaded modules. They run pure
    # sympy and never touch CUDA, so forking a CUDA-initialized parent is
    # fine (DataLoader-worker contract) — but fork from a process with many
    # live threads risks child deadlocks, hence warm_verify_pool() below.
    return ProcessPoolExecutor(
        max_workers=min(8, os.cpu_count() or 1),
        mp_context=mp.get_context("fork"),
        initializer=_warm_worker,
    )


def warm_verify_pool() -> None:
    """Create the verify pool eagerly. The trainer calls this at init, BEFORE
    vLLM / NCCL / wandb spawn their threads — forking from a (nearly)
    single-threaded process is the safe window."""
    global _VERIFY_POOL
    if _VERIFY_POOL is None:
        _VERIFY_POOL = _make_pool()


def math_correct_batch(gts: list[str], preds: list[str]) -> list[bool]:
    """Vectorized `_math_correct`: numeric pairs short-circuit inline, LaTeX
    pairs verify in parallel on the process pool (math_verify's built-in 5 s
    timeouts bound each comparison inside the worker)."""
    global _VERIFY_POOL
    out: list[bool] = [False] * len(gts)
    pool_idx: list[int] = []
    pool_pairs: list[tuple[str, str]] = []
    for i, (gt, pred) in enumerate(zip(gts, preds)):
        g, p = (gt or "").strip(), (pred or "").strip()
        if not g or not p:
            continue
        if _PURE_NUM_RE.match(g) and _PURE_NUM_RE.match(p):
            out[i] = _math_correct(g, p)
        else:
            pool_idx.append(i)
            pool_pairs.append((g, p))
    if pool_pairs:
        from concurrent.futures.process import BrokenProcessPool

        warm_verify_pool()
        pool_gts, pool_preds = zip(*pool_pairs)
        try:
            results = list(_VERIFY_POOL.map(_math_correct, pool_gts, pool_preds))
        except BrokenProcessPool:
            # A killed worker poisons the executor permanently. Rebuild once;
            # if that also breaks, degrade to inline serial — never let the
            # reward path take down a training step.
            _VERIFY_POOL = _make_pool()
            try:
                results = list(_VERIFY_POOL.map(_math_correct, pool_gts, pool_preds))
            except BrokenProcessPool:
                results = [_math_correct(g, p) for g, p in pool_pairs]
        for i, ok in zip(pool_idx, results):
            out[i] = bool(ok)
    return out


def math_correctness_reward(
    responses: list[str], answers: list[str], cfg: RewardConfig = DEFAULT_REWARD_CONFIG
) -> list[float]:
    """correct_bonus on math_verify equivalence; wrong_penalty otherwise.
    Batched through the verify pool — LaTeX comparisons run in parallel."""
    extracted = [extract_answer(r) for r in responses]
    flags = math_correct_batch(answers, extracted)
    return [cfg.correct_bonus if ok else cfg.wrong_penalty for ok in flags]


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
