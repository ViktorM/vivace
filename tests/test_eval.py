"""Eval-harness correctness: env-owned verification + canonical maj@k voting.

These pin the P0 fix from the 2026-06-09 review: eval accuracy / pass@k /
maj@k previously used float-only matching, scoring every non-numeric (LaTeX)
ground truth as wrong regardless of the response.
"""

from vivace.envs.base import Example
from vivace.envs.gsm8k import GSM8KEnv
from vivace.envs.math500 import MATH500Env
from vivace.eval.runner import maj_at_k, pass_at_k


def _resp(answer: str) -> str:
    return f"<think>\nreasoning\n</think>\n<answer>\n{answer}\n</answer>"


def _ex(answer: str) -> Example:
    return Example(problem="p", answer=answer)


# --- env.is_correct -----------------------------------------------------------

def test_math_is_correct_exact_latex():
    env = MATH500Env()
    assert env.is_correct(_resp("\\frac{14}{3}"), _ex("\\frac{14}{3}"))
    assert not env.is_correct(_resp("\\frac{14}{5}"), _ex("\\frac{14}{3}"))


def test_math_is_correct_latex_equivalence():
    env = MATH500Env()
    # The exact failure mode of the float-only matcher: equivalent but
    # non-identical forms of a non-float ground truth.
    assert env.is_correct(_resp("0.5"), _ex("\\frac{1}{2}"))
    assert env.is_correct(_resp("14/3"), _ex("\\frac{14}{3}"))
    assert env.is_correct(_resp("2\\sqrt{3}"), _ex("\\sqrt{12}"))


def test_math_is_correct_no_answer_tag():
    env = MATH500Env()
    assert not env.is_correct("free-form text, no tags", _ex("\\frac{1}{2}"))


def test_gsm8k_is_correct_numeric_tolerance():
    env = GSM8KEnv()
    assert env.is_correct(_resp("72"), _ex("72"))
    assert env.is_correct(_resp("72.0"), _ex("72"))
    assert not env.is_correct(_resp("71"), _ex("72"))
    assert not env.is_correct("no tags", _ex("72"))


# --- pass@k / maj@k -----------------------------------------------------------

def test_pass_at_k_uses_env_verifier():
    env = MATH500Env()
    samples = [_resp("1"), _resp("\\sqrt{2}")]
    assert pass_at_k([samples], [_ex("\\sqrt{2}")], env, k=2) == 1.0
    assert pass_at_k([samples], [_ex("\\sqrt{2}")], env, k=1) == 0.0


def test_maj_at_k_merges_numeric_formatting_variants():
    env = GSM8KEnv()
    # Three formatting variants of the correct answer vs two identical wrong
    # votes. Raw-string voting elects '13' (2 > 1+1+1); canonical voting pools
    # the correct variants (3 > 2).
    samples = [_resp("72.0"), _resp("72"), _resp("72.00"), _resp("13"), _resp("13")]
    assert maj_at_k([samples], [_ex("72")], env, k=5) == 1.0


def test_maj_at_k_winner_verified_with_env_verifier():
    env = MATH500Env()
    # Majority answer is LaTeX-equivalent to the ground truth — the winning
    # bucket must be scored by math_verify, not float comparison.
    samples = [_resp("\\frac{1}{2}"), _resp("\\frac{1}{2}"), _resp("3")]
    assert maj_at_k([samples], [_ex("0.5")], env, k=3) == 1.0


def test_maj_at_k_no_parseable_answers_scores_zero():
    env = GSM8KEnv()
    assert maj_at_k([["junk", "junk"]], [_ex("72")], env, k=2) == 0.0
