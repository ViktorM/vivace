"""Shared prompt for the LaTeX-answer math envs (MATHEnv, MATH500Env, AIME*Env).

Identical wording to GSM8K's prompt — keeps the comparison across math benchmarks
(GSM8K → MATH-500 → AIME) honest by holding the schema fixed.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""


def make_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\nProblem: {question}\n"
