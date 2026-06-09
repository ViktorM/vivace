"""MATH-500 evaluation environment.

500-problem held-out subset of the Hendrycks MATH test set
(https://huggingface.co/datasets/HuggingFaceH4/MATH-500). The canonical
LaTeX-answer math eval; pair with `MATHEnv` for training.

Eval-only: training on MATH-500 leaks the test set, so `load_split("train")`
raises with a pointer to MATHEnv.
"""

from __future__ import annotations

from typing import Callable

from vivace.envs.base import Env, Example
from vivace.envs.math_prompt import SYSTEM_PROMPT, make_prompt
from vivace.rewards import _math_correct, extract_answer, math_reward_single


class MATH500Env(Env):
    """HuggingFaceH4/MATH-500 — 500 LaTeX-answer competition-math problems."""

    name = "math500"

    def __init__(self, n_eval: int | None = None):
        self.n_eval = n_eval
        self._eval: list[Example] | None = None

    def _load(self) -> None:
        if self._eval is not None:
            return
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [
            Example(
                problem=row["problem"],
                answer=row["answer"],
                difficulty=float(row["level"]) if row.get("level") is not None else None,
                topic=row.get("subject"),
                metadata={"unique_id": row.get("unique_id")},
            )
            for row in ds
        ]
        self._eval = rows[: self.n_eval] if self.n_eval is not None else rows

    def load_split(self, split: str) -> list[Example]:
        if split == "train":
            raise ValueError(
                "MATH-500 is eval-only; train on MATHEnv (Hendrycks MATH train) instead"
            )
        if split != "eval":
            raise ValueError(f"unknown split: {split!r} (MATH-500 only has 'eval')")
        self._load()
        return self._eval  # type: ignore[return-value]

    def format_prompt(self, example: Example) -> str:
        return make_prompt(example.problem)

    @property
    def reward_fn(self) -> Callable:
        return math_reward_single

    def is_correct(self, response: str, example: Example) -> bool:
        # LaTeX ground truths — sympy-backed equivalence, same as the reward path.
        return _math_correct(example.answer, extract_answer(response))
