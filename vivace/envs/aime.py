"""AIME 2024 / 2025 / 2026 evaluation environments.

The American Invitational Mathematics Examination — 30 integer-answer
problems per year (0-999). Pair with MATHEnv or NuminaMathEnv for training;
training on AIME itself leaks the eval set.

  AIME2024Env  → Maxwell-Jia/AIME_2024              (30 problems, single split)
  AIME2025Env  → opencompass/AIME2025-I + -II       (15 + 15 = 30 problems)
  AIME2026Env  → MathArena/aime_2026                (30 problems, single split)

The integer-answer shape means `math_reward_single`'s numeric fast-path always
fires here — no sympy roundtrip per eval sample.
"""

from __future__ import annotations

from typing import Callable

from vivace.envs.base import Env, Example
from vivace.envs.math_prompt import make_prompt
from vivace.rewards import _math_correct, extract_answer, math_reward_single


class _AIMEBase(Env):
    """Shared scaffolding for AIME envs. Subclasses set `name` and `_load_rows`."""

    def __init__(self, n_eval: int | None = None):
        self.n_eval = n_eval
        self._eval: list[Example] | None = None

    def _load(self) -> None:
        if self._eval is not None:
            return
        rows = self._load_rows()
        self._eval = rows[: self.n_eval] if self.n_eval is not None else rows

    def _load_rows(self) -> list[Example]:
        raise NotImplementedError

    def load_split(self, split: str) -> list[Example]:
        if split == "train":
            raise ValueError(
                f"{self.name} is eval-only; train on MATHEnv (or another corpus) instead"
            )
        if split != "eval":
            raise ValueError(f"unknown split: {split!r} ({self.name} only has 'eval')")
        self._load()
        return self._eval  # type: ignore[return-value]

    def format_prompt(self, example: Example) -> str:
        return make_prompt(example.problem)

    @property
    def reward_fn(self) -> Callable:
        return math_reward_single

    def is_correct(self, response: str, example: Example) -> bool:
        # Integer answers hit _math_correct's numeric fast path; same verifier
        # as the reward so eval accuracy and training reward can't diverge.
        return _math_correct(example.answer, extract_answer(response))


class AIME2024Env(_AIMEBase):
    """AIME 2024 — 30 problems, one split."""

    name = "aime24"

    def _load_rows(self) -> list[Example]:
        from datasets import load_dataset
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [
            Example(
                problem=r["Problem"],
                answer=str(r["Answer"]).strip(),
                metadata={"id": r.get("ID")},
            )
            for r in ds
        ]


class AIME2025Env(_AIMEBase):
    """AIME 2025 — 30 problems split across configs AIME2025-I (15) and -II (15)."""

    name = "aime25"

    def _load_rows(self) -> list[Example]:
        from datasets import load_dataset
        rows: list[Example] = []
        for cfg in ("AIME2025-I", "AIME2025-II"):
            ds = load_dataset("opencompass/AIME2025", cfg, split="test")
            for r in ds:
                rows.append(Example(
                    problem=r["question"],
                    answer=str(r["answer"]).strip(),
                    metadata={"section": cfg},
                ))
        return rows


class AIME2026Env(_AIMEBase):
    """AIME 2026 — 30 problems, single split. Source: MathArena/aime_2026."""

    name = "aime26"

    def _load_rows(self) -> list[Example]:
        from datasets import load_dataset
        ds = load_dataset("MathArena/aime_2026", split="train")
        return [
            Example(
                problem=r["problem"],
                answer=str(r["answer"]).strip(),
                metadata={"problem_idx": r.get("problem_idx")},
            )
            for r in ds
        ]
