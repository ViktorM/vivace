"""MATH training environment.

Loads competition-math training data via the `corpus` parameter:

    corpus="hendrycks"  → EleutherAI/hendrycks_math (7 subject configs, ~7.5K train)
    corpus="numina"     → not yet wired
    corpus="deepmath"   → not yet wired

The eval counterpart is `MATH500Env` (500-problem held-out subset of the MATH
test split). Training on MATH-500 leaks the eval set, so this env intentionally
does not expose the test split — pair it with MATH500Env for evaluation.

Ground-truth answers are extracted from the `\\boxed{...}` block at the end of
each problem's `solution` field — the standard answer-marker convention used
across MATH-derivative datasets.
"""

from __future__ import annotations

import re
from typing import Callable

from vivace.envs.base import Env, Example
from vivace.envs.math_prompt import make_prompt
from vivace.rewards import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    math_reward_batch,
    math_reward_single,
)


_HENDRYCKS_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _extract_boxed(text: str) -> str | None:
    """Pull the contents of the last \\boxed{...} in `text`.

    Brace-balanced — naive regex misses nested braces (\\boxed{\\frac{1}{2}}).
    Returns None if no \\boxed{ token is present.
    """
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def _level_to_float(level: str | None) -> float | None:
    """'Level 5' → 5.0; missing or malformed → None."""
    if not level:
        return None
    m = re.search(r"\d+", level)
    return float(m.group()) if m else None


class MATHEnv(Env):
    """Competition-math training corpus. Defaults to Hendrycks MATH train.

    `reward_overrides` lets a yaml override individual RewardConfig fields
    (e.g. `{"correct_bonus": 5.0, "strict_format_bonus": 0.05}`) without
    needing a new env class. Configure via `env_kwargs` in the trainer yaml.
    """

    name = "math"

    def __init__(
        self,
        corpus: str = "hendrycks",
        n_train: int | None = None,
        reward_overrides: dict | None = None,
    ):
        self.corpus = corpus
        self.n_train = n_train
        if reward_overrides:
            base = DEFAULT_REWARD_CONFIG.__dict__
            self._reward_cfg: RewardConfig = RewardConfig(**{**base, **reward_overrides})
        else:
            self._reward_cfg = DEFAULT_REWARD_CONFIG
        self._train: list[Example] | None = None

    def _load(self) -> None:
        if self._train is not None:
            return
        if self.corpus == "hendrycks":
            self._train = self._load_hendrycks()
        elif self.corpus in {"numina", "deepmath"}:
            raise NotImplementedError(
                f"corpus={self.corpus!r} not yet wired; only 'hendrycks' is implemented"
            )
        else:
            raise ValueError(f"unknown corpus: {self.corpus!r}")

        if self.n_train is not None:
            self._train = self._train[: self.n_train]

    def _load_hendrycks(self) -> list[Example]:
        from datasets import load_dataset
        rows: list[Example] = []
        for subj in _HENDRYCKS_SUBJECTS:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
            for r in ds:
                ans = _extract_boxed(r["solution"])
                if ans is None:
                    # Skip rare rows whose solution doesn't end in \boxed{}
                    # (parser would have nothing to verify against).
                    continue
                rows.append(Example(
                    problem=r["problem"],
                    answer=ans,
                    difficulty=_level_to_float(r.get("level")),
                    topic=r.get("type") or subj,
                    metadata={"subject": subj},
                ))
        return rows

    def load_split(self, split: str) -> list[Example]:
        if split == "eval":
            raise ValueError(
                "MATHEnv is train-only — use MATH500Env for evaluation "
                "(MATH-500 is the canonical held-out subset of the MATH test set)"
            )
        if split != "train":
            raise ValueError(f"unknown split: {split!r}")
        self._load()
        return self._train  # type: ignore[return-value]

    def format_prompt(self, example: Example) -> str:
        return make_prompt(example.problem)

    @property
    def reward_fn(self) -> Callable:
        cfg = self._reward_cfg
        if cfg is DEFAULT_REWARD_CONFIG:
            return math_reward_single   # fast path — no closure when no override
        def _scored(
            response: str, example,
            response_token_count: int | None = None,
            max_new_tokens: int | None = None,
        ) -> float:
            tokens = [response_token_count] if response_token_count is not None else None
            return math_reward_batch(
                [response], [example.answer], cfg,
                response_token_counts=tokens, max_new_tokens=max_new_tokens,
            )[0]
        return _scored
