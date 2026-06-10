"""Env base interface.

Every benchmark (GSM8K, MATH, DeepMath-103K, Omni-MATH, OpenEnv tasks, ...)
implements this. The contract is deliberately narrow:

    - sample_batch(n) -> list[Example]    # for training rollouts
    - eval_set()      -> list[Example]    # the held-out set for evaluation
    - format_prompt(example) -> str       # how to turn a problem into a model prompt
    - reward_fn       -> callable         # rule-based reward (response, gt) -> float

The Env does NOT do generation. That's the rollout worker's job. The Env only
provides problems, formats prompts, and knows how to score answers.

ToolRL note: for tool-use benchmarks, extend this with a `step()` method like
a gym env. Keep the base class minimal; add tool support as a subclass mixin
or a separate `ToolEnv` interface later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from vivace.rewards import answer_match, extract_answer, to_float


@dataclass
class Example:
    """One problem. Extend with extra fields as needed per benchmark."""
    problem: str
    answer: str                    # ground truth (verifier target)
    difficulty: float | None = None
    topic: str | None = None
    metadata: dict | None = None


class Env(ABC):
    name: str = "base"

    @abstractmethod
    def load_split(self, split: str) -> list[Example]:
        """Return all examples in a split. `split` is 'train' or 'eval'."""
        ...

    @abstractmethod
    def format_prompt(self, example: Example) -> str:
        """Format a problem into the prompt string fed to the policy."""
        ...

    @property
    @abstractmethod
    def reward_fn(self) -> Callable[[str, Example], float]:
        """Returns a function (response_text, example) -> float.

        The reward function receives the full Example (not just the answer string)
        so it has access to the problem statement, metadata, difficulty, etc.
        This enables rewards that depend on more than answer correctness — e.g.
        bonus for attempting relevant math operations, penalty for ignoring
        problem constraints, or code envs that need the problem to run tests.

        This is the rule-based reward. For math: symbolic answer extraction + verify.
        For code: run tests. For anything needing a model: not allowed here.
        """
        ...

    def is_correct(self, response: str, example: Example) -> bool:
        """Binary verifier for eval metrics (accuracy_pct, pass@k, maj@k).

        Default: numeric match on the extracted <answer> block. Envs whose
        ground truths are not plain numbers (LaTeX math) MUST override —
        float comparison scores every non-numeric answer as wrong.
        Must agree with the correctness component of `reward_fn`.
        """
        ans = extract_answer(response)
        return bool(ans) and answer_match(to_float(example.answer), to_float(ans))

    def is_correct_batch(self, responses: list[str], examples: list[Example]) -> list[bool]:
        """Vectorized `is_correct`. Override when batch scoring can fan out
        (the math family routes LaTeX pairs to the math_verify process pool)."""
        return [self.is_correct(r, ex) for r, ex in zip(responses, examples)]

    def sample_batch(self, n: int, rng) -> list[Example]:
        """Default: uniform sampling from the training split. Override for curriculum."""
        train = self.load_split("train")
        idx = rng.choice(len(train), size=n, replace=False)
        return [train[int(i)] for i in idx]

    def reward_breakdown(
        self,
        responses: list[str],
        examples: list[Example],
        response_token_counts: list[int] | None = None,
        max_new_tokens: int | None = None,
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Compute total rewards + per-component breakdown for a batch.

        Default impl: call `reward_fn` per response, return empty components dict.
        Subclasses that wrap a `*_reward_batch` (gsm8k, math, ...) override this
        to return the {component_name: per_response_list} dict for wandb logging.
        """
        tcs = response_token_counts if response_token_counts is not None else [None] * len(responses)
        fn = self.reward_fn
        totals = [fn(r, ex, response_token_count=tc, max_new_tokens=max_new_tokens)
                  for r, ex, tc in zip(responses, examples, tcs)]
        return totals, {}
