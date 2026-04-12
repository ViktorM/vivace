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

    def sample_batch(self, n: int, rng) -> list[Example]:
        """Default: uniform sampling from the training split. Override for curriculum."""
        train = self.load_split("train")
        idx = rng.choice(len(train), size=n, replace=False)
        return [train[int(i)] for i in idx]
