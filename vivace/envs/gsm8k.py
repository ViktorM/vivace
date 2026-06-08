"""GSM8K environment.

Parsing, prompt formatting, and reward wiring for GSM8K. Loads
`openai/gsm8k` via HuggingFace datasets, parses the `####`-delimited
answer field, and exposes the train/test splits as `Example` lists.

The system prompt asks the model to respond in
`<think>...</think><answer>...</answer>` format. This makes the format
rewards in `vivace/rewards.py` meaningful.
"""

from __future__ import annotations

from typing import Callable

from vivace.envs.base import Env, Example
from vivace.rewards import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    gsm8k_reward_batch,
    gsm8k_reward_single,
)

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


def parse_gsm8k_row(row: dict) -> Example:
    """Split GSM8K's `answer` field into reasoning + final answer."""
    parts = row["answer"].split("####")
    reasoning = parts[0].strip() if len(parts) > 1 else ""
    answer = parts[-1].strip().replace(",", "")
    return Example(
        problem=row["question"],
        answer=answer,
        metadata={"reasoning": reasoning, "id": str(hash(row["question"]))},
    )


class GSM8KEnv(Env):
    """OpenAI GSM8K grade-school math word problems.

    Usage:
        env = GSM8KEnv()
        train = env.load_split("train")     # ~7.5K examples
        eval_ = env.load_split("eval")      # ~1.3K examples
        prompt = env.format_prompt(train[0])
    """

    name = "gsm8k"

    def __init__(self, n_eval: int | None = None, reward_overrides: dict | None = None):
        self.n_eval = n_eval
        self._splits: dict[str, list[Example]] = {}
        if reward_overrides:
            base = DEFAULT_REWARD_CONFIG.__dict__
            self._reward_cfg: RewardConfig = RewardConfig(**{**base, **reward_overrides})
        else:
            self._reward_cfg = DEFAULT_REWARD_CONFIG

    def _load(self) -> None:
        if self._splits:
            return
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main")
        self._splits["train"] = [parse_gsm8k_row(ex) for ex in ds["train"]]
        eval_examples = [parse_gsm8k_row(ex) for ex in ds["test"]]
        if self.n_eval is not None:
            eval_examples = eval_examples[: self.n_eval]
        self._splits["eval"] = eval_examples

    def load_split(self, split: str) -> list[Example]:
        self._load()
        if split not in self._splits:
            raise ValueError(f"unknown split: {split!r} (expected 'train' or 'eval')")
        return self._splits[split]

    def format_prompt(self, example: Example) -> str:
        return make_prompt(example.problem)

    @property
    def reward_fn(self) -> Callable:
        cfg = self._reward_cfg
        if cfg is DEFAULT_REWARD_CONFIG:
            return gsm8k_reward_single   # fast path — no closure when no override
        def _scored(
            response: str, example,
            response_token_count: int | None = None,
            max_new_tokens: int | None = None,
        ) -> float:
            tokens = [response_token_count] if response_token_count is not None else None
            return gsm8k_reward_batch(
                [response], [example.answer], cfg,
                response_token_counts=tokens, max_new_tokens=max_new_tokens,
            )[0]
        return _scored
