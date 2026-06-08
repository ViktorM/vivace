"""math_python — first concrete MultiTurnEnv.

THEORY
======

Wraps the existing Hendrycks MATH train + MATH-500 eval, but lets the model
call a Python interpreter mid-generation. The model emits

    <python>
    import sympy
    x = sympy.symbols('x')
    print(sympy.solve(x**2 - 2, x))
    </python>

The rollout loop:
  1. vLLM stops when it produces `</python>`
  2. `parse_tool_call` extracts the code block between the opening / closing tags
  3. The handler runs the code in a sandbox, captures stdout
  4. The output is inserted into the trajectory wrapped in `<result>...</result>`
  5. vLLM resumes with the new context

The model is trained to use these tool calls when arithmetic / symbolic
computation would be more reliable than chain-of-thought, then box the final
answer in `\\boxed{...}` as in the single-turn MATH env.

WHAT TO IMPLEMENT
=================

This file is a skeleton. The hot spots that need real code:

1. `python_exec` — the actual sandbox. The skeleton uses a naive
   `subprocess.run` placeholder; that's NOT safe for training (it'll happily
   `rm -rf /` if the model emits that). Pick a real sandbox: nsjail, Docker
   per-call, or a long-lived Jupyter kernel pool. See `docs/toolrl_design.md`
   for the trade-offs.

2. `MathPythonEnv.parse_tool_call` — robust `<python>` / `</python>` parser.
   The hint in the docstring is enough; just write it.

3. `MathPythonEnv.format_prompt` — needs the system message that TELLS the
   model the tool is available and how to call it. Otherwise the policy
   never emits `<python>` tags and the env is just MATH with extra latency.

4. `MathPythonEnv.reward_fn` — typically the SAME math_verify reward as
   `MATHEnv`. The reward function operates on the concatenated assistant
   text (skipping tool outputs), so existing math_verify pipelines work.

5. `MathPythonEnv.tool_reward` — optional. Start with default (0.0). After
   the first runs, look at logs and decide if per-call shaping is needed.

THIS FILE DOES NOT TOUCH
========================
- The rollout loop (`vivace/rollout/multiturn.py`)
- The trainer (`vivace/train/trainer.py`)
- The loss (`vivace/algos/policy_gradient.py`)

If those need changes to plumb tool envs end-to-end, do those changes
separately. This env should be implementable and unit-testable on its own.
"""

from __future__ import annotations

import re
import subprocess
from typing import Callable

from vivace.envs.base import Example
from vivace.envs.multiturn_base import MultiTurnEnv, Trajectory, ToolCall


# ============================================================================
# 1. The Python execution tool
# ============================================================================


def python_exec(code: str, timeout: float = 5.0) -> str:
    """Run `code` in a sandboxed Python interpreter.

    SECURITY: the implementation below is NOT a sandbox. It runs untrusted
    model output with full file system access. DO NOT use this for any
    real training run. Replace with one of:

    - **nsjail**: lowest overhead, namespaces + seccomp isolation.
      ~20 ms startup per call. Suitable for training-rate (1000s of calls/min).
    - **Docker per-call**: clean isolation, ~500 ms startup per call. Too
      slow for training; OK for eval where call volume is small.
    - **Long-lived Jupyter kernel pool**: isolation via separate Python
      processes, near-zero per-call startup. Best throughput; more
      implementation work to manage the pool.

    Returns: stdout if execution succeeds, "Error: <msg>" otherwise.
    Never raises — the model needs to see and learn to recover from errors.
    """
    # PLACEHOLDER — replace this whole block with a real sandbox.
    # Keep the signature and the return-error-as-string contract.
    raise NotImplementedError(
        "python_exec is a placeholder. Wire a real sandbox here. "
        "See docs/toolrl_design.md for the trade-off matrix."
    )

    # Reference shape for after you wire the sandbox (DO NOT just uncomment):
    #
    # try:
    #     result = subprocess.run(
    #         ["nsjail", "--config", "/etc/nsjail-python.cfg", "--",
    #          "python3", "-c", code],
    #         capture_output=True, timeout=timeout, text=True,
    #     )
    #     return result.stdout or result.stderr or "(no output)"
    # except subprocess.TimeoutExpired:
    #     return "Error: timeout"
    # except Exception as e:
    #     return f"Error: {type(e).__name__}: {e}"


# ============================================================================
# 2. The MultiTurnEnv subclass
# ============================================================================


# Regex shapes you'll likely want. Define them at module scope so they
# compile once. The exact patterns belong to your implementation.
_PYTHON_OPEN_TAG = "<python>"
_PYTHON_CLOSE_TAG = "</python>"
_RESULT_OPEN_TAG = "<result>"
_RESULT_CLOSE_TAG = "</result>"
_FINAL_ANSWER_RE = re.compile(r"\\boxed\{[^{}]+\}")    # matches \boxed{42}


class MathPythonEnv(MultiTurnEnv):
    """Hendrycks MATH with a Python-exec tool.

    Wraps the existing MATH train + MATH-500 eval splits and tokenizer
    contract. Adds the `<python>...</python>` tool-call protocol.
    """

    name = "math_python"
    tool_stop_strings = (_PYTHON_CLOSE_TAG,)

    # Generation budget: per-turn cap is moderate (the model needs room
    # to reason between tool calls). Total cap holds the trajectory bounded.
    max_new_tokens_per_turn: int = 512
    max_total_tokens: int = 4096
    max_turns: int = 8

    def __init__(self, corpus: str = "hendrycks", reward_overrides: dict | None = None):
        # HINT: lean on the existing single-turn MATHEnv for split loading
        # and reward configuration. The tool layer doesn't need to re-implement
        # data loading. Pseudocode:
        #
        #     from vivace.envs.math import MATHEnv
        #     self._inner = MATHEnv(corpus=corpus, reward_overrides=reward_overrides)
        #
        # Then `load_split` and `format_prompt` and `reward_fn` can mostly
        # delegate to `self._inner`, with `format_prompt` augmenting the
        # system message to advertise the tool.
        raise NotImplementedError("Wire up the inner MATHEnv and store it on self.")

    # --- Env contract --------------------------------------------------------

    def load_split(self, split: str) -> list[Example]:
        """Delegate to the wrapped MATHEnv. Tool support doesn't change the data."""
        raise NotImplementedError("Return self._inner.load_split(split)")

    def format_prompt(self, example: Example) -> str:
        """Format the math problem AND tell the model the tool exists.

        IMPLEMENTATION HINTS
        --------------------
        - Take the wrapped MATHEnv's prompt as a base.
        - Prepend a system instruction explaining the `<python>...</python>`
          syntax and that results come back wrapped in `<result>...</result>`.
        - Final answer convention stays the same: `\\boxed{...}` at the end.
        - Without this system message, the policy never emits tool calls
          (cold-start problem). You may want a short SFT warmup on a few
          curated tool-using examples to prime the behavior before RL —
          discuss with Viktor before implementing this.

        Reference prompt shape (you'll iterate on the wording):

            You are a math problem solver. You may use a Python interpreter
            to verify calculations by emitting <python>code</python>.
            The result will appear as <result>output</result>.
            End your solution with the final answer in \\boxed{...}.

            Problem: {example.problem}

            Solution:
        """
        raise NotImplementedError

    @property
    def reward_fn(self) -> Callable[[str, Example], float]:
        """Same math_verify-based reward as MATHEnv.

        The rollout loop passes ONLY the concatenated assistant text to
        the reward function (NOT the tool outputs). So existing math_verify
        logic that looks for `\\boxed{...}` works unchanged.
        """
        raise NotImplementedError("Return self._inner.reward_fn")

    # --- MultiTurnEnv contract -----------------------------------------------

    @property
    def tool_registry(self) -> dict[str, Callable[[str], str]]:
        """One tool: python_exec."""
        return {"python_exec": python_exec}

    def parse_tool_call(self, partial_response: str) -> ToolCall | None:
        """Extract a <python>...</python> call from the END of the response.

        IMPLEMENTATION HINTS
        --------------------
        - vLLM stops on `</python>`, so `partial_response` SHOULD end with
          the closing tag (possibly with trailing whitespace).
        - Find the LAST opening `<python>` before the closing tag.
        - If no matching opening tag, the model emitted a stray `</python>`
          — return None (malformed call; treat as no-call, let the model
          learn that this doesn't work).
        - The `args` of the ToolCall is the raw code between the tags
          (preserve whitespace; Python is whitespace-sensitive).
        - `span_start` = char offset of `<python>`; `span_end` = char offset
          just past `</python>`.

        Return None if the buffer doesn't end with `</python>` at all
        (model hit EOS or `max_tokens` for some other reason).
        """
        raise NotImplementedError(
            "Parse `<python>...</python>` from the end of `partial_response`. "
            "Return None if malformed or absent."
        )

    def is_done(self, traj: Trajectory) -> bool:
        """Terminate when the last assistant turn has `\\boxed{...}` OR budget exhausted.

        IMPLEMENTATION HINTS
        --------------------
        - Check budget first (cheap):
            if traj.total_new_tokens >= self.max_total_tokens: return True
            if len(traj.turns) >= self.max_turns: return True
        - Then check final answer in the LAST assistant turn:
            assistant_turns = [t for t in traj.turns if t.role == "assistant"]
            if not assistant_turns: return False
            return bool(_FINAL_ANSWER_RE.search(assistant_turns[-1].text))
        - Don't peek at intermediate assistant turns — the model might
          write `\\boxed{42}` early in its reasoning as an example and
          later correct itself. Only the LAST assistant turn counts.
        """
        raise NotImplementedError
