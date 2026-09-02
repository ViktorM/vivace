"""MultiTurnEnv — tool-using rollout interface.

THEORY (read this first)
========================

The existing `Env` ABC assumes one prompt → one response → one reward.
ToolRL breaks all three:

    one prompt → assistant turn → tool call → tool result → assistant turn → ... → reward

Each "turn" is either a stretch of model-generated tokens OR a stretch of
deterministic tool-output tokens. The KEY distinction for RL:

- **Assistant turns** are sampled from the policy. They carry policy gradient.
  The `response_mask` is 1 over these tokens, so they participate in the loss.

- **Tool-output turns** are deterministic (`subprocess.run`, math kernel,
  search API, etc.). They have **no policy logprob** and the model can't
  control them. The `response_mask` is 0 over these tokens — they appear in
  the context (so the next assistant turn can condition on them) but they
  do NOT enter the loss.

This mask plumbing is the core data-structural change. If you get the
boundaries right, the existing loss math works unchanged — `compute_loss`
takes one `mask` for every `loss_type`.

DESIGN DECISION: stateless re-prompt (Option A)
================================================

This skeleton assumes the **stateless re-prompt** rollout (Option A from
`docs/toolrl_design.md`). Each turn sends `prompt + concat(all_prior_turns)`
as a fresh `llm.generate()` call to vLLM. Pros: trivial to implement, no
new vLLM internals. Cons: pays a re-prefill cost on every turn (vLLM's
prefix caching helps but not perfectly).

The alternative (Option B — stateful KV cache continuation) requires either
vLLM's logits-processor hook OR a prefix-extension API plus session-lifecycle
management. Save for v1.1 after Option A is validated end-to-end.

WHAT TO IMPLEMENT
=================

Subclasses override three abstract methods plus `reward_fn` from the base
`Env`. See `vivace/envs/math_python.py` for the first concrete example.

HINTS / GOTCHAS
===============

- `parse_tool_call` should be a streaming-aware string parser: vLLM stops on
  a `stop=` string (e.g. `</python>`) and you parse just the partial buffer
  it returns. Don't try to detect calls mid-token — wait for the stop token.

- `is_done` should check (a) was the LAST assistant turn final-answer-shaped
  (e.g. contains `\\boxed{...}`), AND (b) is the total trajectory token budget
  exhausted. Return True on either.

- Tool outputs that are too long are a real footgun. Truncate or summarize
  before inserting into the trajectory; otherwise a single `python_exec`
  output can blow your context window.

- Errors during tool execution should be RETURNED AS TEXT to the model
  (`"Error: timeout"`, `"Error: NameError ..."`), never raised. The model
  needs to learn to recover from broken tool calls — that's part of the
  learning signal.

- For the loss-mask plumbing: the rollout loop builds two parallel lists,
  `all_token_ids` (the full trajectory) and `all_masks` (1 for assistant
  tokens, 0 for tool tokens). These flow into `RolloutBatch.response_mask`
  unchanged from the single-turn path.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal

from vivace.envs.base import Env, Example


@dataclass
class ToolCall:
    """One detected tool call in a partial assistant response.

    `span_start` / `span_end` are character offsets into the partial response
    string. The rollout loop uses these to extract the tool args and to know
    where the model wants the tool result inserted.

    For Option A (stateless re-prompt), spans aren't strictly needed — you
    concatenate the full prior text and the tool result. But carrying them
    keeps the door open for Option B (stateful KV continuation) without
    breaking the interface.
    """
    name: str                  # tool registry key, e.g. "python_exec"
    args: str                  # raw arg string the tool will receive
    span_start: int            # char offset in the partial response where the call begins
    span_end: int              # char offset where the call ends (after the closing stop string)


@dataclass
class Turn:
    """One segment of the trajectory: either model-generated or tool output.

    `role="assistant"` turns have policy logprob and carry response_mask=1.
    `role="tool"` turns are deterministic and carry response_mask=0.
    """
    role: Literal["assistant", "tool"]
    text: str                  # the text content of this turn
    token_ids: list[int]       # tokenized form (vLLM gives these for assistant;
                               # tokenize manually for tool outputs)
    # Optional per-call metadata for reward shaping / debugging:
    tool_call: ToolCall | None = None       # set if this assistant turn ended in a tool call
    tool_result_summary: str | None = None  # e.g. "ok" / "Error: timeout" — for reward_fn


@dataclass
class Trajectory:
    """The full conversation for one rollout: prompt + interleaved turns.

    The `response_mask` is computed by concatenating turn-level masks:
    `[1]*len(assistant_tokens) + [0]*len(tool_tokens) + ...`

    `total_new_tokens` tracks how many tokens we've generated/inserted past
    the prompt — used to enforce `max_total_tokens`.

    `final_reward` is set by the env's `reward_fn` after `is_done` triggers.
    Optional `step_rewards` allow per-tool-call shaping (penalty for failed
    Python, bonus for using a verified tool, etc.).
    """
    example: Example
    prompt: str
    prompt_token_ids: list[int]
    turns: list[Turn] = field(default_factory=list)
    total_new_tokens: int = 0
    final_reward: float | None = None
    step_rewards: list[float] = field(default_factory=list)


class MultiTurnEnv(Env):
    """Env that supports tool calls during rollout.

    All three abstract methods must be implemented. The base class also
    inherits `reward_fn` from `Env` — your subclass should provide that as
    a property pointing at a function `(response_text, example) -> float`
    just like single-turn envs. For ToolRL the "response_text" is the
    concatenation of just the assistant turns (NOT the tool outputs);
    most verifier-style rewards (math_verify, regex extractors) work on
    that concatenation unchanged.
    """

    name: str = "multiturn-base"

    # vLLM `stop=` strings marking a tool-call boundary; the rollout loop then
    # hands the buffer to `parse_tool_call`. Per env, e.g. ("</python>",).
    tool_stop_strings: tuple[str, ...] = ()

    @property
    @abstractmethod
    def tool_registry(self) -> dict[str, Callable[[str], str]]:
        """name → handler. Handler signature: `args: str -> result: str`.

        IMPLEMENTATION HINTS
        --------------------
        - Build the dict lazily inside the subclass `__init__`, not at class
          load time — sandboxed handlers (nsjail, kernel pool) often need
          per-instance state.
        - Handlers must NEVER raise. Wrap in try/except and return the error
          message as a string. The model must learn to recover from failures.
        - Handlers should enforce their own timeouts (e.g. `subprocess.run(...,
          timeout=5)`). A runaway Python call kills the rollout's wall budget.
        """
        raise NotImplementedError(
            "Subclasses must return a dict mapping tool names to callables. "
            "See math_python.py for the python_exec example."
        )

    @abstractmethod
    def parse_tool_call(self, partial_response: str) -> ToolCall | None:
        """Detect a tool call at the end of `partial_response`.

        Called AFTER vLLM stops on one of the `tool_stop_strings`. The string
        contains everything the model emitted in the current turn including
        the stop token.

        Returns:
            ToolCall(name, args, span_start, span_end) if a complete, well-formed
            call is at the end of the buffer.
            None if the model stopped for some other reason (EOS, max_tokens, etc.)
            — in that case the rollout loop treats this as the final answer.

        IMPLEMENTATION HINTS
        --------------------
        - Use a robust parser, not just `str.rfind`. The model WILL emit
          malformed calls during training (unclosed tags, nested calls,
          attempts at multi-tool routing). Return None on malformed calls
          and let the model learn from the absence of a tool result.
        - The `args` string passes through verbatim to the tool handler.
          If your tool expects structured input (JSON), parse it inside the
          handler, not here — keep this layer dumb.
        - `span_start` should mark where the OPENING tool tag is in the
          partial response; `span_end` should mark just past the CLOSING tag
          (including the stop string itself). Option A doesn't strictly use
          spans, but logging / reward shaping will want them.
        """
        raise NotImplementedError

    @abstractmethod
    def is_done(self, traj: Trajectory) -> bool:
        """Decide whether `traj` is terminal.

        Called after every assistant turn (before tool execution). Return True
        when:
        - The last assistant turn produced a final answer (e.g. has `\\boxed{...}`)
        - OR the trajectory has hit `max_total_tokens` (budget exhausted)
        - OR the model emitted EOS without requesting a tool

        IMPLEMENTATION HINTS
        --------------------
        - For math: detect the marker `reward_fn` extracts — `<answer>...</answer>`
          for the single-turn math envs, not `\\boxed{}` — in the LAST assistant
          turn's text. If present → done.
        - Be careful with "is_done" being too lenient: if the model writes
          `\\boxed{...}` AND then keeps thinking, you probably still want to
          let it finish that turn but mark done after it. The simplest rule:
          done iff the LAST assistant turn (just completed) contains a
          well-formed final answer.
        - Trajectory budget enforcement is the safety net. Always include it.
        """
        raise NotImplementedError

    # ----- optional reward shaping (default: no shaping) ---------------------

    def tool_reward(self, call: ToolCall, result: str) -> float:
        """Per-tool-call shaping reward. Default: 0.0 (no shaping).

        Override to penalize bad tool use (e.g. SyntaxError) or bonus useful
        calls (e.g. arithmetic-relevant Python). Returned values are summed
        into `Trajectory.step_rewards` by the rollout loop. The terminal
        verifier reward (`reward_fn`) is the dominant signal; step_rewards
        is for shaping.

        Typical magnitudes:
          - Per-call penalty for failed tool: -0.1
          - Per-call bonus for verified-correct intermediate: +0.05
          - Per-call cost for excessive use (>10 calls): -0.02 each

        Keep step rewards small relative to the terminal reward (~2.0 for
        a correct math answer) — they're regularizers, not the main signal.
        """
        return 0.0

    # ----- config knobs ------------------------------------------------------

    # Per-turn cap = max_tokens of one llm.generate call; max_total_tokens bounds
    # the whole trajectory. Subclasses override.
    max_new_tokens_per_turn: int = 512
    max_total_tokens: int = 4096
    max_turns: int = 8                # hard limit on turn count regardless of tokens
