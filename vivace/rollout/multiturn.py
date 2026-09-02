"""Multi-turn rollout loop — Option A (stateless re-prompt).

THEORY
======

The single-turn rollout (existing `vllm_worker.generate`) issues ONE
`llm.generate()` call per prompt and returns the full completion. For
ToolRL with Option A, we issue MULTIPLE calls per prompt:

    turn 1:    llm.generate(prompt,                              stop=[</python>])
    turn 2:    llm.generate(prompt + turn1_text + <result>...</result>, stop=[</python>])
    turn 3:    llm.generate(prompt + turn1+result1 + turn2_text + <result>...</result>, stop=[</python>])
    ...

Each call's input is the full prior trajectory. vLLM's prefix caching keeps
the previous call's tokens warm, so each turn prefills only the tool-output
tokens just inserted.

The loop exits when:
  - `env.is_done(traj)` returns True (final answer reached or budget hit)
  - The last assistant turn did NOT request a tool (model emitted final answer
    or EOS without a `<python>` call)

OUTPUTS (must match single-turn shape)
======================================

The trainer's rollout_phase and the loss math expect the same fields they get
from `vllm_worker.generate`:

- `response_token_ids`: list[list[int]] — assistant + tool tokens concatenated
  in trajectory order. NOT the prompt.
- `response_mask`: list[list[int]] — 1 for model-generated tokens, 0 for
  tool-output tokens. Same length as `response_token_ids`.
- `reward`: list[float] — terminal verifier reward per trajectory.
- Optional: `step_rewards`: list[list[float]] — per-turn shaping rewards.

NOTE: `compute_loss` already takes a per-token `mask`, so tool tokens with
mask=0 contribute zero gradient with no loss-code change. The plumbing job is
in the trainer: `mb["mask"]` today comes from `build_response_mask(...)` (a
contiguous span) and must instead be this per-token mask, shifted by one.

OPTION B (KV-cache continuation) NOTES
=======================================

Once Option A is validated, Option B (see docs/toolrl_design.md) replaces the
`llm.generate(prompt + concat(turns), ...)` calls with KV-cache continuation
across turns. Not a drop-in: needs a prefix-extension or logits-processor
hook, session lifecycle, and weight sync coordinated with the open session.
Same `Trajectory`, same trainer outputs; the loss math doesn't care.

CURRENT STATUS
==============

This file is a skeleton. The hot spots that need real code:

1. `rollout_multiturn` — the main loop. Pseudocode in the docstring; you
   write it.
2. `tokenize_tool_output` — convert a tool result string to token IDs
   in the same vocabulary as the assistant turns. The tokenizer is passed
   in (already loaded by the trainer).
3. `build_response_arrays` — concatenate the per-turn token IDs and masks
   into the flat arrays that match the single-turn `RolloutBatch` shape.

THIS FILE DOES NOT TOUCH
========================
- `vllm_worker.py` (the underlying `llm.generate` wrapper stays as-is)
- The trainer's RolloutBatch builder (you'll wire that up separately once
  this rollout function works in isolation)
- The loss math
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vivace.envs.base import Example
from vivace.envs.multiturn_base import MultiTurnEnv, ToolCall, Trajectory, Turn


# ============================================================================
# Output container (one per prompt; rollout returns a list of these)
# ============================================================================


@dataclass
class MultiTurnRolloutOutput:
    """All data the trainer needs from one multi-turn rollout.

    Mirrors the single-turn output shape. The trainer's rollout_phase pads B*G
    of these into [B*G, S] `full_ids` + mask micro-batches (not yet wired).

    Fields:
        prompt_token_ids:  prompt only (assistant context starts after this)
        response_token_ids: concatenated trajectory tokens (assistant + tool, in order)
        response_mask:      1 for assistant tokens, 0 for tool tokens — same length as response_token_ids
        response_text:      the concatenated *assistant text only* (for reward_fn)
        terminal_reward:    final verifier reward (env.reward_fn applied to response_text)
        step_rewards:       per-tool-call shaping rewards (sum is added to terminal in the trainer if desired)
        n_turns:            assistant turn count (for logging / debugging)
        truncated_by_budget: True if rollout hit max_total_tokens or max_turns before is_done — useful metric
        trajectory:         optional: the full Trajectory for debug logging / replay
    """
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_mask: list[int]
    response_text: str
    terminal_reward: float
    step_rewards: list[float]
    n_turns: int
    truncated_by_budget: bool
    trajectory: Trajectory | None = None


# ============================================================================
# Helpers
# ============================================================================


def tokenize_tool_output(text: str, tokenizer: Any) -> list[int]:
    """Tokenize a tool-result string in the SAME vocabulary as the policy.

    IMPLEMENTATION HINTS
    --------------------
    - Use the policy's tokenizer (the same one vLLM uses internally).
    - DO NOT add special tokens (BOS/EOS) — these are mid-trajectory tokens.
      Use `tokenizer.encode(text, add_special_tokens=False)`.
    - Wrap the raw tool output in delimiter tags (<result>...</result>) so the
      model can distinguish tool output from its own text. Decide whether to
      tokenize the tags as a single block with the result, or separately —
      matters for mask precision. Easiest: tokenize the whole wrapped string
      as one unit; mask the whole block as `0`.
    """
    raise NotImplementedError(
        "Tokenize text in the policy's vocabulary without special tokens. "
        "Wrap in <result>...</result> tags first."
    )


def build_response_arrays(traj: Trajectory) -> tuple[list[int], list[int], str]:
    """Flatten a trajectory into (response_token_ids, response_mask, response_text).

    Iterate through traj.turns IN ORDER, concatenating tokens and emitting
    masks per turn:
      - assistant turn:  mask = [1] * len(turn.token_ids)
      - tool turn:       mask = [0] * len(turn.token_ids)

    `response_text` is the concatenation of just the ASSISTANT turn texts.
    The reward function reads this; tool outputs aren't part of "what the
    model said."

    IMPLEMENTATION HINTS
    --------------------
    - Keep all three outputs consistent in length:
        len(response_token_ids) == len(response_mask)
        sum(response_mask) == total assistant tokens
    - response_text concatenates assistant turn texts WITHOUT the tool-call
      tags stripped — the reward function expects the full "what the model
      generated" output, which includes its own `<python>...</python>`
      blocks (those count as model output, just like any other token).
    - Do NOT include the prompt tokens here. The trainer attaches the prompt
      separately.
    """
    raise NotImplementedError


# ============================================================================
# Main rollout loop (Option A — stateless re-prompt)
# ============================================================================


def rollout_multiturn(
    env: MultiTurnEnv,
    example: Example,
    *,
    vllm_worker,                  # the existing VLLMRolloutWorker instance
    tokenizer,                    # HF tokenizer matching the policy vocab
    n: int = 1,                   # how many trajectories to generate per prompt
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int | None = None,
) -> list[MultiTurnRolloutOutput]:
    """Generate `n` multi-turn trajectories for `example` using Option A.

    PSEUDOCODE — write this body
    -----------------------------

        prompt = env.format_prompt(example)
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        outs: list[MultiTurnRolloutOutput] = []

        for traj_idx in range(n):
            traj = Trajectory(example=example, prompt=prompt,
                              prompt_token_ids=prompt_token_ids)
            truncated = False

            while True:
                # Build the full input string for this turn: prompt + all prior turns.
                # For Option A, every turn pays a re-prefill cost on the new tokens.
                turn_input = prompt + concat_turn_texts(traj.turns)

                # Cap per-turn tokens AND respect the trajectory budget
                budget_left = env.max_total_tokens - traj.total_new_tokens
                if budget_left <= 0:
                    truncated = True
                    break
                turn_max = min(env.max_new_tokens_per_turn, budget_left)

                # vLLM call — stop on any tool boundary or natural EOS
                outputs, _ = vllm_worker.generate(
                    prompts=[turn_input],
                    max_tokens=turn_max,
                    n=1,
                    temperature=temperature, top_p=top_p,
                    seed=(seed + traj_idx if seed is not None else None),
                    # HINT: you'll need to extend `vllm_worker.generate` to
                    # accept a `stop` parameter for the tool-stop strings.
                    # vLLM's SamplingParams supports `stop=...`.
                )
                completion = outputs[0].outputs[0]   # CompletionOutput
                assistant_text = completion.text
                assistant_ids = list(completion.token_ids)

                # Record the assistant turn (mask = 1)
                traj.turns.append(Turn(role="assistant",
                                       text=assistant_text,
                                       token_ids=assistant_ids))
                traj.total_new_tokens += len(assistant_ids)

                # Did the model request a tool, or end the trajectory?
                call = env.parse_tool_call(assistant_text)
                if call is None:
                    # No tool call → this is the final assistant turn
                    break

                # Execute the tool (handler returns str, never raises)
                handler = env.tool_registry[call.name]
                result_text = handler(call.args)
                step_reward = env.tool_reward(call, result_text)
                traj.step_rewards.append(step_reward)

                # Tokenize tool output (with <result>...</result> wrapper)
                tool_ids = tokenize_tool_output(result_text, tokenizer)
                traj.turns.append(Turn(role="tool",
                                       text=result_text,
                                       token_ids=tool_ids,
                                       tool_call=call,
                                       tool_result_summary=_summarize(result_text)))
                traj.total_new_tokens += len(tool_ids)

                # Budget / done check before next assistant turn
                if env.is_done(traj):
                    break
                if traj.total_new_tokens >= env.max_total_tokens:
                    truncated = True
                    break
                if len([t for t in traj.turns if t.role == "assistant"]) >= env.max_turns:
                    truncated = True
                    break

            # Build outputs
            response_token_ids, response_mask, response_text = build_response_arrays(traj)
            terminal_reward = env.reward_fn(response_text, example)
            traj.final_reward = terminal_reward

            outs.append(MultiTurnRolloutOutput(
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
                response_mask=response_mask,
                response_text=response_text,
                terminal_reward=terminal_reward,
                step_rewards=traj.step_rewards,
                n_turns=sum(1 for t in traj.turns if t.role == "assistant"),
                truncated_by_budget=truncated,
                trajectory=traj,
            ))

        return outs

    NOTES / GOTCHAS
    ---------------

    - **Different seeds per trajectory.** When `n>1`, each trajectory needs
      its own seed offset OR you'll get n identical rollouts (vLLM is
      deterministic at a given seed). Pattern above: `seed + traj_idx`.

    - **vllm_worker.generate currently does NOT accept a `stop` arg.** You'll
      need to extend it (or call vLLM directly here). The single-line change
      in `vllm_worker.generate` is to thread `stop` through `SamplingParams`.

    - **Concurrency.** For training-rate rollouts you'll want to batch across
      prompts AND trajectories within a turn (vLLM does the batching
      internally if you submit a list of prompts in one call). The
      pseudocode above is per-trajectory for clarity; you'll likely refactor
      to issue all in-flight turn-N calls together. Worth doing *after* the
      single-trajectory version is correct.

    - **Truncation as a metric.** Log `truncated_by_budget` per rollout.
      A spike in truncations means the model is going in circles or your
      budget is too tight — both useful signals.

    - **HF rollout path (vivace/rollout/hf_sampler.py).** The single-turn
      `Env` works in both vLLM and HF backends. ToolRL on the HF path is
      out of scope for v1 — it'd be useful for debugging but the throughput
      is too low for actual training. Don't try to make this generic
      across rollout backends until vLLM works end-to-end.

    - **n>1 for matched-budget groups.** Group advantages (grpo / dr_grpo /
      rloo) are per-prompt across the n trajectories of the SAME prompt. The
      trainer batches [B, G]; this function returns [G] for one prompt, so
      pass n = `cfg.rl.group_size` and let the trainer loop over B.
    """
    raise NotImplementedError(
        "Implement the multi-turn rollout loop. Pseudocode in the docstring."
    )


def _summarize(text: str, max_chars: int = 80) -> str:
    """First-line summary of a tool result, for logging / reward shaping."""
    text = text.strip().splitlines()[0] if text.strip() else "(empty)"
    return text[:max_chars] + ("..." if len(text) > max_chars else "")
