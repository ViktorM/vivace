# vLLM Integration Guide — `full_ids` Reconstruction

## The Problem

The training pipeline (`compute_token_logprobs`, `compute_loss`, `rl_step`)
operates on `full_ids` — a tensor of token IDs shaped `[B*G, S]` containing
**prompt + response** concatenated, left-padded. It also needs `plen` — the
padded prompt length — so it can mask out prompt tokens from the loss.

**HF sampler gives you this for free.** `model.generate()` returns the full
`[B*G, prompt_len + response_len]` tensor directly. `plen` is just
`enc["input_ids"].shape[1]`. Done.

**vLLM gives you `RequestOutput`s.** `generate()` returns `(raw_outputs, texts)` —
per-completion `token_ids` and `text`, no padded tensor. You must reconstruct
`full_ids` and `plen` yourself.

Three approaches, ordered from simplest to most robust.

---

## Approach 1: Re-tokenize prompt+response as one string

**Idea:** Concatenate prompt text + response text, tokenize the whole thing.

```python
prompts_flat = [p for p in unique_prompts for _ in range(G)]
responses_flat = [r for group in vllm_output for r in group]

full_texts = [p + r for p, r in zip(prompts_flat, responses_flat)]
enc = tokenizer(full_texts, return_tensors="pt", padding=True)
full_ids = enc.input_ids.to(device)
```

**Problem 1 — plen is ambiguous:** Each response has a different length,
so with left-padding the prompt doesn't start at the same position in every
sequence. `plen` as a single int breaks.

**Problem 2 — token boundary mismatch:** BPE/SentencePiece tokenizers
may produce different tokens when tokenizing "promptresponse" as one string
vs tokenizing "prompt" and "response" separately and concatenating. Example:

    prompt ends with "\n", response starts with "<"
    Separately: [..., token("\n"), token("<"), ...]
    Together:   [..., token("\n<"), ...]             ← merged into one token!

This means `full_ids[:, :plen]` won't match the original prompt tokens,
and the response mask in `compute_token_logprobs` will be off by one or
more positions. The log-probs will be subtly wrong, the importance ratio
will be wrong, and training will produce garbage — silently.

**Verdict:** Don't use this approach. The token boundary problem is a
silent correctness bug.

---

## Approach 2: Use vLLM's output token IDs (recommended)

**Idea:** vLLM's `RequestOutput` contains `prompt_token_ids` and each
`CompletionOutput` contains `token_ids`. Concatenate them directly — no
re-tokenization, no boundary problem.

But there's a subtlety: you must ensure vLLM uses the **exact same
tokenization** as your HF tokenizer. The safest way is to pass
`prompt_token_ids` to `llm.generate()` instead of raw text, so vLLM
doesn't re-tokenize the prompt itself.

### Step-by-step implementation

**Step 1: Tokenize prompts ONCE with your HF tokenizer.**

```python
unique_prompts = [self.env.format_prompt(ex) for ex in batch_ex]

# Left-pad prompts to uniform length (same as HF sampler does)
prompt_enc = self.tokenizer(unique_prompts, return_tensors="pt", padding=True)
prompt_ids_batch = prompt_enc.input_ids.tolist()   # list of list[int], shape [B, plen]
plen = prompt_enc.input_ids.shape[1]               # single int, uniform across batch
```

**Step 2: Pass token IDs to vLLM (not raw text), leading pads stripped.**

```python
# vLLM has no attention_mask: fed the pads it conditions on them and diverges from HF.
stripped = [ids[next((j for j, t in enumerate(ids) if t != pad_id), len(ids)):]
            for ids in prompt_ids_batch]
outputs, _ = self.rollout_worker.generate(
    prompt_token_ids=stripped, temperature=rl.temperature,
    top_p=rl.top_p, max_tokens=rl.max_new_tokens, n=G,
)
```

**Step 3: Collect response token IDs; right-pad them.**

```python
all_resp_ids, responses = [], []
for req_output in outputs:
    for completion in req_output.outputs:
        all_resp_ids.append(list(completion.token_ids))
        responses.append(completion.text)

# True lengths before padding drive both the attention and loss masks
# (pad_id == eos_id on Qwen, so pads can't be told from EOS afterwards).
response_lengths = torch.tensor([len(r) for r in all_resp_ids], device=self.device)
max_resp_len = max(len(r) for r in all_resp_ids)
all_resp_ids = [r + [pad_id] * (max_resp_len - len(r)) for r in all_resp_ids]
```

**Step 4: Prepend the *padded* prompt from step 1 (not the stripped one).**

```python
# [left_pad | prompt | response | right_pad], every row plen + max_resp_len wide
all_ids = [prompt_ids_batch[b] + all_resp_ids[b * G + g]
           for b in range(len(outputs)) for g in range(G)]
full_ids = torch.tensor(all_ids, device=self.device)
```

**Why this works:**
- `plen` is still a single int: prompts were padded to the same length in
  step 1 and responses on the right, so every response starts at target
  index `plen - 1`.
- The loss mask needs no forward pass:
  `build_response_mask(plen, S, response_lengths)` is `[plen-1, plen-1+resp_len)`.
- `compute_token_logprobs` masks left pads with `1 - cumprod(is_pad)` and right
  pads with `pos < plen + resp_len`, then sets `position_ids = cumsum(mask) - 1`
  so RoPE starts at 0 on the first real token — the positions vLLM used on the
  stripped prompt. Without it, a differently padded batch shifts every rotary
  angle and HF/vLLM logprobs disagree.

### Why strip the pads instead of letting vLLM see them?

Correctness, not efficiency. HF masks pads via `attention_mask`; vLLM has no
such input, so `[pad ... pad | prompt]` yields different logits than `[prompt]`
and the rollout policy no longer matches what `compute_token_logprobs` scores.
The padded `prompt_ids_batch` still builds `full_ids`, so `plen` stays one int;
per-sequence prompt lengths (Approach 3) would drop the padding entirely.

---

## Approach 3: Per-sequence prompt lengths (cleanest, future)

**Idea:** Change `plen` from a single `int` to a tensor `[B*G]` where
each entry is the (un-padded) prompt length for that sequence. This
removes all padding-related workarounds.

Requires changing `compute_token_logprobs`'s mask logic from:

```python
start = max(prompt_len - 1, 0)        # single int
mask[:, start:] = ...
```

to per-sequence masking:

```python
for i in range(B):
    start = max(prompt_lens[i] - 1, 0)
    mask[i, start:] = ...
```

(or vectorized with `torch.arange` broadcasting)

**Benefits:**
- vLLM receives un-padded prompts (no wasted compute on padding tokens)
- Works naturally with variable-length prompts (multi-turn, tool-use)
- Cleaner conceptually

**Cost:**
- Touches the `compute_token_logprobs` API (everything that passes `plen`
  needs updating)
- The vectorized mask construction is trickier to get right

**Recommendation:** Implement Approach 2 now. Move to Approach 3 when
you add multi-turn or tool-use (where prompts genuinely have different
lengths within a batch).

---

## Summary

| Approach | Correctness | Complexity | When to use |
|----------|-------------|------------|-------------|
| 1. Re-tokenize | Broken (token boundaries) | Simple | Never |
| 2. vLLM token IDs + uniform plen | Correct | Medium | Now |
| 3. Per-sequence plen | Correct + efficient | Higher | Multi-turn / tool-use |

**Approach 2 is what ships** (`rollout_phase`, vLLM branch). The key steps:
1. Tokenize prompts with HF tokenizer → `prompt_ids_batch`, `plen`
2. Strip leading pads; pass `prompt_token_ids=` to `rollout_worker.generate`
3. Record `response_lengths`; right-pad `completion.token_ids`
4. `full_ids` = padded prompt + padded response
5. `plen` stays a single int; `build_response_mask` is the loss mask, no forward needed

---

## Checklist for the vLLM path in rollout_phase (all shipped)

- [x] Build `unique_prompts` (B prompts, NOT repeated G times)
- [x] Tokenize with HF tokenizer → `prompt_ids_batch`, `plen`; strip leading pads for vLLM
- [x] `rollout_worker.generate(prompt_token_ids=..., n=G)` — vLLM handles G-duplication internally
- [x] Flatten outputs: `[B][G]` → `[B*G]` for token IDs and response strings; record `response_lengths`
- [x] Build `examples` list: `[ex for ex in batch_ex for _ in range(G)]` (same as HF path)
- [x] Construct `full_ids`: padded prompt + right-padded response
- [x] Compute rewards, advantages, ref_logp — identical to HF path (`old_logp` comes from `rl_step`'s epoch-1 forward)
- [x] Pack micro_batch dict — same keys as HF path

---

## Future cleanup: encapsulate vLLM generation in the worker

Half done. `VLLMRolloutWorker.generate()` accepts `prompt_token_ids` and
`rollout_phase` calls it (no more reaching into `rollout_worker.llm`), but it
returns the raw `RequestOutput`s, so `rollout_phase` still reads
`completion.token_ids` / `completion.text` itself.

Remaining step:

```python
def generate(self, prompt_token_ids: list[list[int]], *,
             temperature: float, top_p: float, top_k: int,
             max_tokens: int, n: int,
) -> tuple[list[list[str]], list[list[list[int]]]]:
    """Returns (response_texts[B][G], response_token_ids[B][G][T])"""
```

and the `full_ids` construction moves into the worker or a shared helper. Low
priority: the raw-output access is three lines, and `verify_weights_match` reads
`outputs[0].outputs[0].logprobs`, so the raw return has a second caller.
