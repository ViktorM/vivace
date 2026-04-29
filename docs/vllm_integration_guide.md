# vLLM Integration Guide — `full_ids` Reconstruction

## The Problem

The training pipeline (`compute_token_logprobs`, `compute_loss`, `rl_step`)
operates on `full_ids` — a tensor of token IDs shaped `[B*G, S]` containing
**prompt + response** concatenated, left-padded. It also needs `plen` — the
padded prompt length — so it can mask out prompt tokens from the loss.

**HF sampler gives you this for free.** `model.generate()` returns the full
`[B*G, prompt_len + response_len]` tensor directly. `plen` is just
`enc["input_ids"].shape[1]`. Done.

**vLLM gives you strings.** `generate()` returns `list[list[str]]` — response
texts only, no prompt, no token IDs. You must reconstruct `full_ids` and
`plen` yourself.

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

**Step 2: Pass token IDs to vLLM (not raw text).**

```python
sp = SamplingParams(
    temperature=cfg.temperature, top_p=cfg.top_p,
    max_tokens=cfg.max_new_tokens, n=G,
)
outputs = self.rollout_worker.llm.generate(
    prompt_token_ids=prompt_ids_batch,
    sampling_params=sp,
)
```

This way vLLM sees the exact tokens your tokenizer produced — no
re-tokenization, no boundary issues.

**Step 3: Build full_ids from vLLM's output.**

```python
all_ids = []       # will be [B*G] list of list[int]
all_responses = [] # will be [B*G] list of str

for req_output in outputs:
    p_ids = list(req_output.prompt_token_ids)  # includes left-padding
    for completion in req_output.outputs:
        r_ids = list(completion.token_ids)
        all_ids.append(p_ids + r_ids)
        all_responses.append(completion.text)
```

**Step 4: Left-pad to uniform total length.**

```python
max_len = max(len(ids) for ids in all_ids)
pad_id = self.tokenizer.pad_token_id

full_ids = torch.tensor(
    [[pad_id] * (max_len - len(ids)) + ids for ids in all_ids],
    device=self.device,
)
```

**Why this works:**
- `plen` is still a single int — all prompts were padded to the same
  length in step 1.
- Prompt tokens in `full_ids` start at position `left_pad_offset + 0`
  where `left_pad_offset = max_len - len(this_sequence)`. This is
  exactly the same left-padding convention as the HF sampler path.
- The response mask in `compute_token_logprobs` uses
  `start = max(plen - 1, 0)` which is measured from position 0 of the
  un-padded content. Since the left-padding is accounted for by the
  attention mask (`cumprod(is_pad)`), the mask logic works correctly.

### What about `prompt_token_ids` including padding?

Yes — when you tokenize with `padding=True`, shorter prompts get
left-padded with `pad_token_id`. These padding tokens become part of
`prompt_token_ids` passed to vLLM. vLLM will see them as real tokens
and "attend" to them. This is slightly wasteful (vLLM processes padding)
but correct — the attention mask in `compute_token_logprobs` handles it.

For better vLLM efficiency, you could pass un-padded prompt IDs (each
prompt has different length) and then re-pad the full_ids yourself.
But this makes `plen` per-sequence, which requires API changes
(Approach 3). Not worth the complexity for now.

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

**Start with Approach 2.** The key steps are:
1. Tokenize prompts with HF tokenizer → `prompt_ids_batch`, `plen`
2. Pass `prompt_token_ids=prompt_ids_batch` to vLLM
3. Concatenate `prompt_token_ids + completion.token_ids` per response
4. Left-pad all sequences to `max_len`
5. `plen` stays a single int, everything downstream works unchanged

---

## Checklist for the vLLM path in rollout_phase

- [ ] Build `unique_prompts` (B prompts, NOT repeated G times)
- [ ] Tokenize with HF tokenizer → `prompt_ids_batch`, `plen`
- [ ] Build `SamplingParams(n=G, ...)` — vLLM handles G-duplication internally
- [ ] Call `llm.generate(prompt_token_ids=prompt_ids_batch, sampling_params=sp)`
- [ ] Flatten outputs: `[B][G]` → `[B*G]` for both token IDs and response strings
- [ ] Build `examples` list: `[ex for ex in batch_ex for _ in range(G)]` (same as HF path)
- [ ] Construct `full_ids` tensor: left-pad `prompt_ids + response_ids` to `max_len`
- [ ] Compute rewards, advantages, old_logp, ref_logp — identical to HF path from here
- [ ] Pack micro_batch dict — same keys as HF path

---

## Future cleanup: encapsulate vLLM generation in the worker

The current implementation calls `self.rollout_worker.llm.generate()` directly
from `rollout_phase`, reaching into the worker's internal `LLM` instance.
This works but breaks encapsulation — `rollout_phase` knows about vLLM's
`RequestOutput` structure, `prompt_token_ids`, `completion.token_ids`, etc.

A cleaner alternative: extend `VLLMRolloutWorker.generate()` to accept
`prompt_token_ids` and return both texts AND token IDs:

```python
def generate(self, prompt_token_ids: list[list[int]], *,
             temperature: float, top_p: float, top_k: int,
             max_tokens: int, n: int,
) -> tuple[list[list[str]], list[list[list[int]]]]:
    """Returns (response_texts[B][G], response_token_ids[B][G][T])"""
```

Then `rollout_phase` calls `self.rollout_worker.generate(prompt_ids_batch, ...)`
and gets back everything it needs without knowing about vLLM internals.
The `full_ids` construction moves into the worker or into a shared helper.

Do this when the current approach is validated and stable — not before.
