"""Hugging Face `model.generate` rollout — single-GPU dev path (`use_vllm: false`).

Wraps `model.generate` in the rollout-backend interface the trainer expects, so
smoke tests and early dev train end-to-end without building a vllm.LLM. Tops
out at ~hundreds of prompts/min on one GPU; past that use `vllm_worker.py`
(`use_vllm: true`, the default).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def sample_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[str], int, torch.Tensor]:
    """Batched HF sampling. Generates one response per entry in `prompts`.

    Returns:
        full_ids: [B, prompt_len + response_len] (LEFT-PADDED prompt + right-padded response)
        response_strings: list of decoded response strings (length B)
        prompt_len: int — the padded prompt length so the trainer can split
        response_lengths: [B] long tensor — true per-sample response length
            BEFORE right-padding. Includes the EOS token when present; equals
            max_new_tokens for sequences that didn't emit EOS. The trainer
            uses this to build correct attention + loss masks (pad_id == eos_id
            means we can't tell right-pads from real EOS by token value alone).

    BATCHING NOTE
    -------------
    One batched `model.generate` call, not a Python loop: all `len(prompts)`
    sequences decode in lockstep, sharing each step's kernels and one KV cache.

    GROUP sampling (G responses per prompt, as in GRPO): the caller repeats each
    prompt G times (trainer: `np.repeat(prompts, G)`); `do_sample=True` +
    temperature gives each copy a different continuation. This function is
    group-unaware — B*G prompts in, B*G responses out.

    Cost: HF prefills the prompt KV G times, once per copy. vLLM's
    `SamplingParams(n=G)` shares the prompt KV across siblings and diverges only
    at decode (`vllm_worker.py::generate`); hf_sampler trades that for
    simplicity and zero non-torch dependencies.

    THEORY
    ------
    HF tokenizers default to right-padding, but `model.generate` with batched
    prompts requires LEFT-padding (otherwise the position-ids and attention
    mask end up wrong because pad tokens "interrupt" the prompt). The
    tokenizer must be configured with `tokenizer.padding_side = "left"` and
    `tokenizer.pad_token = tokenizer.eos_token` BEFORE calling this function.
    Do this in the trainer's __init__, not here.

    The was_training save/restore around eval()/train() is necessary
    because dropout would otherwise stay disabled after sampling.

    GOTCHAS
    -------
    - do_sample=True is required for temperature/top_p to actually do
      anything. Without it generate() just does greedy.
    - pad_token_id=tokenizer.eos_token_id avoids a warning AND ensures
      eval-time padding behaves correctly.
    - The returned full_ids contains BOTH prompt and response. The trainer
      uses prompt_len to slice them apart for log-prob recomputation.
    - Don't run this under torch.compile — generate() and torch.compile
      don't get along.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device) # dict
    plen = enc["input_ids"].shape[1]

    was_training = model.training
    model.eval()
    gen = model.generate(**enc, do_sample=True, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=top_p,
                         pad_token_id=tokenizer.eos_token_id)
    if was_training:
        model.train()

    # Response portion (may contain right-pads for samples that emitted EOS
    # before others in the batch — generate() fills with pad_token_id=eos).
    resp_ids = gen[:, plen:]  # [B, R]
    eos_id = tokenizer.eos_token_id
    R = resp_ids.shape[1]

    # True response length = index of first EOS + 1 (include EOS); if no EOS,
    # the whole response is real (hit max_new_tokens without stopping).
    is_eos = (resp_ids == eos_id)
    any_eos = is_eos.any(dim=1)
    first_eos = is_eos.int().argmax(dim=1)   # 0 when no EOS; gated by any_eos below
    response_lengths = torch.where(any_eos, first_eos + 1, torch.full_like(first_eos, R))

    responses = tokenizer.batch_decode(resp_ids, skip_special_tokens=True)

    return gen, responses, plen, response_lengths
