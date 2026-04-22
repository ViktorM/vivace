"""Hugging Face `model.generate` rollout — single-GPU dev path.

For early development and CI smoke tests, you don't want vLLM in the
critical path. This module wraps `model.generate` in the same interface
the trainer expects from a rollout backend, so you can train end-to-end
without ever building a vllm.LLM.

When you outgrow this (anything beyond ~hundreds of prompts/min on
single GPU), switch the trainer over to `vivace/rollout/vllm_worker.py`.
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
) -> tuple[torch.Tensor, list[str], int]:
    """Batched HF sampling. Generates one response per entry in `prompts`.

    Returns:
        full_ids: [B, prompt_len + response_len] (LEFT-PADDED prompt + response)
        response_strings: list of decoded response strings (length B)
        prompt_len: int — the padded prompt length so the trainer can split

    BATCHING NOTE
    -------------
    This is a single batched `model.generate` call — `len(prompts)` forward
    passes happen in parallel, one kernel launch, one KV cache allocation.
    It is NOT a Python loop over prompts.

    For GROUP sampling (G responses per unique prompt, as in GRPO), the
    caller duplicates each prompt G times in the input list:

        prompts = [p for ex in batch_ex for p in [make_prompt(ex)] * G]

    Each duplicate gets a different continuation because `do_sample=True`
    + temperature introduces sampling noise. This function is unaware of
    groups — it just sees B*G distinct prompts and returns B*G responses.

    This is slightly wasteful on the prompt side: HF recomputes the prompt
    KV cache G times (once per duplicate). vLLM's `SamplingParams(n=G)`
    is the more efficient production path — it shares the prompt KV cache
    across siblings and only diverges during the decode phase. See
    `vivace/rollout/vllm_worker.py::generate`. `hf_sampler` trades that
    efficiency for simplicity and zero non-torch dependencies, which is
    what you want during single-GPU dev.

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

    HINTS
    -----
    - enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    - plen = enc["input_ids"].shape[1]
    - was_training = model.training
    - model.eval()
    - gen = model.generate(**enc, do_sample=True, max_new_tokens=max_new_tokens,
                           temperature=temperature, top_p=top_p,
                           pad_token_id=tokenizer.eos_token_id)
    - if was_training: model.train()
    - responses = tokenizer.batch_decode(gen[:, plen:], skip_special_tokens=True)
    - return gen, responses, plen

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

    responses = tokenizer.batch_decode(gen[:, plen:], skip_special_tokens=True)

    return gen, responses, plen
