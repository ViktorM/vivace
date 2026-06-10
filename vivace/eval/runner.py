"""Evaluation harness — pass@1, maj@k, pass@k.

Supports two generation backends:
  - HF: model.generate() — works everywhere, slower
  - vLLM: VLLMRolloutWorker.generate() — much faster, requires a worker instance

`evaluate_model` is the workhorse: greedy decode (T=0), score each response
with the env's reward function, return metrics + correct/incorrect lists.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch

from vivace.rewards import extract_answer, to_float


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    examples: list,
    env,
    n: int = 500,
    batch_size: int = 32,
    max_new_tokens: int = 192,
    device: str = "cuda",
    vllm_worker=None,
) -> tuple[dict, list, list]:
    """Greedy eval. Returns (metrics_dict, correct_list, incorrect_list).

    Args:
        model: HF model (used for HF backend, ignored if vllm_worker is set)
        tokenizer: HF tokenizer
        examples: list of Example objects from env.load_split("eval")
        env: Env instance (for format_prompt and reward_fn)
        n: number of examples to evaluate (-1 = all)
        batch_size: batch size for generation
        max_new_tokens: max response length
        device: device for HF generation
        vllm_worker: if provided, use vLLM for generation instead of HF
    """
    subset = examples if n == -1 else examples[:n]
    format_ok = correct = 0
    reward_sum = 0.0
    token_lengths: list[int] = []      # response length in tokens (primary)
    char_lengths: list[int] = []       # response length in characters (secondary)
    correct_list, incorrect_list = [], []
    t0 = time.time()

    if vllm_worker is not None:
        # --- vLLM path: batch all prompts, greedy decode ---
        all_responses = _eval_generate_vllm(
            vllm_worker, tokenizer, subset, env,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
    else:
        # --- HF path: batch through model.generate ---
        model.eval()
        all_responses = _eval_generate_hf(
            model, tokenizer, subset, env,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            device=device,
        )
        model.train()

    # --- Score responses ---
    # Correctness is env-owned: the same verifier as the reward path (numeric
    # for gsm8k, sympy/math_verify for the LaTeX math family). Both rewards and
    # correctness are computed BATCHED so the math family can fan LaTeX
    # comparisons out to the verify process pool instead of serial sympy.
    texts = [resp for resp, _ in all_responses]
    rewards, _ = env.reward_breakdown(texts, subset)
    correct_flags = env.is_correct_batch(texts, subset)

    n_capped = 0
    for ex, (resp, n_tok), r, ok in zip(subset, all_responses, rewards, correct_flags):
        ans = extract_answer(resp)
        if ans:
            format_ok += 1
        reward_sum += r
        char_lengths.append(len(resp))
        token_lengths.append(n_tok)
        capped = n_tok >= max_new_tokens
        if capped:
            n_capped += 1
        detail = {
            "question": ex.problem,
            "ground_truth": ex.answer,
            "predicted": ans,
            "reward": r,
            "response": resp,
            "length_tokens": n_tok,
            "capped": capped,
        }
        if ok:
            correct += 1
            correct_list.append(detail)
        else:
            incorrect_list.append(detail)

    total = len(subset)
    elapsed = time.time() - t0
    backend = "vllm" if vllm_worker else "hf"
    return (
        {
            "n": total,
            "format_rate_pct": 100.0 * format_ok / total,
            "accuracy_pct": 100.0 * correct / total,
            "avg_reward": reward_sum / total,
            "avg_length_tokens": float(np.mean(token_lengths)) if token_lengths else 0.0,
            "avg_length_chars": float(np.mean(char_lengths)) if char_lengths else 0.0,
            "n_capped": n_capped,
            "cap_rate_pct": 100.0 * n_capped / total if total else 0.0,
            "eval_time_s": elapsed,
            "eval_backend": backend,
            "n_correct": correct,
            "n_format_ok": format_ok,
            "reward_sum": reward_sum,
            "token_sum": float(np.sum(token_lengths)),
            "char_sum": float(np.sum(char_lengths)),
        },
        correct_list,
        incorrect_list,
    )


def _eval_generate_hf(
    model, tokenizer, examples, env,
    batch_size: int, max_new_tokens: int, device: str,
) -> list[tuple[str, int]]:
    """Greedy HF generation. Returns (text, generated_token_count) per example.
    Token count is the unpadded generated-portion length (excludes prompt + right-pad)."""
    all_out = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [env.format_prompt(ex) for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        plen = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_only = gen[:, plen:]
        responses = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        # Per-row real length: count non-pad positions. pad_id == eos_id with HF's
        # default => collapses EOS into pad, but that's the same first-stop signal.
        real_lens = (gen_only != tokenizer.eos_token_id).sum(dim=1).tolist()
        # If a row has zero pad/eos, it filled the budget — clamp to max_new_tokens.
        real_lens = [min(rl, max_new_tokens) if rl > 0 else max_new_tokens for rl in real_lens]
        all_out.extend(zip(responses, real_lens))
    return all_out


def _eval_generate_vllm(
    vllm_worker, tokenizer, examples, env,
    batch_size: int, max_new_tokens: int,
) -> list[tuple[str, int]]:
    """Greedy vLLM generation. Returns (text, generated_token_count) per example.

    All prompts go in ONE generate call — vLLM's continuous batching schedules
    them better than any outer chunking, which left the engine draining to
    near-empty between chunks. `batch_size` only bounds the HF path's memory.
    """
    from vllm import SamplingParams

    prompts = [env.format_prompt(ex) for ex in examples]
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=1)
    outputs = vllm_worker.llm.generate(
        prompts, sp,
        lora_request=vllm_worker._current_lora,
        use_tqdm=False,
    )
    return [(req.outputs[0].text, len(req.outputs[0].token_ids)) for req in outputs]


def sample_evaluate(
    examples: list,
    env,
    tokenizer,
    k: int = 8,
    n: int = 200,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.95,
    model=None,
    vllm_worker=None,
    device: str = "cuda",
) -> dict:
    """Generate k responses per prompt and compute pass@k + maj@k.

    Uses vLLM when available (much faster for k>1 via SamplingParams(n=k)).
    Falls back to HF generate with prompt duplication.

    Returns dict with pass_at_k, maj_at_k, and per-prompt details.
    """
    subset = examples if n == -1 else examples[:n]
    t0 = time.time()

    if vllm_worker is not None:
        responses_per_prompt = _sample_generate_vllm(
            vllm_worker, tokenizer, subset, env,
            k=k, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
        )
    elif model is not None:
        responses_per_prompt = _sample_generate_hf(
            model, tokenizer, subset, env,
            k=k, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            device=device,
        )
    else:
        raise ValueError("Either model or vllm_worker must be provided")

    elapsed = time.time() - t0

    return {
        "pass_at_k": pass_at_k(responses_per_prompt, subset, env, k),
        "maj_at_k": maj_at_k(responses_per_prompt, subset, env, k),
        "k": k,
        "n": len(subset),
        "eval_time_s": elapsed,
    }


@torch.no_grad()
def _sample_generate_hf(
    model, tokenizer, examples, env,
    k: int, max_new_tokens: int, temperature: float, top_p: float,
    device: str,
) -> list[list[str]]:
    """Generate k responses per prompt using HF generate (prompt duplication)."""
    model.eval()
    responses_per_prompt = []
    for ex in examples:
        prompt = env.format_prompt(ex)
        prompts_k = [prompt] * k
        enc = tokenizer(prompts_k, return_tensors="pt", padding=True).to(device)
        plen = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        responses = tokenizer.batch_decode(gen[:, plen:], skip_special_tokens=True)
        responses_per_prompt.append(responses)
    model.train()
    return responses_per_prompt


def _sample_generate_vllm(
    vllm_worker, tokenizer, examples, env,
    k: int, max_new_tokens: int, temperature: float, top_p: float,
) -> list[list[str]]:
    """Generate k responses per prompt using vLLM (native sibling sampling)."""
    prompts = [env.format_prompt(ex) for ex in examples]
    _, texts = vllm_worker.generate(
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=k,
    )
    return texts  # already [B][k]


def compare_metrics(before: dict, after: dict, label: str = "") -> None:
    print(f"\n{'=' * 65}")
    if label:
        print(label)
    if "n" in before:
        print(f"Eval set size: {int(before['n'])}")
    print(f"{'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"{'=' * 65}")
    for k in before:
        if k == "n":
            continue        # constant across before/after; printed once above
        b, a = before[k], after[k]
        if not isinstance(b, (int, float)):
            continue
        d = a - b
        arrow = "^" if d > 0.001 else "v" if d < -0.001 else "="
        is_int = isinstance(b, int) or (float(b).is_integer() and float(a).is_integer())
        if "pct" in k:
            fmt, fmt2, dfmt = f"{b:>9.1f}%", f"{a:>9.1f}%", f"{d:>+8.1f}%"
        elif is_int:
            fmt, fmt2, dfmt = f"{b:>10.0f}", f"{a:>10.0f}", f"{d:>+9.0f}"
        elif "length" in k:
            fmt, fmt2, dfmt = f"{b:>10.1f}", f"{a:>10.1f}", f"{d:>+9.1f}"
        else:
            fmt, fmt2, dfmt = f"{b:>10.3f}", f"{a:>10.3f}", f"{d:>+9.3f}"
        print(f"{k:<25} {fmt} {fmt2} {dfmt} {arrow}")
    print(f"{'=' * 65}")


def show_examples(correct: list, incorrect: list, n: int = 3) -> None:
    for label, items in [("CORRECT", correct), ("INCORRECT", incorrect)]:
        print(f"\n{label} ({len(items)} total):")
        for ex in items[:n]:
            print(f"  Q: {ex['question']}")
            print(f"  GT: {ex['ground_truth']}  Pred: {ex['predicted']}  R: {ex['reward']:.2f}")
            print(f"  {ex['response'][:300]}")
            print()


@torch.no_grad()
def preview_progress(
    model,
    tokenizer,
    examples: list,
    env,
    n: int = 3,
    max_new_tokens: int = 256,
    label: str = "",
    device: str = "cuda",
) -> None:
    """Quick visual sanity check during training. Greedy decode N random examples."""
    import random

    model.eval()
    subset = random.sample(examples, min(n, len(examples)))
    prompts = [env.format_prompt(ex) for ex in subset]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    plen = enc["input_ids"].shape[1]
    gen = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    responses = tokenizer.batch_decode(gen[:, plen:], skip_special_tokens=True)
    print(f"\n{'=' * 60}\n {label}\n{'=' * 60}")
    for ex, resp in zip(subset, responses):
        ans = extract_answer(resp)
        match = "OK" if ans == ex.answer else "MISS"
        print(f"Q: {ex.problem}")
        print(f"GT: {ex.answer}  Pred: {ans}  {match}")
        print(f"{resp[:400]}\n{'-' * 60}")
    model.train()


def _vote_key(ans: str) -> str:
    """Canonical voting key: numeric answers collapse ('72' == '72.0' == '72.00');
    everything else votes on the stripped string. LaTeX equivalence classes are
    NOT merged (would need a sympy parse per sample — too slow at eval scale)."""
    f = to_float(ans)
    return repr(f) if f is not None else ans.strip()


def pass_at_k(
    responses_per_prompt: list[list[str]], examples: list, env, k: int
) -> float:
    """Fraction of prompts with at least one correct answer in top-k samples.
    Correctness is `env.is_correct` — same verifier as the reward path."""
    assert len(responses_per_prompt) == len(examples)
    hits = sum(
        1 for samples, ex in zip(responses_per_prompt, examples)
        if any(env.is_correct(r, ex) for r in samples[:k])
    )
    return hits / max(len(examples), 1)


def maj_at_k(
    responses_per_prompt: list[list[str]], examples: list, env, k: int
) -> float:
    """Majority-vote accuracy at sample size k. Ties broken by first-seen.

    Votes are bucketed by `_vote_key` so numeric formatting variants pool their
    votes; the winning bucket's first response is then checked with
    `env.is_correct` (the full verifier, LaTeX-aware for math envs).
    """
    assert len(responses_per_prompt) == len(examples)
    hits = 0
    for samples, ex in zip(responses_per_prompt, examples):
        votes: Counter = Counter()
        representative: dict[str, str] = {}   # vote key -> first full response
        for r in samples[:k]:
            ans = extract_answer(r)
            if ans:
                key = _vote_key(ans)
                votes[key] += 1
                representative.setdefault(key, r)
        if not votes:
            continue
        winner, _ = votes.most_common(1)[0]
        if env.is_correct(representative[winner], ex):
            hits += 1
    return hits / max(len(examples), 1)
