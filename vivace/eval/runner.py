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

from vivace.rewards import answer_match, extract_answer, to_float


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
    lengths = []
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
    for ex, resp in zip(subset, all_responses):
        ans = extract_answer(resp)
        if ans:
            format_ok += 1
        gt = to_float(ex.answer)
        pred = to_float(ans) if ans else None
        r = env.reward_fn(resp, ex)
        reward_sum += r
        lengths.append(len(resp))
        detail = {
            "question": ex.problem,
            "ground_truth": gt,
            "predicted": pred,
            "reward": r,
            "response": resp,
        }
        if answer_match(gt, pred):
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
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "eval_time_s": elapsed,
            "eval_backend": backend,
        },
        correct_list,
        incorrect_list,
    )


def _eval_generate_hf(
    model, tokenizer, examples, env,
    batch_size: int, max_new_tokens: int, device: str,
) -> list[str]:
    """Generate greedy responses using HF model.generate()."""
    all_responses = []
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
        responses = tokenizer.batch_decode(gen[:, plen:], skip_special_tokens=True)
        all_responses.extend(responses)
    return all_responses


def _eval_generate_vllm(
    vllm_worker, tokenizer, examples, env,
    batch_size: int, max_new_tokens: int,
) -> list[str]:
    """Generate greedy responses using vLLM worker. Much faster than HF."""
    from vllm import SamplingParams

    all_responses = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [env.format_prompt(ex) for ex in batch]

        # Use vLLM's generate directly — greedy (temperature=0), n=1
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=1)
        outputs = vllm_worker.llm.generate(
            prompts, sp,
            lora_request=vllm_worker._current_lora,
            use_tqdm=False,
        )
        for req_output in outputs:
            all_responses.append(req_output.outputs[0].text)
    return all_responses


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

    answers = [ex.answer for ex in subset]
    elapsed = time.time() - t0

    return {
        "pass_at_k": pass_at_k(responses_per_prompt, answers, k),
        "maj_at_k": maj_at_k(responses_per_prompt, answers, k),
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
    print(f"{'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"{'=' * 65}")
    for k in before:
        b, a = before[k], after[k]
        if isinstance(b, (int, float)):
            d = a - b
            arrow = "^" if d > 0.001 else "v" if d < -0.001 else "="
            fmt = f"{b:>9.1f}%" if "pct" in k else f"{b:>10.3f}"
            fmt2 = f"{a:>9.1f}%" if "pct" in k else f"{a:>10.3f}"
            dfmt = f"{d:>+8.1f}%" if "pct" in k else f"{d:>+9.3f}"
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


def pass_at_k(
    responses_per_prompt: list[list[str]], answers: list[str], k: int
) -> float:
    """Fraction of prompts with at least one correct answer in top-k samples."""
    assert len(responses_per_prompt) == len(answers)
    hits = 0
    for samples, gt in zip(responses_per_prompt, answers):
        any_correct = any(
            answer_match(to_float(gt), to_float(extract_answer(r)))
            for r in samples[:k]
        )
        if any_correct:
            hits += 1
    return hits / max(len(answers), 1)


def maj_at_k(
    responses_per_prompt: list[list[str]], answers: list[str], k: int
) -> float:
    """Majority-vote accuracy at sample size k. Ties broken by first-seen."""
    assert len(responses_per_prompt) == len(answers)
    hits = 0
    for samples, gt in zip(responses_per_prompt, answers):
        votes: Counter = Counter()
        for r in samples[:k]:
            ans = extract_answer(r)
            if ans:
                votes[ans] += 1
        if not votes:
            continue
        winner, _ = votes.most_common(1)[0]
        if answer_match(to_float(gt), to_float(winner)):
            hits += 1
    return hits / max(len(answers), 1)
