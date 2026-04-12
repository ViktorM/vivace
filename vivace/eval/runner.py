"""Evaluation harness — pass@1, maj@k, pass@k, plus helpers.

`evaluate_model` is the workhorse: greedy decode (T=0), batch through
the model, score each response with the env's reward function, return
metrics + correct/incorrect lists. `pass_at_k` and `maj_at_k` provide
higher-sample metrics.
"""

from __future__ import annotations

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
    n: int = 100,
    batch_size: int = 16,
    max_new_tokens: int = 192,
    device: str = "cuda",
) -> tuple[dict, list, list]:
    """Greedy eval. Returns (metrics_dict, correct_list, incorrect_list)."""
    model.eval()
    subset = examples[:n]
    format_ok = correct = 0
    reward_sum = 0.0
    lengths = []
    correct_list, incorrect_list = [], []

    for i in range(0, len(subset), batch_size):
        batch = subset[i : i + batch_size]
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

        for ex, resp in zip(batch, responses):
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

    model.train()
    total = len(subset)
    return (
        {
            "n": total,
            "format_rate_pct": 100.0 * format_ok / total,
            "accuracy_pct": 100.0 * correct / total,
            "avg_reward": reward_sum / total,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        },
        correct_list,
        incorrect_list,
    )


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
    """Fraction of prompts with at least one correct answer in top-k samples.

    `responses_per_prompt[i]` is a list of >=k samples for prompt i.
    `answers[i]` is the ground-truth answer string for prompt i.
    """
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
