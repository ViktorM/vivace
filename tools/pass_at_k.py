"""Pass@K baseline for base models on eval envs.

Reports:
  pass@1     — greedy accuracy (T=0, single sample per problem)
  pass@k     — fraction of problems where ≥1 of k T-sampled responses is correct

Uses the same `env.format_prompt` and correctness reward as the training pipeline
so the baseline is directly comparable to wandb's eval/{env}/accuracy_pct.

Usage:
  .venv/bin/python tools/pass_at_k.py \\
    --model Qwen/Qwen2.5-0.5B --env gsm8k --k 8 \\
    [--max-new-tokens 1024] [--temperature 0.7] [--n-problems 100] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_USE_FASTOKENS", "1")  # Rust BPE backend (vLLM 0.22+ flag)


ENV_CLASSES = {
    "gsm8k":   ("vivace.envs.gsm8k",   "GSM8KEnv"),
    "math500": ("vivace.envs.math500", "MATH500Env"),
    "aime24":  ("vivace.envs.aime",    "AIME2024Env"),
    "aime25":  ("vivace.envs.aime",    "AIME2025Env"),
    "aime26":  ("vivace.envs.aime",    "AIME2026Env"),
}


def load_env(name: str):
    import importlib
    mod, cls = ENV_CLASSES[name]
    return getattr(importlib.import_module(mod), cls)()


def is_correct(env_name: str, response: str, answer: str) -> bool:
    """Reuse the SAME correctness reward the trainer logs as eval/{env}/accuracy."""
    from vivace.rewards import correctness_reward, math_correctness_reward
    if env_name == "gsm8k":
        return correctness_reward([response], [answer])[0] > 0
    return math_correctness_reward([response], [answer])[0] > 0  # math500, aime*


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--env", required=True, choices=list(ENV_CLASSES))
    ap.add_argument("--k", type=int, default=8, help="number of samples per problem")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-problems", type=int, default=None,
                    help="subset of eval split for smoke testing; default: full split")
    ap.add_argument("--gpu-memory-util", type=float, default=0.7)
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size; use 2 to shard 7B+ across both 4090s")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="default: prompt_max + max_new_tokens + 64")
    ap.add_argument("--out", default=None, help="optional per-problem CSV")
    ap.add_argument("--skip-greedy", action="store_true", help="skip pass@1 (sampled-only)")
    args = ap.parse_args()

    env = load_env(args.env)
    examples = env.load_split("eval")
    if args.n_problems is not None:
        examples = examples[: args.n_problems]
    n = len(examples)
    prompts = [env.format_prompt(ex) for ex in examples]
    answers = [ex.answer for ex in examples]

    print(f"Pass@K baseline")
    print(f"  model:       {args.model}")
    print(f"  env:         {args.env}  ({n} problems)")
    print(f"  k:           {args.k}")
    print(f"  temperature: {args.temperature}, top_p={args.top_p}, seed={args.seed}")
    print(f"  max_new:     {args.max_new_tokens}")

    from vllm import LLM, SamplingParams
    max_model_len = args.max_model_len or (max(len(p) for p in prompts) // 3 + args.max_new_tokens + 64)
    llm = LLM(model=args.model, dtype="bfloat16",
              tensor_parallel_size=args.tp,
              tokenizer_mode="fastokens",
              gpu_memory_utilization=args.gpu_memory_util,
              enforce_eager=True, max_model_len=max_model_len)

    # pass@1 — greedy
    if not args.skip_greedy:
        print()
        print(f"Generating greedy (T=0) over {n} prompts...")
        greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens))
        greedy_correct = sum(is_correct(args.env, g.outputs[0].text, a) for g, a in zip(greedy, answers))
        pass_at_1 = greedy_correct / n
        print(f"  pass@1 = {greedy_correct}/{n} = {100 * pass_at_1:.2f}%")
    else:
        pass_at_1 = None

    # pass@k — sampled
    print()
    print(f"Generating k={args.k} samples (T={args.temperature}) over {n} prompts...")
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, n=args.k, seed=args.seed)
    sampled = llm.generate(prompts, sp)
    rows, pass_k_count = [], 0
    for i, (req, ans) in enumerate(zip(sampled, answers)):
        n_correct = sum(is_correct(args.env, o.text, ans) for o in req.outputs)
        passed = int(n_correct > 0)
        pass_k_count += passed
        rows.append({"problem_id": i, "n_correct": n_correct, "k": args.k, "pass_at_k": passed})

    pass_at_k = pass_k_count / n
    print(f"  pass@{args.k} = {pass_k_count}/{n} = {100 * pass_at_k:.2f}%")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["problem_id", "n_correct", "k", "pass_at_k"])
            w.writeheader()
            w.writerows(rows)
        print(f"  per-problem CSV: {args.out}")

    print()
    print("=" * 60)
    print(f"Summary  {args.model}  /  {args.env}  ({n} problems)")
    if pass_at_1 is not None:
        print(f"  pass@1       = {100 * pass_at_1:5.2f}%")
    print(f"  pass@{args.k:<8} = {100 * pass_at_k:5.2f}%")
    if pass_at_1 is not None:
        headroom = pass_at_k - pass_at_1
        print(f"  sampling-reachable headroom (pass@{args.k} − pass@1) = {100 * headroom:+5.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
