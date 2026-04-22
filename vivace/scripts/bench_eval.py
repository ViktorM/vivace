"""Benchmark eval performance: HF generate vs vLLM.

Usage:
    python -m vivace.scripts.bench_eval --model Qwen/Qwen2.5-1.5B --n 200 500
    python -m vivace.scripts.bench_eval --model Qwen/Qwen2.5-1.5B --n 200 500 --vllm-gpu 1
"""

from __future__ import annotations

import argparse
import os
import time
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from vivace.envs.gsm8k import GSM8KEnv
from vivace.eval.runner import evaluate_model


def parse_args():
    p = argparse.ArgumentParser(prog="bench-eval")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--n", nargs="+", type=int, default=[200, 500])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--trainer-gpu", type=int, default=0)
    p.add_argument("--vllm-gpu", type=int, default=None,
                   help="GPU for vLLM. If not set, skip vLLM benchmark.")
    p.add_argument("--gpu-mem-util", type=float, default=0.80)
    return p.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.trainer_gpu}"

    # --- Load model + tokenizer ---
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    model.eval()

    # --- Load env ---
    env = GSM8KEnv()
    eval_data = env.load_split("eval")
    print(f"Eval set: {len(eval_data)} examples")

    # --- Build vLLM worker if requested ---
    vllm_worker = None
    if args.vllm_gpu is not None:
        from vivace.rollout.vllm_worker import VLLMRolloutWorker
        print(f"Building vLLM worker on GPU {args.vllm_gpu}...")
        vllm_worker = VLLMRolloutWorker(
            model_name=args.model,
            gpu_ids=[args.vllm_gpu],
            gpu_memory_utilization=args.gpu_mem_util,
            enforce_eager=False,
        )

    # --- Run benchmarks ---
    print(f"\n{'='*60}")
    print(f"  Eval Benchmark: {args.model}")
    print(f"  batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}")
    print(f"{'='*60}")
    print(f"{'Backend':<10} {'N':>6} {'Time (s)':>10} {'Tok/s':>10} {'Accuracy':>10}")
    print(f"{'-'*60}")

    for n in args.n:
        # HF benchmark
        torch.cuda.synchronize()
        t0 = time.time()
        hf_metrics, _, _ = evaluate_model(
            model, tokenizer, eval_data, env,
            n=n, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device, vllm_worker=None,
        )
        torch.cuda.synchronize()
        hf_time = time.time() - t0
        hf_tps = n * hf_metrics["avg_length_tokens"] / hf_time if hf_time > 0 else 0
        print(f"{'HF':<10} {n:>6} {hf_time:>10.1f} {hf_tps:>10.0f} {hf_metrics['accuracy_pct']:>9.1f}%")

        # vLLM benchmark
        if vllm_worker is not None:
            torch.cuda.synchronize()
            t0 = time.time()
            vllm_metrics, _, _ = evaluate_model(
                model, tokenizer, eval_data, env,
                n=n, batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                device=device, vllm_worker=vllm_worker,
            )
            torch.cuda.synchronize()
            vllm_time = time.time() - t0
            vllm_tps = n * vllm_metrics["avg_length_tokens"] / vllm_time if vllm_time > 0 else 0
            speedup = hf_time / vllm_time if vllm_time > 0 else 0
            print(f"{'vLLM':<10} {n:>6} {vllm_time:>10.1f} {vllm_tps:>10.0f} {vllm_metrics['accuracy_pct']:>9.1f}%  ({speedup:.1f}x)")

            # Sanity check: accuracy should match
            if abs(hf_metrics["accuracy_pct"] - vllm_metrics["accuracy_pct"]) > 2.0:
                print(f"  WARNING: accuracy mismatch! HF={hf_metrics['accuracy_pct']:.1f}% vs vLLM={vllm_metrics['accuracy_pct']:.1f}%")

    print(f"{'='*60}")

    # Cleanup
    if vllm_worker is not None:
        del vllm_worker.llm
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
