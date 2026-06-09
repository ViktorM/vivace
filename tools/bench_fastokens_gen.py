"""fastokens vs HF tokenizer timing at the output lengths we actually use.

CPU-only — no GPU generation needed. fastokens accelerates *tokenization*
(encode + streaming detokenization), not GPU decode, so the honest "improvement
table" is encode/decode wall-time at 256 / 1024 / 4096 token outputs.

This is what the earlier subprocess-per-cell version got wrong: it paid a full
vLLM cold-init (~1-2 min) per cell for ~seconds of real work. This runs the
whole table in a few seconds.

Run: .venv/bin/python tools/bench_fastokens_gen.py [--model ...] [--lengths 256,1024,4096]
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--lengths", default="256,1024,4096")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(args.model)

    import fastokens
    fastokens.patch_transformers()
    ft = AutoTokenizer.from_pretrained(args.model)

    # Realistic token stream: tokenize a chunk of math-ish prose, slice to length.
    sample = ("To solve this problem, we first note that the sum of the digits "
              "must be divisible by 9. Let x = 7, then we compute step by step. ") * 200
    base_ids = hf.encode(sample, add_special_tokens=False)

    def bench_decode(tok, ids, n):
        t0 = time.perf_counter()
        for _ in range(n):
            tok.decode(ids, skip_special_tokens=False)
        return (time.perf_counter() - t0) * 1000 / n  # ms per call

    def bench_encode(tok, text, n):
        t0 = time.perf_counter()
        for _ in range(n):
            tok.encode(text, add_special_tokens=False)
        return (time.perf_counter() - t0) * 1000 / n

    print(f"fastokens vs HF tokenizer — {args.model}")
    print(f"({args.iters} iters/cell, CPU-only)")
    print()
    print(f"{'output_len':>10} {'op':>7} {'hf_ms':>9} {'ft_ms':>9} {'speedup':>9} {'match':>7}")
    print("-" * 56)
    for L in lengths:
        ids = base_ids[:L]
        text = hf.decode(ids, skip_special_tokens=False)
        # decode (detokenization — the generation-time cost)
        hf_d = bench_decode(hf, ids, args.iters)
        ft_d = bench_decode(ft, ids, args.iters)
        dmatch = hf.decode(ids) == ft.decode(ids)
        print(f"{L:>10} {'decode':>7} {hf_d:>9.2f} {ft_d:>9.2f} {hf_d/ft_d:>8.2f}x {('y' if dmatch else 'N'):>7}")
        # encode (prompt-time cost, for reference)
        hf_e = bench_encode(hf, text, args.iters)
        ft_e = bench_encode(ft, text, args.iters)
        ematch = hf.encode(text, add_special_tokens=False) == ft.encode(text, add_special_tokens=False)
        print(f"{L:>10} {'encode':>7} {hf_e:>9.2f} {ft_e:>9.2f} {hf_e/ft_e:>8.2f}x {('y' if ematch else 'N'):>7}")
    print()
    print("Note: GPU decode dominates end-to-end generation wall-time, so these")
    print("tokenizer speedups translate to a small % of total at training/eval time.")
    print("The win is largest for long shared prompts (tool-use) and batch detok.")


if __name__ == "__main__":
    main()
