"""Inspect eval samples from a completed training run.

Reads `eval_samples_final.json` (written by the trainer's final-eval block)
and prints per-sample question + response with a clear label for which list
(CORRECT / INCORRECT), char + token lengths, and reward.

Token counting needs a tokenizer; defaults to the Qwen2.5-0.5B-Instruct
tokenizer used by the current configs. Override with --tokenizer for other
models. Token length is computed in this script if the saved samples don't
already carry it (older runs predate that field).

Usage:
    python -m tests.inspect_eval_samples --run-dir runs/dapo_rloo_nocompile_seed42_
    python -m tests.inspect_eval_samples --run-dir runs/<r> --filter incorrect --n 10
    python -m tests.inspect_eval_samples --run-dir runs/<r> --interactive --filter incorrect
    python -m tests.inspect_eval_samples --run-dir runs/<r> --top-longest 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path


def _load(run_dir: str) -> dict:
    p = Path(run_dir) / "eval_samples_final.json"
    if not p.exists():
        raise SystemExit(f"no eval_samples_final.json in {run_dir} — run finished before "
                         "the sample-save patch landed, or eval_samples_final.json was moved")
    with open(p) as f:
        return json.load(f)


def _ensure_token_lengths(items: list[dict], tokenizer_name: str) -> None:
    """Populate `length_tokens` on every item; mutates in place. No-op if already present."""
    if not items:
        return
    if "length_tokens" in items[0]:
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    for x in items:
        x["length_tokens"] = len(tok.encode(x["response"], add_special_tokens=False))


def _print_sample(tag: str, idx: int, total: int, x: dict) -> None:
    """Print one sample with the formatting Viktor asked for: clear label, blank line
    between header and response, both char and token length."""
    print()
    cap_marker = "  [CAPPED]" if x.get("capped") else ""
    print(f"=== {tag}  {idx + 1}/{total} ==={cap_marker}")
    print(f"Q: {x['question']}")
    print(
        f"GT={x['ground_truth']}  PRED={x['predicted']}  REWARD={x['reward']:.3f}  "
        f"CHARS={len(x['response'])}  TOKENS={x['length_tokens']}"
    )
    print()
    print(f"RESP:\n{x['response']}")


def _filter_items(data: dict, which: str) -> list[tuple[str, list[dict]]]:
    """Return [(label, list)] pairs for the selected filter."""
    if which == "correct":
        return [("CORRECT", data["correct"])]
    if which == "incorrect":
        return [("INCORRECT", data["incorrect"])]
    return [("CORRECT", data["correct"]), ("INCORRECT", data["incorrect"])]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="inspect_eval_samples")
    p.add_argument("--run-dir", required=True,
                   help="run directory containing eval_samples_final.json")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF tokenizer name for token-length counting "
                        "(only used if samples don't already carry length_tokens)")
    p.add_argument("--filter", choices=["correct", "incorrect", "all"], default="all",
                   help="which list to show")
    p.add_argument("--n", type=int, default=5,
                   help="number of random samples per list (ignored with --interactive / --top-longest)")
    p.add_argument("--seed", type=int, default=0,
                   help="random sampling seed for reproducible inspection")
    p.add_argument("--interactive", action="store_true",
                   help="step through samples one at a time; [Enter] = next, q = quit")
    p.add_argument("--top-longest", type=int, default=0,
                   help="instead of random, show the N longest-response samples (by tokens)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    data = _load(args.run_dir)
    n_correct, n_incorrect = len(data["correct"]), len(data["incorrect"])
    n_total = n_correct + n_incorrect
    n_capped = sum(1 for x in data["correct"] + data["incorrect"] if x.get("capped"))
    print(f"Loaded {n_correct} correct + {n_incorrect} incorrect "
          f"({n_capped}/{n_total} = {100.0 * n_capped / max(n_total, 1):.1f}% capped) "
          f"from {args.run_dir}/eval_samples_final.json")

    # Token lengths: fast no-op if samples already carry the field; one tokenizer load otherwise.
    _ensure_token_lengths(data["correct"] + data["incorrect"], args.tokenizer)

    groups = _filter_items(data, args.filter)

    if args.top_longest > 0:
        for label, items in groups:
            top = sorted(items, key=lambda x: x["length_tokens"], reverse=True)[: args.top_longest]
            for i, x in enumerate(top):
                _print_sample(f"{label} (longest)", i, len(top), x)
        return 0

    if args.interactive:
        for label, items in groups:
            for i, x in enumerate(items):
                _print_sample(label, i, len(items), x)
                cmd = input("\n[Enter] next, q to quit, s to skip rest of this list: ").strip()
                if cmd == "q":
                    return 0
                if cmd == "s":
                    break
        return 0

    rng = random.Random(args.seed)
    for label, items in groups:
        if not items:
            print(f"\n=== {label} (empty) ===")
            continue
        sample = rng.sample(items, min(args.n, len(items)))
        for i, x in enumerate(sample):
            _print_sample(label, i, len(sample), x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
