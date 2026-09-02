"""CLI entry point for `vivace-eval` — run greedy eval on a checkpoint.

Loads a model + tokenizer + env, runs `evaluate_model` against the
eval split, prints metrics. Useful for checking accuracy of a saved
checkpoint without re-running training.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vivace-eval")
    p.add_argument("--checkpoint", required=True, help="path to a checkpoint dir")
    p.add_argument("--env", default="gsm8k", help="env name")
    p.add_argument("--n", type=int, default=200, help="number of examples to eval")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    # TODO: load `<run_dir>/final_model` (peft adapter + tokenizer, or a full HF
    # checkpoint for full FT), build the env, call vivace.eval.runner.evaluate_model.
    # Only periodic ckpt-<step> dirs wait on checkpointing.load_checkpoint.
    raise NotImplementedError(
        f"vivace-eval is a stub until checkpointing.load_checkpoint exists. "
        f"args: {vars(args)}"
    )


if __name__ == "__main__":
    main()
