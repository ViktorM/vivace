"""CLI entry point for `vivace-sft` — standalone SFT warmup.

Loads an SFT YAML config, builds a model + tokenizer + dataset, runs
the SFT training loop, and writes a checkpoint that `vivace-train` can
later resume from.

Stub: the SFT loop itself lives in `vivace/algos/sft.py` and is left
to be implemented.
"""

from __future__ import annotations

import argparse
import sys

import yaml


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vivace-sft")
    p.add_argument("--config", required=True, help="path to SFT YAML config")
    p.add_argument("--output", default="runs/sft", help="checkpoint output dir")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    with open(args.config) as f:
        _cfg = yaml.safe_load(f)
    # TODO: wire this into vivace.algos.sft.sft_train_loop once that
    # function is implemented. See docs/IMPLEMENTATION_NOTES.md.
    raise NotImplementedError("vivace-sft is a stub until vivace/algos/sft.py is implemented")


if __name__ == "__main__":
    main()
