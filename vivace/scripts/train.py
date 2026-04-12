"""CLI entry point for `vivace-train`.

Usage:
    vivace-train --config vivace/configs/grpo_gsm8k_0.5b_lora.yaml
    vivace-train --config <path> --num-steps 5    # quick smoke test
"""

from __future__ import annotations

import argparse
import sys

import yaml

from vivace.train.trainer import Trainer, TrainerConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vivace-train")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--num-steps", type=int, default=None, help="override TrainerConfig.num_steps")
    p.add_argument("--run-dir", type=str, default=None, help="override TrainerConfig.run_dir")
    p.add_argument("--seed", type=int, default=None, help="override TrainerConfig.seed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg_dict = load_config(args.config)

    # Tuples vs lists for hashable fields
    if "lora_target_modules" in cfg_dict:
        cfg_dict["lora_target_modules"] = tuple(cfg_dict["lora_target_modules"])

    # CLI overrides win over YAML
    if args.num_steps is not None:
        cfg_dict["num_steps"] = args.num_steps
    if args.run_dir is not None:
        cfg_dict["run_dir"] = args.run_dir
    if args.seed is not None:
        cfg_dict["seed"] = args.seed

    cfg = TrainerConfig(**cfg_dict)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
