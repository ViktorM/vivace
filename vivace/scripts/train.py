"""CLI entry point for `vivace-train`.

Usage:
    vivace-train --config vivace/configs/grpo_gsm8k_0.5b_lora.yaml
    vivace-train --config <path> --num-steps 5    # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

from vivace.algos.types import RLConfig
from vivace.train.trainer import Trainer, TrainerConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_trainer_config(cfg_dict: dict) -> TrainerConfig:
    """Instantiate TrainerConfig from a parsed YAML dict.

    The YAML uses a nested `rl:` block for RL hyperparameters (see RLConfig).
    This helper converts that nested dict into an RLConfig instance before
    constructing TrainerConfig, since dataclasses don't auto-convert nested
    dicts into nested dataclasses.

    Tuple fields (e.g. lora_target_modules) are also coerced from YAML lists.
    """
    if "lora_target_modules" in cfg_dict:
        cfg_dict["lora_target_modules"] = tuple(cfg_dict["lora_target_modules"])
    rl_dict = cfg_dict.pop("rl", None) or {}
    cfg_dict["rl"] = RLConfig(**rl_dict)
    return TrainerConfig(**cfg_dict)


def _maybe_enable_vllm_callable_rpc(cfg_dict: dict) -> None:
    """Allow pickling user callables through vLLM's collective_rpc.

    vLLM 0.19's default serializer rejects user-defined callables — without
    this flag, the NCCL weight-sync path (which ships Python callables to the
    vLLM worker subprocess) fails at encode time. Must be set before the vLLM
    subprocess is spawned (i.e., before Trainer is instantiated) so the
    subprocess inherits it. Only set when NCCL sync is actually configured —
    don't relax serialization defaults when not needed.
    """
    if cfg_dict.get("weight_sync_method") == "nccl":
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vivace-train")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--num-steps", type=int, default=None, help="override TrainerConfig.num_steps")
    p.add_argument("--run-dir", type=str, default=None, help="override TrainerConfig.run_dir")
    p.add_argument("--seed", type=int, default=None, help="override TrainerConfig.seed")
    p.add_argument("--weight-sync-method", choices=["disk", "nccl"], default=None,
                   help="override TrainerConfig.weight_sync_method (disk | nccl)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg_dict = load_config(args.config)

    # CLI overrides win over YAML
    if args.num_steps is not None:
        cfg_dict["num_steps"] = args.num_steps
    if args.run_dir is not None:
        cfg_dict["run_dir"] = args.run_dir
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    if args.weight_sync_method is not None:
        cfg_dict["weight_sync_method"] = args.weight_sync_method

    _maybe_enable_vllm_callable_rpc(cfg_dict)

    cfg = build_trainer_config(cfg_dict)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
