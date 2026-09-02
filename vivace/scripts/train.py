"""CLI entry point for `vivace-train`.

Usage:
    # Basic (single GPU)
    vivace-train --config vivace/configs/gsm8k/dapo_0.5b_colo.yaml
    vivace-train --config <path> --num-steps 5                   # smoke test

    # Common hyperparameter overrides (explicit flags)
    vivace-train --config <path> --lr 3e-5 --batch-size 8 --group-size 8
    vivace-train --config <path> --algo cispo --loss cispo --kl-coef 0.02
    vivace-train --config <path> --lora-rank 32 --gpu-mem-util 0.45

    # Anything else via the generic --set escape hatch (dotted path, yaml value):
    vivace-train --config <path> --set rl.clip_cispo_high=10.0 \
                                 --set rl.adam_beta2=0.95 \
                                 --set rl.lr_restart=true
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def _peek_mode_from_argv() -> str:
    """Read `mode` from the YAML pointed to by --config without importing torch.

    We need to set PYTORCH_CUDA_ALLOC_CONF *before* torch initializes its CUDA
    allocator, but the right value depends on `mode`:
      - disaggregated: `expandable_segments:True` reduces fragmentation OOMs
        on long runs (we hit this at step 12 with the default allocator).
      - colocated: `expandable_segments` is INCOMPATIBLE with vLLM's sleep()
        which uses CUDA memory pools (PyTorch issue #147851). Must stay default.
    Returns "colocated" if config can't be parsed — safer default.
    """
    try:
        argv = sys.argv
        for i, a in enumerate(argv):
            if a == "--config" and i + 1 < len(argv):
                with open(argv[i + 1]) as f:
                    return yaml.safe_load(f).get("mode", "colocated")
            if a.startswith("--config="):
                with open(a.split("=", 1)[1]) as f:
                    return yaml.safe_load(f).get("mode", "colocated")
    except Exception:
        pass
    return "colocated"


if _peek_mode_from_argv() == "disaggregated":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Colocated: don't set; expandable_segments breaks vLLM sleep/wake. If you want
# fragmentation mitigation in colocated mode, try `max_split_size_mb:512`
# manually — that's pool-compatible.

from vivace.algos.types import RLConfig
from vivace.train.trainer import Trainer, TrainerConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_trainer_config(cfg_dict: dict) -> TrainerConfig:
    """Instantiate TrainerConfig from a parsed YAML dict.

    The nested `rl:` block becomes an RLConfig (dataclasses don't auto-convert
    nested dicts); `lora_target_modules` is coerced from YAML list to tuple.
    `profiling:` stays a dict — Trainer builds ProfilingConfig from it.
    """
    if "lora_target_modules" in cfg_dict:
        cfg_dict["lora_target_modules"] = tuple(cfg_dict["lora_target_modules"])
    rl_dict = cfg_dict.pop("rl", None) or {}
    cfg_dict["rl"] = RLConfig(**rl_dict)
    return TrainerConfig(**cfg_dict)


def _maybe_enable_vllm_callable_rpc(cfg_dict: dict) -> None:
    """Allow pickling user callables through vLLM's collective_rpc.

    vLLM's default serializer rejects user-defined callables — without this flag,
    sync paths that ship Python callables (NCCL receiver loop, IPC apply loop)
    fail at encode time. Must be set before the vLLM subprocess is spawned
    (i.e., before Trainer is instantiated) so the subprocess inherits it.
    Only set when actually needed — don't relax serialization defaults otherwise.
    """
    if cfg_dict.get("weight_sync_method") in ("nccl", "ipc"):
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vivace-train")
    p.add_argument("--config", required=True, help="path to YAML config")

    # --- Run lifecycle ---
    p.add_argument("--num-steps", type=int, default=None, help="override TrainerConfig.num_steps")
    p.add_argument("--run-dir", type=str, default=None, help="override TrainerConfig.run_dir")
    p.add_argument("--seed", type=int, default=None, help="override TrainerConfig.seed")

    # --- Algorithm / loss ---
    p.add_argument("--algo", type=str, default=None,
                   help="override algo_name (grpo / dr_grpo / dapo / gspo / cispo); label only, dispatch is --loss/--adv")
    p.add_argument("--loss", type=str, default=None,
                   help="override rl.loss_type")
    p.add_argument("--adv", type=str, default=None,
                   help="override rl.adv_type (grpo / dr_grpo / rloo)")

    # --- Most-tweaked RL hyperparameters ---
    p.add_argument("--lr", type=float, default=None, help="override rl.lr")
    p.add_argument("--batch-size", type=int, default=None, help="override rl.batch_size")
    p.add_argument("--group-size", type=int, default=None, help="override rl.group_size")
    p.add_argument("--grad-accum-steps", type=int, default=None,
                   help="override rl.grad_accum_steps")
    p.add_argument("--optim-epochs", type=int, default=None, help="override rl.optim_epochs")
    p.add_argument("--kl-coef", type=float, default=None, help="override rl.kl_coef")
    p.add_argument("--grad-clip", type=float, default=None,
                   help="override rl.grad_clip (gradient norm cap)")
    p.add_argument("--max-new-tokens", type=int, default=None, help="override rl.max_new_tokens")
    p.add_argument("--max-prompt-tokens", type=int, default=None,
                   help="override max_prompt_tokens (cap on input length)")

    # --- Model / mode / scale ---
    p.add_argument("--model", type=str, default=None, help="override model_name (HF repo id)")
    p.add_argument("--env", type=str, default=None, help="override env_name")
    p.add_argument("--mode", choices=["disaggregated", "colocated"], default=None,
                   help="override mode")
    p.add_argument("--lora-rank", type=int, default=None, help="override lora_rank")
    p.add_argument("--gpu-mem-util", type=float, default=None,
                   help="override gpu_memory_utilization (vLLM fraction)")
    p.add_argument("--vllm-max-num-seqs", type=int, default=None,
                   help="override vllm_max_num_seqs (rollout concurrency cap)")

    # --- Weight sync ---
    p.add_argument("--weight-sync-method", choices=["disk", "nccl", "ipc"], default=None,
                   help="override TrainerConfig.weight_sync_method (disk | nccl | ipc)")
    p.add_argument("--weight-sync-disk-path", type=str, default=None,
                   help="override TrainerConfig.weight_sync_disk_path. Defaults to "
                        "/dev/shm/vivace_sync_<run_basename>; pass e.g. 'runs/<tag>/adapter' "
                        "to force NVMe.")

    # --- wandb ---
    p.add_argument("--wandb-project", type=str, default=None,
                   help="override TrainerConfig.wandb_project")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="override TrainerConfig.wandb_run_name (defaults to run_dir basename)")
    p.add_argument("--wandb-group", type=str, default=None,
                   help="override TrainerConfig.wandb_group (clusters runs in wandb UI)")

    # --- Generic escape hatch for any other field ---
    # Repeatable: --set rl.clip_cispo_high=10.0 --set rl.adam_beta2=0.95
    # Values are yaml-parsed so types are inferred (int, float, bool, str, list).
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   dest="overrides",
                   help="dotted-path override of any config field, e.g. "
                        "'--set lora_dropout=0.05 --set rl.adam_beta2=0.95'. Repeatable.")

    return p.parse_args(argv)


def _apply_set_overrides(cfg_dict: dict, overrides: list[str]) -> None:
    """Apply repeatable --set FIELD=VALUE overrides to a nested config dict.

    FIELD is a dotted path (e.g. 'rl.lr' or 'lora_rank'). VALUE is yaml-parsed,
    so '3e-5' → float, '8' → int, 'true' → bool, '[1,2]' → list. Missing nested
    keys raise KeyError so typos fail loud rather than silently no-op.
    """
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--set expects FIELD=VALUE, got: {spec!r}")
        key, raw = spec.split("=", 1)
        value = yaml.safe_load(raw)
        parts = key.split(".")
        target = cfg_dict
        for p in parts[:-1]:
            if p not in target or not isinstance(target[p], dict):
                raise KeyError(f"--set path {key!r}: '{p}' is not a dict in the config")
            target = target[p]
        target[parts[-1]] = value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg_dict = load_config(args.config)

    # CLI overrides win over YAML. Order: explicit shortcuts first, then the
    # generic --set escape hatch (so --set wins if both target the same field).
    # Top-level fields:
    for arg_name, cfg_key in [
        ("num_steps", "num_steps"),
        ("run_dir", "run_dir"),
        ("seed", "seed"),
        ("algo", "algo_name"),
        ("model", "model_name"),
        ("env", "env_name"),
        ("mode", "mode"),
        ("lora_rank", "lora_rank"),
        ("gpu_mem_util", "gpu_memory_utilization"),
        ("vllm_max_num_seqs", "vllm_max_num_seqs"),
        ("max_prompt_tokens", "max_prompt_tokens"),
        ("weight_sync_method", "weight_sync_method"),
        ("weight_sync_disk_path", "weight_sync_disk_path"),
        ("wandb_project", "wandb_project"),
        ("wandb_run_name", "wandb_run_name"),
        ("wandb_group", "wandb_group"),
    ]:
        val = getattr(args, arg_name)
        if val is not None:
            cfg_dict[cfg_key] = val

    # RL-block fields (apply under cfg_dict["rl"]):
    rl_block = cfg_dict.setdefault("rl", {})
    for arg_name, rl_key in [
        ("loss", "loss_type"),
        ("adv", "adv_type"),
        ("lr", "lr"),
        ("batch_size", "batch_size"),
        ("group_size", "group_size"),
        ("grad_accum_steps", "grad_accum_steps"),
        ("optim_epochs", "optim_epochs"),
        ("kl_coef", "kl_coef"),
        ("grad_clip", "grad_clip"),
        ("max_new_tokens", "max_new_tokens"),
    ]:
        val = getattr(args, arg_name)
        if val is not None:
            rl_block[rl_key] = val

    # Generic --set FIELD=VALUE applies last (escape hatch wins ties).
    _apply_set_overrides(cfg_dict, args.overrides)

    _maybe_enable_vllm_callable_rpc(cfg_dict)

    cfg = build_trainer_config(cfg_dict)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
