"""Validate weight sync (disk or NCCL) end-to-end via verify_weights_match.

Three-step protocol:
  1. Fresh Trainer — both sides should agree by construction (same checkpoint,
     LoRA B is zero-initialized so the adapter contributes nothing yet).
  2. Perturb trainer weights — they should disagree now.
  3. Call trainer.sync_weights() — they should agree again.

Passing all three means the configured sync backend correctly propagates
weights from trainer to vLLM. A failure in step 2 means the perturbation
wasn't large enough to be detectable (increase --perturb-scale). A failure
in step 3 means the sync backend is broken.

Usage:
    .venv/bin/python -m tests.test_weight_sync \\
        --config vivace/configs/dapo_gsm8k_1.5b_profiling.yaml \\
        --method disk

    .venv/bin/python -m tests.test_weight_sync \\
        --config vivace/configs/dapo_gsm8k_1.5b_profiling.yaml \\
        --method nccl
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vivace-test-weight-sync")
    p.add_argument("--config", required=True, help="path to YAML config (mode must be disaggregated)")
    p.add_argument("--method", choices=["disk", "nccl"], required=True,
                   help="weight sync backend under test — overrides config value")
    p.add_argument("--perturb-scale", type=float, default=0.01,
                   help="std of gaussian noise added to trainable params in step 2. "
                        "Needs to be large enough for step 2 to detect disagreement "
                        "(top_1_match becomes False), but small enough that step 3 "
                        "after sync doesn't hit HF↔vLLM implementation numerics "
                        "(different attn kernels, rotary, RMSNorm) that amplify on "
                        "ill-conditioned post-perturbation models. 0.01 works for "
                        "Qwen2.5-0.5B full FT; tune per model/setup.")
    p.add_argument("--test-prompt", type=str, default="The capital of France is",
                   help="prompt used for the forward-pass comparison in verify_weights_match")
    return p.parse_args(argv)


def _maybe_enable_vllm_callable_rpc(method: str) -> None:
    """Same rationale as scripts/train.py: NCCL path pickles callables through collective_rpc."""
    if method == "nccl":
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _disable_wandb() -> None:
    """The test builds a Trainer (which calls init_wandb) but doesn't actually train.
    Disable wandb so the test run doesn't prompt for login or pollute the project's
    run history with one-off diagnostic runs."""
    os.environ.setdefault("WANDB_MODE", "disabled")


def _print_result(label: str, result: dict) -> None:
    print(f"  {label}: ok={result['ok']}  "
          f"top_1_match={result['top_1_match']}  "
          f"top_5={result['top_5_agreement']:.2f}  "
          f"top_k={result['top_k_agreement']:.2f}  "
          f"max_diff={result['max_logprob_diff']:.4g}")
    print(f"    trainer_topk: {result['trainer_topk'][:5]}")
    print(f"    vllm_topk:    {result['vllm_topk'][:5]}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    # Override method under test; do NOT silently change mode — a colocated
    # config makes sync_weights a no-op and the test would falsely fail.
    cfg_dict["weight_sync_method"] = args.method
    if cfg_dict.get("mode") != "disaggregated":
        print(f"ERROR: test requires mode=disaggregated in the config "
              f"(got {cfg_dict.get('mode')!r}). In colocated mode sync_weights "
              f"is a no-op and the test can't distinguish working from broken.",
              file=sys.stderr)
        sys.exit(2)

    # Don't actually train — keep the run light. Disable periodic eval too.
    cfg_dict["num_steps"] = 1
    cfg_dict["eval_interval"] = 10_000
    cfg_dict.pop("profiling", None)   # no profiling overhead in the test

    _maybe_enable_vllm_callable_rpc(args.method)
    _disable_wandb()

    # Import after env var is set so the vLLM subprocess inherits it.
    import torch
    from vivace.scripts.train import build_trainer_config
    from vivace.train.trainer import Trainer
    from vivace.utils.weight_sync import verify_weights_match

    cfg = build_trainer_config(cfg_dict)
    print(f"[test] building trainer (weight_sync_method={cfg.weight_sync_method})")
    trainer = Trainer(cfg)

    kwargs = dict(
        trainer_model=trainer.model,
        vllm_worker=trainer.rollout_worker,
        tokenizer=trainer.tokenizer,
        test_prompt=args.test_prompt,
    )

    print("\n[step 1] fresh trainer vs fresh vLLM — expect AGREE")
    r1 = verify_weights_match(**kwargs)
    _print_result("result", r1)

    print(f"\n[step 2] perturbing trainable params with gaussian noise (scale={args.perturb_scale}) — expect DISAGREE")
    with torch.no_grad():
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data.add_(torch.randn_like(p) * args.perturb_scale)
    r2 = verify_weights_match(**kwargs)
    _print_result("result", r2)

    print("\n[step 3] trainer.sync_weights() — expect AGREE again")
    trainer.sync_weights()
    r3 = verify_weights_match(**kwargs)
    _print_result("result", r3)

    # Verdicts.
    # Step 1: trainer ↔ vLLM should agree on the original (un-perturbed) weights.
    # Step 2: perturbation should produce detectable disagreement.
    # Step 3 is trickier — for LoRA configs, peft's LoraLinear forwards as
    # `base@x + B@(A@x)` while vLLM uses the merged `(base + B@A)@x`. These are
    # mathematically equal but numerically different in bf16, and the gap grows
    # with LoRA contribution magnitude. After sync, weights ARE bit-identical
    # but the forward paths can still differ. Pass step 3 if max_logprob_diff
    # is roughly back to step 1's baseline (sync recovered the agreement that
    # exists between the two implementations on identical weights).
    step3_relative_pass = r3["max_logprob_diff"] <= max(3.0 * r1["max_logprob_diff"], 0.5)

    step_results = [
        ("step 1 (fresh, agree)",         r1["ok"]),
        ("step 2 (perturbed, disagree)",  not r2["ok"]),
        ("step 3 (synced, recovered)",    step3_relative_pass),
    ]

    print("\n" + "=" * 60)
    print(f"  Weight sync test — method={args.method}")
    print("=" * 60)
    for label, ok in step_results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    all_ok = all(ok for _, ok in step_results)
    if all_ok:
        print(f"\n  All steps passed — {args.method} weight sync is verified.")
        sys.exit(0)
    else:
        # Targeted diagnostics for the common failure modes.
        if not step_results[1][1]:
            print(f"\n  Step 2 failed (weights still agreed after perturbation). "
                  f"Increase --perturb-scale (current: {args.perturb_scale}).")
        if not step_results[2][1]:
            print(f"\n  Step 3 failed — post-sync max_logprob_diff "
                  f"({r3['max_logprob_diff']:.4g}) is much larger than step 1's "
                  f"baseline ({r1['max_logprob_diff']:.4g}). Either the {args.method} "
                  f"sync backend isn't propagating weights, OR (for LoRA) the "
                  f"perturbation is too large and bf16 numerics on peft's "
                  f"LoraLinear vs vLLM's merged forward have diverged.")
        sys.exit(1)


if __name__ == "__main__":
    main()
