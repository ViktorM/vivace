"""Prefix-cache staleness across a weight sync (GPU, disaggregated + nccl).

vLLM hashes cached KV blocks by tokens and LoRA id, never by weights. Without a
reset after the sync, a request whose prefix is already cached reads K/V computed
by the previous policy. The probe measures the same engine, same weights, with and
without cached blocks, so the difference is the stale-block effect alone.

  1. warm: generate on a prompt with >= 2 full 16-token blocks of shared prefix
  2. perturb the LoRA params, sync (the sync now resets the cache)
  3. next-token logprobs on the same prompt  -> `after_sync`
  4. explicit reset_prefix_cache, same prompt -> `fresh`
  expect after_sync == fresh.  With --show-bug the reset inside the sync is
  bypassed once first, to show what the difference looks like when blocks are stale.

Usage:
    .venv/bin/python tests/probe_prefix_cache_sync.py --config <disaggregated nccl yaml> [--show-bug]
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

PREFIX = ("You are a careful assistant. Respond in the following format: first think step by step "
          "inside <think> tags, then give the final answer inside <answer> tags. Keep the reasoning "
          "short and check the arithmetic twice before answering.\n")
QUESTION = "Problem: A shop sells 12 pens for 30 dollars. How much do 7 pens cost?\n"


def _next_token_logprobs(worker, ids, topk=20) -> dict[int, float]:
    outputs, _ = worker.generate(prompt_token_ids=[ids], temperature=0.0, top_k=1, max_tokens=1, n=1, logprobs=topk)
    return {t: lp.logprob for t, lp in outputs[0].outputs[0].logprobs[0].items()}


def _vllm_param_fingerprint(worker_self, name):
    """Runs inside the vLLM worker: (fp32 sum, first 8 values) of one named parameter."""
    mr = worker_self.model_runner
    m = mr.get_model() if hasattr(mr, "get_model") else mr.model
    p = dict(m.named_parameters())[name].detach()
    return p.float().sum().item(), p.flatten()[:8].float().tolist()


def _max_diff(a: dict[int, float], b: dict[int, float]) -> float:
    common = set(a) & set(b)
    return max(abs(a[t] - b[t]) for t in common) if common else float("inf")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--perturb-scale", type=float, default=0.05)
    p.add_argument("--show-bug", action="store_true", help="first run one sync with the reset bypassed")
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    assert cfg_dict.get("mode") == "disaggregated", "probe needs a disaggregated config (engine never sleeps)"
    cfg_dict["weight_sync_method"] = "nccl"
    cfg_dict["num_steps"] = 1
    cfg_dict["eval_interval"] = 10_000
    cfg_dict.pop("profiling", None)
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")

    import torch
    from vivace.scripts.train import build_trainer_config
    from vivace.train.trainer import Trainer

    trainer = Trainer(build_trainer_config(cfg_dict))
    worker = trainer.rollout_worker
    ids = trainer.tokenizer(PREFIX + QUESTION).input_ids
    print(f"[probe] prompt is {len(ids)} tokens = {len(ids) // 16} full cached blocks")

    def perturb():
        with torch.no_grad():
            for prm in trainer.model.parameters():
                if prm.requires_grad:
                    prm.data.add_(torch.randn_like(prm) * args.perturb_scale)

    def base_fingerprint():
        return [p.detach().float().sum().item() for n, p in trainer.model.named_parameters() if n.endswith("base_layer.weight")]

    base0 = base_fingerprint()
    results = {}
    modes = (["stale (reset bypassed)"] if args.show_bug else []) + ["fixed"]
    for mode in modes:
        _next_token_logprobs(worker, ids)                 # 1. warm the prefix blocks under the current weights
        perturb()                                         # 2. move the policy
        if mode.startswith("stale"):
            real_reset = worker.llm.reset_prefix_cache
            worker.llm.reset_prefix_cache = lambda *a, **k: True
            try:
                trainer.sync_weights()
            finally:
                worker.llm.reset_prefix_cache = real_reset
        else:
            trainer.sync_weights()                        # includes the reset
        after_sync = _next_token_logprobs(worker, ids)    # 3. may read cached blocks
        worker.llm.reset_prefix_cache()
        fresh = _next_token_logprobs(worker, ids)         # 4. everything recomputed under the synced weights
        results[mode] = _max_diff(after_sync, fresh)
        print(f"[probe] {mode:24s} max |logprob(after sync) - logprob(fresh)| = {results[mode]:.3e}")

    # Weight-level check: the engine must hold exactly the trainer's merged buffers
    # (one non-fused target and one fused qkv buffer, layer 0).
    st = trainer._nccl_sync_state
    tr = {"model.layers.0.self_attn.o_proj.weight": st["merged"]["model.layers.0.self_attn.o_proj.weight"],
          "model.layers.0.self_attn.qkv_proj.weight": st["fused_buffers"]["model.layers.0.self_attn.qkv_proj.weight"]}
    weights_ok = True
    for name, t in tr.items():
        vs, vhead = worker.llm.collective_rpc(_vllm_param_fingerprint, args=(name,))[0]
        ts, thead = t.float().sum().item(), t.flatten()[:8].float().tolist()
        same = vs == ts and vhead == thead
        weights_ok &= same
        print(f"[probe] engine == merged buffer for {name}: {same}  (sum trainer {ts:.6f} / engine {vs:.6f})")

    base_ok = base_fingerprint() == base0
    print(f"[probe] frozen base {'unchanged' if base_ok else 'CHANGED'} across {len(modes)} syncs (merged-buffer sync)")
    ok = results["fixed"] < 1e-3 and base_ok and weights_ok
    print(f"[probe] {'PASS' if ok else 'FAIL'}: after the fix the synced engine must not depend on cached blocks"
          + (f"; stale-block effect for reference: {results['stale (reset bypassed)']:.3e}" if args.show_bug else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
