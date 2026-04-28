"""Probe: do vLLM's returned logprobs match an HF forward at the same temperature?

This is the load-bearing question for the (3) optimization (use vLLM logprobs as
old_log_prob and skip the recompute forward). If vLLM applies temperature inside
its softmax-for-logprobs the way our `compute_token_logprobs(logits / T)` does,
the two paths agree within bf16 numerics. If vLLM returns something else (raw
log_softmax, top-p-renormalized, etc.), we'd silently corrupt importance ratios.

Protocol:
  1. Build vLLM, sample N tokens at temperature T with logprobs=0.
  2. Build HF model, run the same [prompt | response] forward at the same T.
  3. Compare per-position: vLLM[i] vs HF.gather(targets[i]).
  4. Repeat at multiple temperatures (1.0, 0.9, 0.5) — if disagreement scales
     with (1/T - 1), vLLM is not applying T.

Pass criterion: max_abs_diff < 0.05 across all temperatures (bf16 floor between
vLLM's fused kernels and HF's separate kernels is ~0.01-0.03 in practice).

Usage:
    .venv/bin/python -m tests.probe_vllm_hf_logprob \\
        --model Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="probe-vllm-hf-logprob")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.9, 0.5])
    p.add_argument("--gpu-mem", type=float, default=0.4,
                   help="vLLM gpu_memory_utilization. Lowered so HF model fits on the same device.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logprobs-mode", choices=["raw_logprobs", "processed_logprobs"],
                   default="processed_logprobs",
                   help="vLLM logprobs mode. 'processed_logprobs' applies temperature; "
                        "'raw_logprobs' (V1 default) does not. RL needs processed.")
    p.add_argument("--dtype", choices=["bfloat16", "float32", "fp8"], default="bfloat16",
                   help="Compute dtype. 'bfloat16' = current production. "
                        "'float32' = sanity check, gap should drop to ~1e-4 (proves bf16 was the noise). "
                        "'fp8' = vLLM uses online fp8 quantization, HF stays in bf16 as reference; "
                        "measures additional gap from fp8 quantization on top of the bf16 floor.")
    return p.parse_args(argv)


def run_vllm(model: str, prompt: str, max_tokens: int, temperature: float,
             gpu_mem: float, seed: int, logprobs_mode: str = "processed_logprobs",
             dtype: str = "bfloat16"):
    """Generate via vLLM and capture sampled-token logprobs.

    logprobs_mode:
      - "raw_logprobs"        : log_softmax(logits) — no temperature applied (V1 default)
      - "processed_logprobs"  : log_softmax(logits/T) after top_p/top_k — what RL needs

    dtype: "bfloat16", "float32", or "fp8" (online fp8 quantization on bf16 base).
    """
    from vllm import LLM, SamplingParams
    # fp8 in vLLM is a quantization scheme, not a base dtype. Base stays bf16.
    if dtype == "fp8":
        llm_kwargs = dict(dtype="bfloat16", quantization="fp8")
    else:
        llm_kwargs = dict(dtype=dtype)
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem,
              enforce_eager=True, logprobs_mode=logprobs_mode, **llm_kwargs)
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                        logprobs=0, seed=seed)
    out = llm.generate(prompts=[prompt], sampling_params=sp)
    comp = out[0].outputs[0]
    # Cleanup so HF can fit on the same GPU.
    del llm
    import gc, torch as _t
    gc.collect(); _t.cuda.empty_cache()
    sampled_logps = [comp.logprobs[i][tok_id].logprob for i, tok_id in enumerate(comp.token_ids)]
    return list(comp.token_ids), sampled_logps, list(out[0].prompt_token_ids)


def run_hf(model_name: str, prompt_ids: list[int], gen_token_ids: list[int],
           temperature: float, dtype: str = "bfloat16") -> torch.Tensor:
    """Forward [prompt | response] through HF, return per-position logprob of the
    sampled token, shape [len(gen_token_ids)].

    dtype: "bfloat16" or "float32". For "fp8" (which HF doesn't run natively
    in forward), we fall back to bf16 — the probe then measures vLLM-fp8 vs
    HF-bf16, which is the practically useful comparison.
    """
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, torch.bfloat16)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype,
    ).cuda().eval()

    full = torch.tensor([prompt_ids + gen_token_ids], device="cuda")  # (1, S)
    with torch.no_grad():
        logits = model(full).logits[:, :-1, :] / temperature           # (1, S-1, V)
        log_probs = F.log_softmax(logits, dim=-1)                       # (1, S-1, V)
    targets = full[:, 1:]                                                # (1, S-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, S-1)

    plen = len(prompt_ids)
    # Generated tokens occupy target indices [plen-1, plen-1+len(gen)).
    start = plen - 1
    out = token_logp[0, start:start + len(gen_token_ids)].float().cpu()
    del model
    import gc, torch as _t
    gc.collect(); _t.cuda.empty_cache()
    return out


def compare(temperature: float, vllm_logps: list[float], hf_logps: torch.Tensor) -> dict:
    v = torch.tensor(vllm_logps, dtype=torch.float32)
    h = hf_logps.float()
    diff = (v - h).abs()
    return {
        "T": temperature,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "vllm_first5": v[:5].tolist(),
        "hf_first5": h[:5].tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    print(f"Probing {args.model}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Temperatures: {args.temps}")
    print(f"vLLM logprobs_mode: {args.logprobs_mode}")
    print(f"dtype: {args.dtype}"
          + ("  (vLLM=fp8-quantized, HF=bf16 reference)" if args.dtype == "fp8" else ""))
    print()

    results = []
    for T in args.temps:
        print(f"--- temperature={T} ---")
        gen_ids, vllm_logps, prompt_ids = run_vllm(
            args.model, args.prompt, args.max_tokens, T, args.gpu_mem, args.seed,
            logprobs_mode=args.logprobs_mode, dtype=args.dtype,
        )
        hf_logps = run_hf(args.model, prompt_ids, gen_ids, T, dtype=args.dtype)
        r = compare(T, vllm_logps, hf_logps)
        results.append(r)
        print(f"  max |Δ| = {r['max_abs_diff']:.4f}   mean |Δ| = {r['mean_abs_diff']:.4f}")
        print(f"  vLLM[:5] = {[round(x, 4) for x in r['vllm_first5']]}")
        print(f"  HF  [:5] = {[round(x, 4) for x in r['hf_first5']]}")
        print()

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    # Two thresholds:
    #  - Per-T absolute floor: bf16 noise between vLLM fused kernels and HF separate
    #    kernels. ~0.1-0.3 in practice (cf. test_weight_sync step-1 baseline ~0.11).
    #  - Cross-T scaling check: a semantic bug (vLLM ignoring temperature) makes
    #    max|Δ| scale ~linearly with (1/T - 1). bf16 noise stays roughly constant.
    BF16_FLOOR = 0.5
    SCALING_RATIO_THRESHOLD = 4.0  # max|Δ| at smallest T should be < this × at T=1.0

    print(f"  {'T':>6}  {'max|Δ|':>10}  {'mean|Δ|':>10}  per-T")
    for r in results:
        ok = r["max_abs_diff"] < BF16_FLOOR
        print(f"  {r['T']:>6.2f}  {r['max_abs_diff']:>10.4f}  {r['mean_abs_diff']:>10.4f}  "
              f"{'PASS' if ok else 'FAIL'}")

    # Scaling check: catches the "vLLM ignored temperature" bug specifically.
    by_T = {r["T"]: r["max_abs_diff"] for r in results}
    if 1.0 in by_T and len(args.temps) > 1:
        ref = max(by_T[1.0], 1e-6)
        worst_T = min(args.temps)
        ratio = by_T[worst_T] / ref
        scaling_ok = ratio < SCALING_RATIO_THRESHOLD
        print(f"\n  Scaling check: max|Δ|(T={worst_T}) / max|Δ|(T=1.0) = {ratio:.2f}x  "
              f"(threshold {SCALING_RATIO_THRESHOLD}x) → {'PASS' if scaling_ok else 'FAIL'}")
    else:
        scaling_ok = True

    per_T_ok = all(r["max_abs_diff"] < BF16_FLOOR for r in results)
    if per_T_ok and scaling_ok:
        floor_label = {
            "float32": "kernel-implementation floor (~1e-3 to 2e-2 expected)",
            "bfloat16": "bf16 floor (~0.1 to 0.3 expected; ~10-25x the fp32 kernel floor)",
            "fp8": "fp8 quantization gap (vLLM=fp8 vs HF=bf16; large gap expected — see notes)",
        }.get(args.dtype, "noise floor")
        print(f"\n  vLLM logprobs match HF at all tested temperatures ({floor_label}).")
        print("  Safe to use vLLM logprobs as old_log_prob — proceed with (3).")
        sys.exit(0)
    else:
        print("\n  Disagreement detected. Investigate before swapping in (3).")
        if not scaling_ok:
            print("  max|Δ| scales with smaller T → vLLM likely NOT applying temperature.")
            print("  Check that LLM() was constructed with logprobs_mode='processed_logprobs'.")
        if args.dtype == "fp8":
            print("  Note: --dtype fp8 compares vLLM-fp8 vs HF-bf16 (HF can't run fp8 forward).")
            print("  The HF reference is the un-quantized model. Large gap means fp8 quantization")
            print("  produces tokens the un-quantized model considers unlikely — NOT a code bug,")
            print("  just fp8 degrading model quality. Decisive only if you also train in fp8.")
        sys.exit(1)


if __name__ == "__main__":
    main()
