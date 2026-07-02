"""Re-evaluate saved LoRA checkpoints at different eval token budgets.

Motivation: Math-trained models write long solutions; the gsm8k eval budget of
256 truncated 35% of responses in the math sweep. This loads the base model
into vLLM ONCE (enable_lora) and sweeps adapters x budgets to pick a budget
for the config update — no retraining, no trainer involved.

Usage (after the sweep frees the GPUs):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/reeval_budget.py \
      --runs 'runs/math3seed/*_1616' --env gsm8k --budgets 512 1024
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from vivace.envs import make_env
from vivace.rewards import answer_match

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def final_answer(resp: str) -> str:
    """Boxed-first extraction: Math-trained models put the whole solution
    inside <answer> and finish with \\boxed{N}, so prefer the last boxed
    content; fall back to the gsm8k tag convention."""
    boxed = BOXED_RE.findall(resp)
    if boxed:
        return boxed[-1]
    if "<answer>" in resp:
        return resp.split("<answer>")[-1].split("</answer>")[0].strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="glob of run dirs containing final_model/")
    ap.add_argument("--env", default="gsm8k")
    ap.add_argument("--budgets", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n", type=int, default=-1, help="-1 = full eval split")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out", default="/tmp/reeval_budget.json")
    args = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(args.runs) if os.path.isdir(os.path.join(d, "final_model")))
    if not run_dirs:
        raise SystemExit(f"no run dirs with final_model/ match {args.runs}")
    print(f"{len(run_dirs)} checkpoints x {args.budgets} budgets on {args.env}")

    env = make_env(args.env)
    examples = env.load_split("eval")
    if args.n != -1:
        examples = examples[: args.n]
    prompts = [env.format_prompt(ex) for ex in examples]

    llm = LLM(model=args.model, dtype="bfloat16", enable_lora=True,
              max_lora_rank=16, max_model_len=1536,
              gpu_memory_utilization=args.gpu_mem_util)

    results = []
    for i, rd in enumerate(run_dirs):
        name = os.path.basename(rd.rstrip("/"))
        lora = LoRARequest(name, i + 1, os.path.join(rd, "final_model"))
        for budget in args.budgets:
            sp = SamplingParams(temperature=0.0, max_tokens=budget)
            t0 = time.time()
            outs = llm.generate(prompts, sp, lora_request=lora)
            responses = [o.outputs[0].text for o in outs]
            flags = env.is_correct_batch(responses, examples)
            flags_boxed = [answer_match(ex.answer, final_answer(r))
                           for r, ex in zip(responses, examples)]
            ncap = sum(len(o.outputs[0].token_ids) >= budget for o in outs)
            acc = 100 * sum(flags) / len(flags)
            row = dict(run=name, budget=budget, acc=round(acc, 2),
                       acc_boxed=round(100 * sum(flags_boxed) / len(flags_boxed), 2),
                       cap_rate=round(100 * ncap / len(outs), 2),
                       eval_s=round(time.time() - t0, 1))
            results.append(row)
            print(row, flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
