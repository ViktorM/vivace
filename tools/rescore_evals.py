"""Offline re-score of saved eval sample dumps with the current matchers.

Recovers corrected accuracies from eval_samples_final_{gsm8k,math500}.json
without re-running evals — used to fix the pre-9f3027e math500 undercount.

Usage: .venv/bin/python tools/rescore_evals.py 'runs/v1_1.5b/*/'
"""
import glob
import json
import sys

from datasets import load_dataset

from vivace.rewards import _math_correct, answer_match


def ext(t):
    if "<answer>" not in t:
        return ""
    return t.split("<answer>")[-1].split("</answer>")[0].strip()


pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/v1_1.5b/*/"

gsm = load_dataset("openai/gsm8k", "main")["test"]
gsm_gt = {r["question"]: r["answer"].split("####")[-1].strip().replace(",", "") for r in gsm}
m500 = load_dataset("HuggingFaceH4/MATH-500")["test"]
m500_gt = {r["problem"]: str(r["answer"]) for r in m500}

runs = sorted(glob.glob(pattern))
print(f"{'run':55s} {'gsm_old':>8s} {'gsm_new':>8s} {'m500_old':>9s} {'m500_new':>9s}")
for rd in runs:
    name = rd.rstrip("/").split("/")[-1]
    row = [name]
    try:
        d = json.load(open(rd + "eval_samples_final_gsm8k.json"))
        n = len(d["correct"]) + len(d["incorrect"])
        old = len(d["correct"])
        new = old
        for s in d["incorrect"]:
            gt = gsm_gt.get(s["question"])
            if gt and answer_match(gt, ext(s["response"])):
                new += 1
        # correct->incorrect flips can't happen: the matcher only loosens
        row += [f"{100*old/n:7.2f}", f"{100*new/n:7.2f}"]
    except FileNotFoundError:
        row += ["-", "-"]
    try:
        d = json.load(open(rd + "eval_samples_final_math500.json"))
        n = len(d["correct"]) + len(d["incorrect"])
        old = len(d["correct"])
        new = 0
        for s in d["correct"] + d["incorrect"]:
            gt = m500_gt.get(s["question"])
            pred = ext(s["response"])
            if gt and pred and _math_correct(gt, pred):
                new += 1
        row += [f"{100*old/n:8.2f}", f"{100*new/n:8.2f}"]
    except FileNotFoundError:
        row += ["-", "-"]
    print(" ".join(f"{c:>9s}" if i else f"{c:55s}" for i, c in enumerate(row)), flush=True)
