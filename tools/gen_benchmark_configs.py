"""Generate the v1 benchmark configs (2x4090 colo, DDP).

Two families:
  gsm8k  — full 5-algo comparison, max_new=192, eval {gsm8k, math500}
  math   — 5 algos, max_new=768, eval {gsm8k, math500, aime24/25/26}
           (AIME eval at 4096 via per-env eval_max_new_tokens; vllm_max_model_len
           sized to fit the longest eval budget)

Per-algo (lr, optim_epochs, kl) carried from the 0.5B pilot picks; these are the
starting points to babysit at 1.5B+ (re-tune if a run collapses early).

Run: .venv/bin/python tools/gen_benchmark_configs.py [--scale 1.5b|3b]
Writes to vivace/configs/experiments/v1_<scale>/.
"""
from __future__ import annotations

import argparse
import os

# Per-scale memory knobs for 24GB colo. The model lives twice on each GPU while
# vLLM is awake (trainer copy + vLLM copy), so the vLLM pool shrinks with scale.
SCALES = {
    "1.5b": dict(
        model="Qwen/Qwen2.5-1.5B",
        mem_util="0.65   # vLLM sleeps during training; 0.65 (not 0.7) leaves ~1GB margin for a desktop session on one GPU — 1.5B DDP-colo at 0.7 OOMs in merge_adapter when the display eats 2.4GB",
        gsm_seqs="256   # Qwen2.5 KV is small; the old 32 cap serialized eval + rollout generation",
        math_seqs="64    # ctx 4608 -> ~129MB KV per seq; 64 fits the 0.65 pool, vs 8 which ran rollouts in 8 serial waves",
    ),
    "3b": dict(
        model="Qwen/Qwen2.5-3B",
        mem_util="0.42   # 3B weights (5.8GB) sit in both the trainer and vLLM while awake; 0.5 OOMed at wake_up on the GPU that also holds the ~4GB desktop session",
        gsm_seqs="64    # ~36KB KV/token at 3B; 64 seqs x 1536 ctx fits the ~4.5GB left in the 0.42 pool",
        math_seqs="24    # ctx 4608 -> ~166MB KV per seq at 3B; 24 fits the 0.42 pool",
    ),
}

# Per-algo rl-block fragments. lr/ep from 0.5B pilot bests; GRPO is the
# negative baseline (its least-bad cell).
ALGOS = {
    "grpo":    dict(loss_type="grpo",    adv_type="grpo", lr_gsm=8.0e-5, ep_gsm=2,
                    lr_math=3.0e-5, ep_math=2, kl=0.04, extra={}),
    "dr_grpo": dict(loss_type="dr_grpo", adv_type="rloo", lr_gsm=8.0e-5, ep_gsm=2,
                    lr_math=3.0e-5, ep_math=2, kl=0.0, extra={}),
    "gspo":    dict(loss_type="gspo",    adv_type="rloo", lr_gsm=1.0e-4, ep_gsm=2,
                    lr_math=4.0e-5, ep_math=2, kl=0.01,
                    extra={"clip_low": 0.2, "clip_high": 0.2}),
    # ep was 1 (0.5B pilot pick); flipped to 2 after the v2 ep-A/B: gsm8k flat,
    # math500 +2.1pp on all 3 paired seeds, and budget-matches the other algos.
    "dapo":    dict(loss_type="dapo",    adv_type="rloo", lr_gsm=8.0e-5, ep_gsm=2,
                    lr_math=3.0e-5, ep_math=2, kl=0.01,
                    extra={"clip_low": 0.2, "clip_high": 0.28}),
    "cispo":   dict(loss_type="cispo",   adv_type="rloo", lr_gsm=6.0e-5, ep_gsm=2,
                    lr_math=3.0e-5, ep_math=2, kl=0.01,
                    extra={"adam_beta2": 0.95, "adam_eps": 1.0e-15,
                           "cispo_normalization": "hybrid",
                           "clip_cispo_high": 5.0, "clip_cispo_low": 0.0}),
}

# gspo added after the gsm8k v2 sweep crowned it (72.2±0.4); grpo added since
# it ranked 3rd in v2 and is the reference algo. The original top-3 pick
# predated the eval-verifier fix.
MATH_ALGOS = ["dapo", "cispo", "dr_grpo", "gspo", "grpo"]


def rl_block(a: dict, lr: float, ep: int, max_new: int) -> str:
    lines = [
        f"  loss_type: {a['loss_type']}",
        f"  adv_type: {a['adv_type']}",
        "  batch_size: 1",
        "  group_size: 8",
        f"  lr: {lr:.1e}",
        "  warmup_steps: 10",
        "  grad_accum_steps: 4",
        f"  optim_epochs: {ep}",
        "  eta_min_ratio: 0.2",
        "  lr_restart: false",
        "  grad_clip: 1.0",
        f"  max_new_tokens: {max_new}",
        "  temperature: 0.7",
        "  top_p: 0.95",
        "  top_k: -1",
        f"  kl_coef: {a['kl']}",
        "  adv_eps: 0.0001",
        "  entropy_chunk_size: 64",
        "  entropy_grad: false",
        "  adaptive_sampling: true",
        "  oversample_factor: 2.0",
        "  min_reward_spread: 0.5",
    ]
    for k, v in a["extra"].items():
        # YAML needs a decimal point in scientific notation (1.0e-15, not 1e-15)
        # or PyYAML parses it as a STRING. str(1.0e-15) == "1e-15" — the trap.
        if isinstance(v, float) and ("e" in repr(v) or "E" in repr(v)):
            lines.append(f"  {k}: {v:.1e}")   # -> "1.0e-15"
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def gsm8k_config(name: str, a: dict, scale: str) -> str:
    s = SCALES[scale]
    return f"""# v1 benchmark, {scale.upper()} / gsm8k — {name.upper()} (full 5-algo comparison).
# max_new=192 (gsm8k responses never approach the cap); eval on gsm8k + math500.
model_name: {s['model']}
env_name: gsm8k
algo_name: {name}
mode: colocated
trainer_gpus: [0, 1]
rollout_gpus: [0, 1]
use_vllm: true
gpu_memory_utilization: {s['mem_util']}
enforce_eager: false
compile_model: false
vllm_max_model_len: 1536
vllm_max_num_seqs: {s['gsm_seqs']}
use_lora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
num_steps: 200
sft_warmup: false
sft_warmup_steps: 0
wandb_project: vivace
log_interval: 1
eval_interval: 50
eval_n: -1
eval_batch_size: 32
eval_use_vllm: true
checkpoint_interval: 200
run_dir: runs/v1_{scale}/{name}_{scale}_gsm8k
weight_sync_method: ipc
seed: 42
eval_envs: [gsm8k, math500]
eval_max_new_tokens: [256, 1024]
rl:
{rl_block(a, a['lr_gsm'], a['ep_gsm'], 192)}
gradient_checkpointing: true
"""


def math_config(name: str, a: dict, scale: str) -> str:
    s = SCALES[scale]
    return f"""# v1 benchmark, {scale.upper()} / math — {name.upper()} (cluster preview + AIME generalization).
# Train on Hendrycks MATH (max_new=768, max_prompt=512). Eval on gsm8k + math500
# + AIME 24/25/26. AIME eval at 4096 via per-env eval_max_new_tokens; the vLLM
# engine is sized (vllm_max_model_len=4608) to fit prompt(512) + AIME gen(4096).
# gsm8k eval at 1024, not 256: Math-trained models write long solutions and 33-62%
# of responses were capped before the answer at 256 (v1_results.md, Math finding 3).
model_name: {s['model']}
env_name: math
env_kwargs:
  corpus: hendrycks
algo_name: {name}
mode: colocated
trainer_gpus: [0, 1]
rollout_gpus: [0, 1]
use_vllm: true
# vLLM sleeps (weights + KV released) during the train phase, so the trainer gets
# the whole card for the 1280-token backward.
gpu_memory_utilization: {s['mem_util']}
enforce_eager: false
compile_model: false
vllm_max_model_len: 4608
vllm_max_num_seqs: {s['math_seqs']}
use_lora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
num_steps: 200
sft_warmup: false
sft_warmup_steps: 0
wandb_project: vivace
log_interval: 1
eval_interval: 50
eval_n: -1
eval_batch_size: 8
eval_use_vllm: true
max_prompt_tokens: 512
checkpoint_interval: 200
run_dir: runs/v1_{scale}/{name}_{scale}_math
weight_sync_method: ipc
seed: 42
eval_envs: [gsm8k, math500, aime24, aime25, aime26]
eval_max_new_tokens: [1024, 1024, 4096, 4096, 4096]
rl:
{rl_block(a, a["lr_math"], a["ep_math"], 768)}
gradient_checkpointing: true
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=sorted(SCALES), default="1.5b")
    scale = ap.parse_args().scale
    out_dir = f"vivace/configs/experiments/v1_{scale}"
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, a in ALGOS.items():
        p = os.path.join(out_dir, f"{name}_{scale}_gsm8k.yaml")
        with open(p, "w") as f:
            f.write(gsm8k_config(name, a, scale))
        written.append(p)
    for name in MATH_ALGOS:
        p = os.path.join(out_dir, f"{name}_{scale}_math.yaml")
        with open(p, "w") as f:
            f.write(math_config(name, ALGOS[name], scale))
        written.append(p)
    print(f"Wrote {len(written)} configs to {out_dir}/:")
    for p in written:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
