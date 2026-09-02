# Experiment plan, September 2026

Three tiers, one software stack, one reward definition. Every number that reaches the
README comes from the same vLLM/torch stack and the same reward code, with 3 seeds.

| tier | hardware | role | cost |
|---|---|---|---|
| 1 | 2×4090 (desk) | re-run the v1 tables on the new stack; cheap decision experiments | free |
| 2 | 2×RTX PRO 6000 (96 GB each, Runpod) | axes the 4090 can't reach: batch, group size, sequence length, 7B | ~$1.5–2 / GPU-h |
| 3 | 1-2-4-8 × H100 SXM (H200 variant) | scaling curves, headline 7B cells, one best-perf record run | per `cluster_support.md` |

Companion docs: [`v1_benchmark_plan.md`](v1_benchmark_plan.md) (targets, base pass@K map,
eval protocol — still valid), [`v1_results.md`](v1_results.md) (what exists),
`reviews/2026-09-01-deep-review.md` (the findings referenced below).

## 0. Gates — decide and land before any re-run

Re-running the tables twice is the expensive mistake; everything below is a one-day
punch list that changes what the runs measure.

1. **Stack.** vLLM 0.28 / torch 2.13 / transformers 5.16 / peft 0.20 (`uv.lock`, Dockerfile).
   Validate on GPU before the first re-run: `tools/vllm_upgrade_smoke.py`, `tests/test_weight_sync.py
   --method disk|nccl`, a 20-step colocated run, a 2-rank DDP run, `tests/probe_vllm_hf_logprob.py`
   (logprobs_mode still "processed"), and one sleep/wake/sync/generate cycle with
   `verify_weights_match`. peft 0.20 may change the bf16 merge path — re-check the drift probe.
2. **Reward definition, frozen.** The xmlcount trailing-junk penalty is unbounded on capped
   responses (−8 on a capped MATH answer vs +3 correct; live in every MATH run) —
   cap it at one tag's credit and stop charging when the closing tag is missing.
   `strict_format_reward` has never fired (needs `</answer>\n`, EOS strips it): delete it
   or fix the regex — the 1.0 "format plateau" was 0.75. `soft_format` uses `re.match`
   (anchored) while its docstring says whitespace-tolerant: `re.search`. All three move
   the reward scale, so they land before tier 1, not after.
3. **Eval.** gsm8k `is_correct` boxed-first (Math-trained models write `\boxed{}` inside
   `<answer>`; the tag+float matcher scores them 0 — the in-run gsm8k transfer numbers are
   artifacts). AIME at 7B: sampled avg@16 at T=0.7 rather than greedy (30 problems quantize
   greedy at 3.3 pp). Pin dataset `revision=` on all six `load_dataset` calls.
4. **Reproducibility.** `run_manifest.json` (resolved config, commit + dirty flag, package and
   CUDA versions, dataset revisions, seeds, launch command); `aggregate_seeds.py` excludes
   unfinished runs and prints per-cell seed lists.
5. **Recipes that change cells.** DAPO `clip_high` 0.28 in every DAPO config (five ship
   0.2, including the README quickstart). Temperature: 0.7 is a 0.5B-pilot artifact —
   A/B at 1.5B first (tier 1, item A1) and regenerate the 3B/7B configs from the result.
   `rloo` vs `dr_grpo` differ by the constant G/(G−1): keep one estimator, relabel.
6. **Perf prerequisites for 7B (tier 3).** Chunked `target_logit − logsumexp` instead of the
   full-vocab `log_softmax` (the [B,S,V] tensor is ~4.6 GiB bf16 at B=16, S=1024) — it is
   what allows a larger micro-batch and gradient checkpointing off on 80 GB. `eval_gpus`
   dedicated eval worker for 8-GPU runs. Behavior-policy gap measured once (vLLM sampled
   logprobs vs HF epoch-1 logprobs; token IS-weight p99) to decide whether TIS is needed.

## 1. Tier 1 — 2×4090 (free)

Colocated DDP, LoRA r=16, 200 steps, protocol of `v1_results.md`. Per-run wall-clock from
v2: gsm8k 1.5B 17 min, MATH 1.5B 53 min, gsm8k 3B 35 min, MATH 3B ≈1.8 h (needs the
desktop GPU clear; 0.42 pool).

| id | cell | runs | wall-clock | why |
|---|---|---|---|---|
| A1 | DAPO 1.5B/gsm8k, T ∈ {0.9, 1.0} × 3 seeds, paired with the existing T=0.7 | 6 | 1.7 h | fixes T for every later cell; RLConfig default (0.9) disagrees with every shipped config |
| A2 | 1.5B/gsm8k, 5 algos × 3 seeds | 15 | 4.3 h | the reference table, on the new stack and reward |
| A3 | 3B/gsm8k, 5 algos × 3 seeds | 15 | 9 h | plan target #2; the DAPO cell exists (81.4 ± 0.8) but on the old stack |
| A4 | GRPO-on-MATH isolating arms at 1.5B: binary reward; grpo loss + rloo adv at kl 0.04; grpo adv at kl 0.01 — 3 seeds each | 9 | 8 h | the release's biggest negative result currently has three confounds (adv_type, kl 0.04, dense shaping) |
| A5 | GSPO ε ∈ {0.01, 0.2}, 3 more seeds each at 0.5B | 6 | 1 h | 3/3 paired wins for 0.01 (+3.6 pp) at n=3 is suggestive; n=6 decides the preset |
| A6 | 1.5B/MATH, 5 algos × 3 seeds | 15 | 13 h | the second reference table |
| A7 | 3B/MATH, 5 algos × 3 seeds | 15 | 27 h | plan target #3 — better on tier 2 (see B3) unless the desk is idle for a week |

Order: A1 → A2 → A3 → A4 → A5 → A6; A7 only if tier 2 is delayed. ≈ 37 h without A7,
four nights. Each cell = one wandb group per (algo, env); `aggregate_seeds.py` per group.

## 2. Tier 2 — 2×RTX PRO 6000 Blackwell (96 GB each), new

Roughly 2–3× a 4090 in bf16 with 4× the memory: the batch/group/sequence axes and the
7B model that the desk cannot hold, at a price where 3 seeds are affordable. Checks first:
the vLLM 0.28 wheel and flash-attn 2.8 kernels on sm_120 (fall back to `attn_implementation:
sdpa` if FA2 refuses); a 20-step colocated 7B LoRA smoke (15 GB weights in trainer and
vLLM, KV in the rest).

| id | cell | runs | why |
|---|---|---|---|
| B1 | Batch scaling, 1.5B/MATH, CISPO and DAPO: 64 → 128 → 256 → 512 traj/step (bs 1→8, gs 8), 1 seed each + 3 seeds at 128 and 512 | 12 | the "small batch causes the step-200 collapse" hypothesis; the throughput curve (tokens/s, s/step) for the perf story |
| B2 | Group size at fixed 256 traj/step, 3B/MATH: gs 8 / 16 / 32 (bs 32 / 16 / 8), DAPO, 3 seeds | 9 | informative groups on hard prompts (plan hypothesis; DAPO/DeepSeek recipes use 16–64) |
| B3 | 3B/MATH, 5 algos × 3 seeds at 256 traj/step, GC off | 15 | the release 3B/MATH row (replaces A7) |
| B4 | 7B/math500 and 7B/aime25: DAPO, CISPO, GSPO × 3 seeds, 128 traj/step, max_new 1024 (2048 for AIME) | 18 | plan targets #1 and #4 without H100; GRPO as one extra seed-3 baseline |
| B5 | Horizon: CISPO and Dr.GRPO 1.5B/MATH at 400 steps × 3 seeds | 6 | decides 200 vs 400 for paid 7B cells (rule: keep 400 only if the step-200→400 math500 gain exceeds 2× seed std) |
| B6 | Long CoT: 3B/MATH at max_new 2048 and 4096, DAPO, 3 seeds | 6 | AIME-style budgets; cap-rate and length dynamics with the fixed xmlcount |
| B7 | Colocated DDP×2 vs disaggregated (trainer + vLLM) at 7B, 50 steps | 2 | the topology decision for tier 3 (vivace's step is sequential, so disagg idles a card in every phase — expect colo to win until async rollout exists) |
| B8 | Full FT 1.5B (fp32 masters) vs LoRA r=16 / r=64, DAPO, 3 seeds | 9 | only after the full-FT fix (pure bf16 at lr 1e-6 rounds every update away) |

≈ 75 runs × 0.5–1.5 h ≈ $200–300. Order: smoke → B7 (informs tier 3 early) → B1 → B3 →
B4 → B2 → B5 → B6 → B8.

## 3. Tier 3 — Runpod 1-2-4-8 H100 SXM (H200 variant)

What stays from `v1_benchmark_plan.md` / `cluster_support.md`: one 8-GPU node and GPU
subsets for all four points (never four separate rentals); H100 SXM for writeup numbers,
A100 only as fallback; 3 seeds per headline cell; training step time reported separately
from eval time.

Changes and additions:

1. **Two scaling curves, not one.** The existing `math/cispo_{1,2,4,8}x80GB_colo.yaml`
   family is *weak* scaling (64 traj per rank, global batch grows with N). Add *strong*
   scaling: global batch fixed at 512 traj/step, grad_accum shrinking with N. Strong
   scaling is what "best performance for a fixed recipe" means; weak scaling is the
   batch-size study. Report both speedup curves plus accuracy at matched traj/step.
2. **Topology per GPU count.** Colocated DDP×N as the default (all cards generate, then all
   train); disaggregated N/2 + N/2 measured once at N=2 and N=8 as the comparison. TP=1
   throughout — 7B LoRA fits one 80 GB card in both roles — so TP=2 and FSDP leave the
   v1 plan. Asymmetric splits (2 vLLM + 6 trainers) need a trainer change; not for v1.
3. **Best-perf record run.** 7B/MATH on 8×H100 (and 8×H200 if available): chunked
   logprobs, gradient checkpointing off where memory allows, vLLM pool 0.6–0.7,
   `vllm_max_num_seqs` 512, per-rank bs 8 × gs 8 × accum 1 → 512 traj/step, CUDA graphs on,
   fastokens on, dedicated eval GPU. Report generated tokens/s, s/step, GPU-hours to 200
   steps, time-to-60% math500, and the final accuracy — the number the README leads with.
4. **Headline accuracy cells trimmed.** 7B/math500 and 7B/aime25 × {DAPO, CISPO, GSPO}
   × 3 seeds (+ GRPO once) ≈ 24 runs instead of 45; 3B cells come from tier 2; 7B/gsm8k
   only after the boxed-first matcher (the "template artifact" verdict was measured through
   the broken matcher).
5. **H100 vs H200 at N=2**, same recipe: H200's 141 GB allows GC off and 2× the KV pool —
   one cost/perf datapoint per GPU-hour.
6. **Manifests and pinned data** for every paid run; crashed runs excluded from aggregates.

Order and cost (prices per `cluster_support.md`): (1) 7B smoke, colo vs disagg timing on
2×H100, 20 steps (~$3) → (2) weak + strong scaling curves at 1.5B and 7B, CISPO and DAPO,
1 seed, 200 steps (≈ 16 runs, ~$150) → (3) headline 7B cells, 3 seeds (≈ 24 runs, ~$150)
→ (4) record run (~$50) → (5) H200 variant of (1) and (4) if the SKU is available.
Roughly $400 for the whole tier before H200.

## Decisions this plan assumes (owner's)

- Reward fixes (xmlcount cap, strict_format, soft_format) land before tier 1 — yes/no.
- Temperature and DAPO clip_high follow A1 and the config corpus — yes/no.
- 3B/MATH moves to tier 2 (B3) instead of 27 desk-hours (A7).
- Tier 3 runs the trimmed headline set (24 runs) rather than the 45-run matrix; the
  2×2 factorial (z-scored vs centered advantage × token vs sequence weighting) is the
  attribution experiment behind it and can replace named-algo cells if preferred.
- AIME is scored avg@16 at 7B.
