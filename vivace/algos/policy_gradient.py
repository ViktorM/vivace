"""Composable policy-gradient core — the heart of vivace.

==============================================================================
DESIGN: CONFIG SWITCHES, NOT CLASS HIERARCHIES
==============================================================================

This file holds the entire PG zoo (GRPO, Dr.GRPO, DAPO, GSPO, CISPO, DG,
DG-CISPO, RLOO) in a single composable form. The composition surface is
two string fields on `RLConfig` — `loss_type` and `adv_type` — that
`compute_loss` and `compute_advantages` dispatch on with if/elif ladders.

WHY this design:
  - Composability for free: loss_type=gspo + adv_type=rloo gives you
    "GSPO with leave-one-out advantages" without writing a new class.
  - Diff-friendly research: adding a new variant is ~10 lines in one
    elif branch, not a new file + registry update.
  - All variants visible side-by-side. Hold the whole zoo in working
    memory and compare clipping styles directly.

WHAT THIS FILE OWNS:
  - build_response_mask  - loss mask from response lengths, no forward
  - compute_advantages   - 3-way dispatch on cfg.adv_type
  - compute_token_logprobs - forward pass, masking, optional entropy
  - compute_kl           - Schulman k3 estimator
  - compute_loss         - 8-way dispatch on cfg.loss_type (dg*: stubs)
  - rl_step              - multi-epoch optimization over the trainer's
                           micro-batches; adaptive LR is a stub

WHAT THIS FILE DOES NOT OWN:
  - Sampling (that's vivace/rollout/{hf_sampler,vllm_worker}.py)
  - Reward computation (that's vivace/rewards.py)
  - Optimizer construction (that's the trainer)
  - Distributed setup (that's vivace/utils/distributed.py)
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.profiler import record_function

from vivace.algos.types import RLConfig
from vivace.utils.distributed import get_world_size


def build_response_mask(
    prompt_len: int,
    seq_len: int,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    """Loss mask over target positions, built purely from response lengths.

    Returns fp32 [B, seq_len - 1]. Targets are shifted by one: target index t
    corresponds to full_ids[t + 1], so the response occupies target positions
    [prompt_len - 1, prompt_len - 1 + resp_len). No forward pass needed —
    this is what lets the trainer skip the old_logp recompute entirely.
    """
    device = response_lengths.device
    start = max(prompt_len - 1, 0)
    resp_lens = response_lengths.long().unsqueeze(1)              # (B, 1)
    pos = torch.arange(seq_len - 1, device=device).unsqueeze(0)   # (1, S-1)
    return ((pos >= start) & (pos < start + resp_lens)).float()


# =============================================================================
# 1. ADVANTAGES — 3-way dispatch on cfg.adv_type
# =============================================================================
def compute_advantages(rewards: torch.Tensor, cfg: RLConfig) -> torch.Tensor:
    """Compute per-sequence advantages from group rewards.

    Args:
        rewards: [B, G] or flat [B*G]; reshaped to (-1, G) internally.
        cfg: RLConfig — `adv_type`, `group_size`, `adv_eps` are read.

    Returns:
        Tensor of shape [B*G], detached. To be broadcast against the
        token mask in the loss.

    THEORY
    -----
    Three flavours, all of them group-relative baselines:

    - "grpo"    : (r - mean) / (std + eps)            — z-scored within group
    - "dr_grpo" : r - mean                            — same baseline, no std
    - "rloo"    : r - leave-one-out mean              — bias-corrected baseline

    Why no GAE / value head? Single-turn RL with terminal reward only:
    there's nothing to bootstrap. The group baseline does the variance
    reduction job.

    HINTS
    -----
    - rg = rewards.view(-1, G) — B inferred from the total, robust to adaptive filtering
    - For RLOO: total = rg.sum(-1, keepdim=True); baseline = (total - rg) / (G - 1)
    - For GRPO z-score: use rg.std(-1, keepdim=True, unbiased=False) — biased std
      is what most papers report.
    - Always .detach() the result. Advantages must NOT carry gradients.
    """
    G = cfg.group_size
    rg = rewards.view(-1, G)   # infer batch from total, robust to adaptive filtering

    if cfg.adv_type == "rloo":
        if G < 2:
            raise ValueError(f"RLOO requires group_size >= 2 (leave-one-out), got {G}")
        baseline = (rg.sum(dim=1, keepdim=True) - rg) / (G - 1)
        adv = rg - baseline
    elif cfg.adv_type == "dr_grpo":
        adv = rg - rg.mean(dim=1, keepdim=True)
    elif cfg.adv_type == "grpo":
        adv = (rg - rg.mean(dim=1, keepdim=True)) / (rg.std(dim=1, keepdim=True, unbiased=False) + cfg.adv_eps)
    else:
        raise ValueError(f"unknown adv_type {cfg.adv_type!r}, expected grpo | dr_grpo | rloo")

    return adv.view(-1).detach()


# =============================================================================
# 2. TOKEN LOG-PROBS — forward pass, masking, optional entropy
# =============================================================================
def compute_token_logprobs(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
    return_entropy: bool = False,
    entropy_chunk_size: int = 64,
    entropy_grad: bool = True,
    pad_token_id: int | None = None,
    stop_token_id: int | None = None,
    padding_side: str = "left",
    response_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Forward pass + per-token log-prob extraction + response mask.

    Returns (token_logp, mask, entropy_or_None), each of shape [B, S-1].
    `entropy_or_None` is None when `return_entropy=False` and an fp32 tensor
    otherwise (cast inside this function so callers don't need to .float()).

    Args:
        model: the model to forward through (policy, old, or ref).
        full_ids: [B, S] — left-padded prompt + response token ids.
        prompt_len: padded prompt length (positions < prompt_len are prompt).
        temperature: logit scaling — must match the sampling temperature.
        return_entropy: if True, also return per-token entropy [B, S-1].
        entropy_chunk_size: when > 0 and < T, compute entropy in slices of
                       this many time steps to bound the `[B, chunk, V]` exp()
                       temporaries; 0 = single-shot. 64 is the sweet spot at
                       Qwen2.5 vocab (`RLConfig.entropy_chunk_size`).
        entropy_grad: True (default) keeps the chunked entropy in the autograd
                       graph via `torch.cat` — needed when entropy enters the
                       loss. False writes a no_grad pre-allocated buffer: frees
                       ~50% more peak memory but no `grad_fn` — backprop through
                       it silently yields zero, so logging-only callers only.
        pad_token_id: token id used for left-padding. Used to build
                      the attention_mask for the forward pass.
        stop_token_id: end-of-response token for the LOSS mask; only used when
                       `response_lengths` is None. Defaults to pad_token_id —
                       right for single-turn GSM8K where EOS == pad == stop.
                       Multi-turn / tool-use: EOS is a turn separator, so pass
                       the real delimiter, e.g. tokenizer.encode("</answer>")[0].
                       If it never appears (max_tokens hit) every response token
                       stays in the mask — the model never chose to stop.
        padding_side: "left" (only supported value for now).
        response_lengths: optional [B] true response lengths (pre right-padding);
                       the trainer always passes it. Builds the response mask
                       exactly; without it the stop-token fallback off-by-ones
                       on max-tokens truncation when pad_id == eos_id.

    THEORY
    ------
    The trainer needs log-probs of the SAMPLED tokens under the CURRENT
    policy ("new"), the rollout policy ("old", importance ratio) and the
    frozen reference ("ref", KL). New and ref call this function with
    different models; old is the detached epoch-1 policy forward, which
    equals a recompute because rollout weights == epoch-1 weights
    (needs dropout == 0, enforced at trainer init).

    TWO MASKS, TWO PURPOSES
    -----------------------
    `attention_mask` and `mask` are different things:

      attention_mask: tells the transformer "which positions exist vs pad."
        Passed to model(full_ids, attention_mask=...). Without it, the model
        attends to pad tokens and produces garbage logits. Built from
        pad_token_id (left pads) + response_lengths (right pads). Operates
        INSIDE the forward pass.

      mask (response mask): tells the loss "which token log-probs count."
        Even on positions where the model produced valid logits, we zero out
        prompt tokens and everything past the response. Built from
        response_lengths (stop_token_id is the fallback). Operates OUTSIDE
        the forward pass, on the loss.

    Temperature scaling on the logits matches the rollout sampling
    temperature so the importance ratio is on the same distribution.

    GOTCHAS
    -------
    - Logits are [B, S, V] but log-probs of the sampled token are
      [B, S-1] because position t predicts token t+1. Slice logits[:, :-1]
      and gather over targets[:, 1:].
    - With LEFT-PADDING (the default in HF tokenizer for generation), the
      attention mask must be 1 - cumprod(is_pad). With right-padding, it's
      different. This codebase uses left-padding throughout — stick with it.
    - log_softmax runs in bf16 to match the rest of the pipeline (model
      forward, vLLM, gradients). Halves the [B, S-1, V] tensor vs fp32
      and avoids an extra cast. Importance ratios and KL still
      reduce in fp32 downstream where it matters.
    - The entropy computation re-uses log_probs — compute probs as
      log_probs.exp() (no redundant softmax) and cast the [B, S-1] reduction
      to fp32 for stable accumulation across the vocab dim.

    HINTS
    -----
    - Build attention_mask from pad_token_id (for the forward pass).
    - logits = model(full_ids, attention_mask=...).logits[:, :-1, :] / temperature
    - targets = full_ids[:, 1:]
    - log_probs = F.log_softmax(logits, dim=-1)   # bf16
    - token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    - RESPONSE MASK: build_response_mask(...) from response_lengths; the
      fallback uses stop_token_id (not pad_token_id):
        stop_id = stop_token_id if stop_token_id is not None else pad_token_id
        is_stop = (targets == stop_id)
        is_stop[:, :start] = False
        no_prior_stop = (is_stop.cumsum(dim=1) - is_stop.long()) == 0
        mask[:, start:] = no_prior_stop[:, start:].float()
    - For entropy: probs = log_probs.exp(); entropy = -(probs * log_probs).sum(dim=-1)

    """
    if padding_side != "left":
        raise ValueError(f"unknown padding {padding_side!r}, only left padding is supported at the moment")

    B, S = full_ids.shape
    device = full_ids.device

    # Attention mask: 1 at real tokens (left-unpadded prompt + first resp_len response tokens).
    # Strips BOTH leading left-pads AND trailing right-pads. Without response_lengths
    # we can't distinguish right-pads from real EOS tokens (pad_id == eos_id commonly),
    # so we fall back to the left-pad-only cumprod trick.
    is_pad = full_ids == pad_token_id
    left_pad_mask = 1 - is_pad.cumprod(dim=-1).long()  # 1 after first non-pad
    if response_lengths is not None:
        resp_lens = response_lengths.to(device).long().unsqueeze(1)  # (B, 1)
        pos = torch.arange(S, device=device).unsqueeze(0)             # (1, S)
        within_content = pos < (prompt_len + resp_lens)                # True for prompt + real response
        attention_mask = (left_pad_mask.bool() & within_content).long()
    else:
        attention_mask = left_pad_mask

    # RoPE rotates by absolute position regardless of attention_mask. With left-padding,
    # default position_ids = arange(S) puts real tokens at K, K+1, ... where K is the
    # leading-pad count — different from vLLM's unpadded 0, 1, ... and from the same
    # prompt in a differently-padded batch. Build position_ids 0-indexed at first real
    # token so RoPE is invariant to padding.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    logits = model(full_ids, attention_mask=attention_mask, position_ids=position_ids).logits[:, :-1, :] / temperature  # (B, S-1, V)
    targets = full_ids[:, 1:]  # (B, S-1)

    log_probs = F.log_softmax(logits, dim=-1)  # (B, S-1, V)
    token_log_prob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) # (B, S-1)

    # Loss mask: which target positions count toward PG/KL.
    # Targets are shifted by 1 — target index t corresponds to full_ids[t+1].
    # Response target range (inclusive): [prompt_len-1, prompt_len-1 + resp_len - 1].
    start = max(prompt_len - 1, 0)
    mask = torch.zeros_like(token_log_prob)
    if response_lengths is not None:
        mask = build_response_mask(prompt_len, S, response_lengths.to(device))
    else:
        # Fallback: stop-token heuristic. Correct when EOS genuinely terminates the
        # sequence; off-by-one (includes first right-pad) for max_tokens-truncated
        # sequences where pad_id == eos_id. Prefer passing response_lengths.
        stop_id = stop_token_id if stop_token_id else pad_token_id
        is_eos = targets == stop_id  # (B, S-1)
        is_eos[:, :start] = False
        no_prior_eos = (is_eos[:, start:].cumsum(dim=-1) - is_eos[:, start:].long()) == 0
        mask[:, start:] = no_prior_eos.float()

    # Keep masks in fp32 so downstream reductions and denominators do not
    # inherit bf16 precision from the model forward.
    mask = mask.float()

    B, S_minus_one, _ = log_probs.shape
    entropy = None
    if return_entropy:
        chunked = bool(entropy_chunk_size) and entropy_chunk_size < S_minus_one
        if entropy_grad:
            # Autograd-friendly path: torch.cat is a differentiable functional
            # op, so chunks stay connected to the graph and entropy can be
            # used in the loss (entropy-bonus variants). Peak savings ~23% vs
            # single-shot — most bytes are spent on `exp()` results retained
            # for backward.
            if chunked:
                chunks = [
                    -(log_probs[:, i:i + entropy_chunk_size, :].exp()
                      * log_probs[:, i:i + entropy_chunk_size, :]).sum(dim=-1)
                    for i in range(0, S_minus_one, entropy_chunk_size)
                ]
                entropy = torch.cat(chunks, dim=1).float()
            else:
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).float()
        else:
            # Memory-optimal path for logging-only entropy. Pre-allocated
            # buffer + in-place writes; nothing retained for backward. Peak
            # savings ~74% vs single-shot. The __setitem__ breaks the graph
            # regardless of no_grad, so backward through `entropy` silently
            # yields zero — DO NOT use this when entropy enters the loss.
            with torch.no_grad():
                entropy = torch.empty((B, S_minus_one), dtype=torch.float32,
                                       device=log_probs.device)
                if chunked:
                    for i in range(0, S_minus_one, entropy_chunk_size):
                        s = log_probs[:, i:i + entropy_chunk_size, :]
                        entropy[:, i:i + entropy_chunk_size] = (
                            -(s.exp() * s).sum(dim=-1).float()
                        )
                else:
                    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).float()

    return token_log_prob, mask, entropy


# =============================================================================
# 3. KL DIVERGENCE — Schulman k3 (default) or k1 naive estimator
# =============================================================================
def compute_kl(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    mask: torch.Tensor,
    estimator: str = "k3",
) -> torch.Tensor:
    """Estimated KL(policy || ref) per sequence. Returns shape [B].

    Args:
        policy_logp: [B, S-1] log-probs of sampled tokens under current policy
        ref_logp:    [B, S-1] log-probs of same tokens under frozen reference
        mask:        [B, S-1] response token mask (1 for real, 0 for padding)
        estimator:   "k3" (default, Schulman unbiased) | "k1" (naive log-ratio)

    THEORY
    ------
    Both estimators approximate KL(policy || ref) from samples drawn from
    `policy`. They differ in variance and sign behavior.

    k1 — NAIVE LOG-RATIO
        KL_k1 ≈ E_{x~policy}[ log policy(x) - log ref(x) ]
        per token: `policy_logp - ref_logp` = -log_r

        + Simplest estimator, unbiased by definition.
        - HIGH VARIANCE: one outlier token with ratio ~0.01 or ~100
          dominates the mean.
        - Can go NEGATIVE on any finite sample — expected, don't "fix" it
          with abs().
        - Noisy signal for adaptive KL control (LR oscillates).

    k3 — SCHULMAN UNBIASED
        KL_k3 ≈ E[ r - log(r) - 1 ]   where r = exp(ref_logp - policy_logp)

        + ALWAYS >= 0 (r - log(r) - 1 has minimum 0 at r=1).
        + Unbiased (same expectation as k1, different finite-sample
          distribution), lower variance in practice, stable signal for
          adaptive KL / LR control.
        - One exp + one add more than k1.

        See John Schulman's blog post "Approximating KL Divergence"
        (http://joschu.net/blog/kl-approx.html) for the derivation and
        variance analysis. This is the GRPO paper's estimator (DAPO drops
        the KL term; vivace's dapo recipes keep it).

    DEFAULT: k3. k1 is for debugging / comparison — run both over many
    steps; the means should agree to within sampling noise.

    HINTS
    -----
    Both branches reduce to: `(per_token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)`

    - k1:   per_token_kl = policy_logp - ref_logp
    - k3:   log_r = ref_logp - policy_logp
            r = torch.exp(log_r)
            per_token_kl = r - log_r - 1

    GOTCHA
    ------
    k3 clamps log_r to [-10, 10] before exp() so one outlier token cannot
    produce inf. `compute_loss` / `rl_step` clamp the POLICY log-ratio
    (policy/old) to [-5, 5] — a different ratio; ±10 is looser and sufficient
    here without distorting typical values.
    """
    policy_logp_f = policy_logp.float()
    ref_logp_f = ref_logp.float()
    mask_f = mask.float()

    if estimator == "k1":
        per_token_kl = policy_logp_f - ref_logp_f
    elif estimator == "k3":
        log_r = (ref_logp_f - policy_logp_f).clamp(-10, 10)
        r = torch.exp(log_r)
        per_token_kl = r - log_r - 1.0
    else:
        raise ValueError(f"unknown KL estimator: {estimator!r}")

    return (per_token_kl * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# =============================================================================
# 4. LOSS — 8-way dispatch on cfg.loss_type
# =============================================================================
def compute_loss(
    cfg: RLConfig,
    policy_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    token_count: torch.Tensor,
    token_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Policy gradient loss for the configured variant.

    Args:
        cfg: RLConfig — reads `loss_type`, `clip_low`, `clip_high`, `clip_cispo_low`,
             `clip_cispo_high`, `cispo_use_token_mask`, `cispo_normalization`, `max_new_tokens`.
        policy_logp: [B*G, S-1] under current policy (with grad).
        old_logp: [B*G, S-1] under rollout policy (no grad).
        advantages: [B*G] (no grad).
        mask: [B*G, S-1] response token mask.
        token_count: [B*G] number of unmasked tokens per sequence.
        token_norm: denominator for the token-mean losses (dapo; cispo
             token/hybrid). Under grad accumulation this must be the GLOBAL
             mean token count per micro-batch (all micro-batches, all DDP
             ranks — `rl_step` computes it), so the papers' sum(obj)/total_tokens
             objective survives the per-micro-batch `/ n` averaging. None
             (single whole batch, tests) falls back to this batch's own count.

    Returns:
        Scalar loss tensor (with grad). Does NOT include KL — that's added
        by `rl_step` outside this function so the KL coefficient stays visible.

    THEORY (one paragraph per branch)
    -----
    - "rloo"     : pure REINFORCE with leave-one-out baseline. No clipping.
                   loss = -mean(adv * mean_logp_per_seq)

    - "grpo"     : token-level PPO ratio with symmetric clipping, then
                   sequence-mean of clipped objective.

    - "dr_grpo"  : like GRPO but normalize by max_new_tokens instead of
                   actual response length. Removes a length bias.

    - "dapo"     : token-level PPO ratio, ASYMMETRIC clipping (clip_high
                   typically > clip_low), token-level normalization (sum
                   over all tokens / total token count, not per-sequence).

    - "gspo"     : SEQUENCE-level ratio. Compute mean log-prob per sequence
                   on both new and old, exponentiate the difference, then
                   clip. One ratio per sequence, not per token.

    - "cispo"    : Importance-clipped policy-gradient. Detach the clamped
                   weight and multiply policy_logp directly. Canonical CISPO
                   preserves gradients for all tokens; the MiniMax-M1 eq. 7
                   / PPO-style token mask is an opt-in via
                   cfg.cispo_use_token_mask. cfg.cispo_normalization picks
                   token mean, per-sequence mean, or hybrid (average; default).

    - "dg"       : STUB (NotImplementedError). Design: gate the PG term by
                   sigmoid(eta * adv * surprisal), surprisal = -policy_logp.detach().

    - "dg_cispo" : STUB (NotImplementedError). Design: CISPO's clamped
                   importance weight AND DG's sigmoid gate.

    GOTCHAS
    -------
    - Clamp log_ratio to [-5, 5] before exp() to avoid overflow on outliers
      (this matters at temperature > 0.7).
    - Advantages must be unsqueezed for broadcasting: a = advantages.unsqueeze(1)
    - Don't forget the negative sign on the final loss (we're maximizing).

    """
    policy_logp_f = policy_logp.float()
    old_logp_f = old_logp.float()
    advantages_f = advantages.float()
    mask_f = mask.float()
    token_count_f = token_count.float().clamp(min=1.0)

    if cfg.loss_type == "rloo":
        mean_logp = (policy_logp_f * mask_f).sum(dim=1) / token_count_f  # (B*G,)
        loss = -(advantages_f * mean_logp).mean()
    # sequence-level ratio
    elif cfg.loss_type == "gspo":
        seq_logp = (policy_logp_f * mask_f).sum(dim=1) / token_count_f  # (B*G,)
        old_seq_logp = (old_logp_f * mask_f).sum(dim=1) / token_count_f
        seq_log_ratio = (seq_logp - old_seq_logp).clamp(-5, 5)  # clamp for stability
        seq_ratio = torch.exp(seq_log_ratio)  # (B*G,)
        # we support asymmetric clipping for GSPO as well
        seq_clipped = torch.clamp(seq_ratio, 1.0 - cfg.clip_low, 1.0 + cfg.clip_high)
        loss = -torch.min(advantages_f * seq_ratio, advantages_f * seq_clipped).mean()
    # all token-level variants
    else:
        log_ratio = (policy_logp_f - old_logp_f).clamp(-5, 5)  # clamp for stability
        ratio = torch.exp(log_ratio)  # (B*G, S-1)
        clipped = torch.clamp(ratio, 1 - cfg.clip_low, 1 + cfg.clip_high)
        a = advantages_f.unsqueeze(1)  # (B*G, 1)
        obj = torch.min(ratio * a, clipped * a)
        # token-level normalization
        if cfg.loss_type == "dapo":
            denom = token_norm if token_norm is not None else mask_f.sum().clamp(min=1.0)
            loss = -(obj * mask_f).sum() / denom
        elif cfg.loss_type == "grpo":
            per_seq = (obj * mask_f).sum(dim=1) / token_count_f
            loss = -per_seq.mean()
        elif cfg.loss_type == "dr_grpo":
            per_seq = (obj * mask_f).sum(dim=1) / cfg.max_new_tokens
            loss = -per_seq.mean()
        elif cfg.loss_type == "cispo":
            # Detached clipped IS weight (MiniMax-M1 eq. 5). Canonical CISPO
            # clips the weight but keeps every response token in the gradient.
            is_weight = ratio.clamp(min=cfg.clip_cispo_low, max=cfg.clip_cispo_high).detach()
            if cfg.cispo_use_token_mask:
                # Optional trust-region mask (MiniMax-M1 eq. 7): drop tokens
                # whose ratio drifted outward in the same direction as the
                # advantage. This intentionally reintroduces PPO-like token
                # dropping and is not the default CISPO objective.
                keep_high = (ratio <= cfg.clip_cispo_high) | (a <= 0.0)
                keep_low = (ratio >= cfg.clip_cispo_low) | (a >= 0.0)
                is_mask = (keep_high & keep_low).to(ratio.dtype)
            else:
                is_mask = torch.ones_like(ratio)

            per_token = -is_weight * a * policy_logp_f * is_mask * mask_f
            denom = token_norm if token_norm is not None else mask_f.sum().clamp(min=1.0)
            token_loss = per_token.sum() / denom
            sequence_loss = (per_token.sum(dim=1) / token_count_f).mean()
            if cfg.cispo_normalization == "token":
                loss = token_loss
            elif cfg.cispo_normalization == "sequence":
                loss = sequence_loss
            elif cfg.cispo_normalization == "hybrid":
                loss = 0.5 * (token_loss + sequence_loss)
            else:
                raise ValueError(f"unknown cispo_normalization: {cfg.cispo_normalization!r}")
        elif cfg.loss_type == "dg":
            raise NotImplementedError("dg loss not yet implemented")
        elif cfg.loss_type == "dg_cispo":
            raise NotImplementedError("dg_cispo loss not yet implemented")
        else:
            raise ValueError(f"unknown loss: {cfg.loss_type!r}")

    return loss


# =============================================================================
# 5. FULL RL STEP — multi-epoch optimize over pre-collected micro-batches, build metrics
# =============================================================================
def rl_step(
    cfg: RLConfig,
    micro_batches: list[dict],
    model,
    ref_model,
    optimizer,
    stats,
    step: int,
    kl_ema: float,
) -> tuple[dict[str, float], float]:
    """One full RL training step.

    Args:
        cfg: RLConfig
        micro_batches: list of pre-collected micro-batches from the trainer's
                       rollout_phase. Keys: full_ids, plen, adv, old_logp (None
                       on entry — filled from the epoch-1 forward), ref_logp
                       (None when kl_coef == 0), mask, token_count, responses,
                       rewards, pad_token_id, response_lengths.
        model: trainable policy (already DDP-wrapped if distributed).
        optimizer: AdamW or similar, already constructed.
        ref_model/stats/step: unused — kept for call-site compatibility
                       (ref_logp is precomputed in rollout_phase; the trainer
                       is the single stats.log() caller).
        kl_ema: returned unchanged (adaptive LR is a stub).

    Returns:
        (metrics_dict, kl_ema). metrics_dict has loss/reward/kl/clip_frac/...
        plus `_*` sufficient stats the trainer reduces across ranks and drops.

    PHASES
    ------
    1. Multi-epoch optimization over the collected micro-batches.
       Loop `cfg.optim_epochs` times (1 for RLOO). Each epoch:
         a. zero_grad
         b. for each micro-batch: forward pass through model -> policy_logp,
            compute_kl, compute_loss, scale by 1/len(micro_batches), backward
         c. (only on the LAST epoch) accumulate clip stats, entropy, etc.
         d. clip_grad_norm_ + optimizer.step

    2. Adaptive LR (cfg.use_adaptive_lr): STUB — `pass`, so kl_target /
       kl_factor / lr_factor / min_lr / max_lr have no effect.

    3. Build and return the metrics dict — no logging here.

    GOTCHAS
    -------
    - The forward pass MUST be inside the inner loop (per micro-batch),
      not amortised across epochs. Multi-epoch means re-doing the forward
      with the latest weights each epoch.
    - Loss is divided by len(micro_batches) so that .backward() correctly
      averages across grad accumulation.
    - clip_frac counting differs between sequence-level (GSPO) and
      token-level (everything else) — see the branching below.
    - RLOO: validate_rl_config (trainer init) warns and forces optim_epochs=1.
    - Learning metrics reflect the LAST epoch only (`if last`).

    """
    model.train()
    n = len(micro_batches)
    n_epochs = cfg.optim_epochs  # RLOO must have optim_epochs=1 — enforced by validate_rl_config

    # Token-mean losses (dapo; cispo token/hybrid) must divide by the GLOBAL
    # token count, not each micro-batch's own — per-micro-batch denominators
    # upweight short-response micro-batches under grad accumulation, and
    # low-token ranks under DDP. Dividing the global count by (n * world_size)
    # keeps the uniform `loss / n` scaling below exact: summed over micro-batches
    # and averaged over ranks by DDP, the gradient is sum(obj) / total_tokens.
    token_norm = None
    needs_token_norm = cfg.loss_type == "dapo" or (
        cfg.loss_type == "cispo" and cfg.cispo_normalization != "sequence")
    if needs_token_norm:
        total_tokens = torch.stack([mb["token_count"].sum() for mb in micro_batches]).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        token_norm = total_tokens / (n * get_world_size())

    # Stats accumulators — only populated on the last epoch
    tot_kl = tot_pg = tot_kl_loss = tot_ent = 0.0
    tot_clip = tot_tok = 0.0
    # GSPO seq_ratio samples for p50/p99 logging (diagnoses inert trust region).
    seq_ratio_chunks: list[torch.Tensor] = []

    # DDP: gradients are only consumed at optimizer.step(), so the all-reduce
    # is needed solely on the LAST micro-batch of each epoch; no_sync() skips
    # it on the others (grad_accum_steps x fewer collectives per epoch).
    inner = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    ddp = inner if isinstance(inner, DistributedDataParallel) else None

    for epoch in range(n_epochs):
        optimizer.zero_grad(set_to_none=True)
        last = (epoch == n_epochs - 1)

        for mb_idx, mb in enumerate(micro_batches):
            sync_ctx = ddp.no_sync() if (ddp is not None and mb_idx < n - 1) else nullcontext()
            with sync_ctx:
                policy_logp, _, entropy = compute_token_logprobs(
                    model, mb["full_ids"], mb["plen"], cfg.temperature,
                    return_entropy=last,
                    entropy_chunk_size=cfg.entropy_chunk_size,
                    entropy_grad=cfg.entropy_grad,
                    pad_token_id=mb.get("pad_token_id"),
                    stop_token_id=mb.get("stop_token_id"),
                    response_lengths=mb.get("response_lengths"),
                )
                # old_logp = the epoch-1 forward, detached. The rollout weights
                # ARE the epoch-1 weights, and the recompute the trainer used to
                # do here is bit-identical to this forward (measured: eval/no-grad
                # vs train/grad logits agree exactly; requires dropout == 0,
                # enforced at trainer init). Saves one full no-grad forward per
                # micro-batch per step.
                if mb["old_logp"] is None:
                    mb["old_logp"] = policy_logp.detach()
                if mb["ref_logp"] is not None:
                    kl = compute_kl(policy_logp, mb["ref_logp"], mb["mask"])
                elif cfg.kl_coef != 0.0:
                    raise ValueError(
                        "ref_logp is None but kl_coef != 0 — the trainer only "
                        "skips the reference forward when the KL term is off"
                    )
                else:
                    # kl_coef == 0: ref forward skipped in rollout_phase; KL ≡ 0.
                    kl = policy_logp.new_zeros(policy_logp.shape[0])
                pg_loss = compute_loss(
                    cfg, policy_logp, mb["old_logp"],
                    mb["adv"], mb["mask"], mb["token_count"],
                    token_norm=token_norm,
                )
                kl_loss = cfg.kl_coef * kl.mean()
                loss = (pg_loss + kl_loss) / n
                loss.backward()

            if last:
                with torch.no_grad():
                    tot_kl += kl.mean()
                    tot_pg += pg_loss
                    tot_kl_loss += kl_loss
                    if entropy is not None:
                        tot_ent += (entropy * mb["mask"]).sum() / mb["mask"].sum().clamp(min=1.0)

                    # Clip fraction: GSPO counts sequences, PPO-style variants count
                    # tokens — not cross-comparable. At optim_epochs=1 old_logp IS this
                    # epoch's forward, so ratio ≡ 1 and clip_frac ≡ 0 by construction;
                    # real clip activity needs epochs >= 2.
                    if cfg.loss_type == "gspo":
                        seq_lp = (policy_logp * mb["mask"]).sum(dim=1) / mb["token_count"]
                        old_seq_lp = (mb["old_logp"] * mb["mask"]).sum(dim=1) / mb["token_count"]
                        ratio = torch.exp((seq_lp - old_seq_lp).clamp(-5, 5))
                        # Asymmetric: count clips in both directions, not |ratio-1| > clip_high.
                        clipped = (ratio < 1 - cfg.clip_low) | (ratio > 1 + cfg.clip_high)
                        tot_clip += clipped.float().sum()
                        tot_tok += float(len(ratio))
                        seq_ratio_chunks.append(ratio.detach())
                    elif cfg.loss_type == "cispo":
                        # CISPO clip_frac reports clipped IS weights, not
                        # token drops. Token dropping is optional and off by
                        # default; clipped weights are always part of CISPO.
                        log_ratio = (policy_logp.float() - mb["old_logp"].float()).clamp(-5, 5)
                        ratio = torch.exp(log_ratio)
                        clip_high = ratio > cfg.clip_cispo_high
                        if cfg.clip_cispo_low > 0.0:
                            clip_low = ratio < cfg.clip_cispo_low
                        else:
                            clip_low = torch.zeros_like(clip_high)
                        clipped_mask = (clip_high | clip_low) & (mb["mask"] > 0)
                        tot_clip += clipped_mask.sum().float()
                        tot_tok += mb["mask"].sum()
                    elif cfg.uses_clipping:
                        ratio = torch.exp((policy_logp - mb["old_logp"]).clamp(-5, 5))
                        clipped_mask = (
                            (ratio < 1 - cfg.clip_low) | (ratio > 1 + cfg.clip_high)
                        ) & (mb["mask"] > 0)
                        tot_clip += clipped_mask.sum().float()
                        tot_tok += mb["mask"].sum()

        grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    # ===== Gather stats across all micro-batches =====
    all_rewards = torch.cat([mb["rewards"] for mb in micro_batches])
    all_responses = [r for mb in micro_batches for r in mb["responses"]]
    all_token_counts = torch.cat([mb["token_count"] for mb in micro_batches])
    all_advantages = torch.cat([mb["adv"] for mb in micro_batches])

    # Format rate: fraction of responses with a parseable <answer> tag
    from vivace.rewards import extract_answer
    fmt_ok = sum(1 for r in all_responses if extract_answer(r) != "")

    # Build the full metrics dict — returned to the trainer, which is the
    # single place that calls stats.log() (avoids double-entry in stats.steps).
    # Wrapped: the first .item() forces a sync waiting for backward+optim,
    # so this block dominates cudaDeviceSynchronize attribution.
    #
    # `_*` keys are sufficient statistics (Σx, Σx², N, raw counts) used by the
    # trainer to derive globally-correct mean / std / ratio under DDP. They
    # never reach wandb or the user-facing stats — the trainer drops them
    # after reduction. Same convention as the eval path's `_raw_keys` plumbing.
    with record_function("metrics_dict"):
        n_samples = len(all_rewards)
        metrics = {
            # Core
            "loss": (tot_pg / n + cfg.kl_coef * tot_kl / n).item(),
            "reward": all_rewards.mean().item(),
            "kl": (tot_kl / n).item(),
            "clip_frac": (tot_clip / tot_tok).item() if tot_tok > 0 else 0.0,
            "grad_norm": grad_norm.item(),
            "entropy": (tot_ent / n).item(),
            # Lengths
            "length_mean": all_token_counts.mean().item(),
            "length_std": all_token_counts.std().item() if len(all_token_counts) > 1 else 0.0,
            "length_max": all_token_counts.max().item(),
            "length_min": all_token_counts.min().item(),
            # Reward distribution
            "reward_std": all_rewards.std().item() if len(all_rewards) > 1 else 0.0,
            "reward_max": all_rewards.max().item(),
            "reward_min": all_rewards.min().item(),
            # Advantage signal
            "advantage_std": all_advantages.std().item() if len(all_advantages) > 1 else 0.0,
            # Format rate
            "format_rate": fmt_ok / len(all_responses) if all_responses else 0.0,
            # Sufficient stats for DDP reduction (consumed and dropped by trainer)
            "_n": n_samples,
            "_format_ok": fmt_ok,
            "_reward_sum": all_rewards.sum().item(),
            "_reward_sumsq": (all_rewards * all_rewards).sum().item(),
            "_length_sum": all_token_counts.sum().item(),
            "_length_sumsq": (all_token_counts * all_token_counts).sum().item(),
            "_advantage_sum": all_advantages.sum().item(),
            "_advantage_sumsq": (all_advantages * all_advantages).sum().item(),
            "_clip_count": tot_clip.item() if torch.is_tensor(tot_clip) else float(tot_clip),
            "_clip_tokens": tot_tok.item() if torch.is_tensor(tot_tok) else float(tot_tok),
        }
        # GSPO sequence-ratio percentiles — diagnoses inert clip (paper's eps≈3e-4
        # vs vivace default 0.2: if p99 stays within 1±0.2, the clip never bites).
        if seq_ratio_chunks:
            seq_ratios = torch.cat(seq_ratio_chunks)
            metrics["seq_ratio_p50"] = seq_ratios.median().item()
            metrics["seq_ratio_p99"] = seq_ratios.quantile(0.99).item()

    if cfg.use_adaptive_lr:
        pass  # skipped for now — add later

    return metrics, kl_ema
