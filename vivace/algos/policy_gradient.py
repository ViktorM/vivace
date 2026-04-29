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
  - compute_advantages   - 3-way dispatch on cfg.adv_type
  - compute_token_logprobs - forward pass, masking, optional entropy
  - compute_kl           - Schulman k3 estimator
  - compute_loss         - 8-way dispatch on cfg.loss_type
  - rl_step              - the full step: collect micro-batches,
                           multi-epoch optimization, adaptive LR

WHAT THIS FILE DOES NOT OWN:
  - Sampling (that's vivace/rollout/{hf_sampler,vllm_worker}.py)
  - Reward computation (that's vivace/rewards.py)
  - Optimizer construction (that's the trainer)
  - Distributed setup (that's vivace/utils/distributed.py)

==============================================================================
NOTE
==============================================================================

`simple_rl_step` (educational scaffolding with no grad accumulation, no
multi-epoch, no adaptive sampling) is intentionally omitted. The full
`rl_step` below subsumes it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.profiler import record_function

from vivace.algos.types import RLConfig


# =============================================================================
# 1. ADVANTAGES — 3-way dispatch on cfg.adv_type
# =============================================================================
def compute_advantages(rewards: torch.Tensor, cfg: RLConfig) -> torch.Tensor:
    """Compute per-sequence advantages from group rewards.

    Args:
        rewards: shape [B, G]. Per-prompt-per-sample scalar rewards.
        cfg: RLConfig — only `adv_type` and `adv_eps` matter here.

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
    - rg = rewards.view(B, G) where B = rewards.shape[0]
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
    pad_token_id: int | None = None,
    stop_token_id: int | None = None,
    padding_side: str = "left",
    response_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Forward pass + per-token log-prob extraction + response mask.

    Returns (token_logp, mask, entropy_or_None), each of shape [B, S-1].

    Args:
        model: the model to forward through (policy, old, or ref).
        full_ids: [B, S] — left-padded prompt + response token ids.
        prompt_len: padded prompt length (positions < prompt_len are prompt).
        temperature: logit scaling — must match the sampling temperature.
        return_entropy: if True, also return per-token entropy [B, S-1].
        pad_token_id: token id used for left-padding. Used to build
                      the attention_mask for the forward pass.
        stop_token_id: which token marks "end of response" for the LOSS mask.
                       Defaults to pad_token_id if None — correct for
                       single-turn GSM8K where EOS == pad == stop.

                       For MULTI-TURN or TOOL-USE: EOS may appear mid-response
                       as a turn separator. In that case set stop_token_id to
                       the actual end-of-response delimiter, e.g.:
                           tokenizer.encode("</answer>")[0]
                       This way the mask includes the full multi-turn body and
                       only stops at the real end signal.

                       If stop_token_id never appears (max_tokens hit), ALL
                       response tokens are included — the model didn't choose
                       to stop, so every token carries signal.
        padding_side: "left" (only supported value for now).

    THEORY
    ------
    The trainer needs log-probs of the SAMPLED tokens under the CURRENT
    policy (call them "new log-probs"). It also needs the same quantity
    under the rollout policy ("old log-probs") for the importance ratio,
    and under the frozen reference ("ref log-probs") for KL.

    All three uses go through this same function. The caller picks which
    model to pass in.

    TWO MASKS, TWO PURPOSES
    -----------------------
    `attention_mask` and `mask` are different things:

      attention_mask: tells the transformer "which positions exist vs pad."
        Passed to model(full_ids, attention_mask=...). Without it, the model
        attends to pad tokens and produces garbage logits. Built from
        pad_token_id. Operates INSIDE the forward pass.

      mask (response mask): tells the loss "which token log-probs count."
        Even on positions where the model produced valid logits, we zero out
        prompt tokens and everything after the first stop token. Built from
        stop_token_id. Operates OUTSIDE the forward pass, on the loss.

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
      forward, vLLM, gradients). Saves ~Vx memory on the [B, S-1, V] tensor
      and avoids an extra cast vs. fp32. Importance ratios and KL still
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
    - For the RESPONSE MASK, use stop_token_id (not pad_token_id):
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
    token_log_prob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Loss mask: which target positions count toward PG/KL.
    # Targets are shifted by 1 — target index t corresponds to full_ids[t+1].
    # Response target range (inclusive): [prompt_len-1, prompt_len-1 + resp_len - 1].
    start = max(prompt_len - 1, 0)
    mask = torch.zeros_like(token_log_prob)
    if response_lengths is not None:
        resp_lens = response_lengths.to(device).long().unsqueeze(1)            # (B, 1)
        pos = torch.arange(S - 1, device=device).unsqueeze(0)                   # (1, S-1)
        in_response = (pos >= start) & (pos < start + resp_lens)
        mask = in_response
    else:
        # Fallback: stop-token heuristic. Correct when EOS genuinely terminates the
        # sequence; off-by-one (includes first right-pad) for max_tokens-truncated
        # sequences where pad_id == eos_id. Prefer passing response_lengths.
        stop_id = stop_token_id if stop_token_id else pad_token_id
        is_eos = targets == stop_id  # (B, S-1)
        is_eos[:, :start] = False
        no_prior_eos = (is_eos[:, start:].cumsum(dim=-1) - is_eos[:, start:].long()) == 0
        mask[:, start:] = no_prior_eos.float()

    entropy = None
    if return_entropy:
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
        KL_k1 ≈ E_{x~policy}[ log policy(x) - log ref(x) ] = E[log_r * -1]

        Shape used here (per-token): `kl_per_token = -(ref_logp - policy_logp)`
                                    = `policy_logp - ref_logp`

        Properties:
          + Simplest possible estimator, one line of math.
          + Unbiased in expectation (that's the definition of KL).
          - HIGH VARIANCE. A single outlier token with ratio ~0.01 or ~100
            dominates the mean.
          - Can go NEGATIVE on any finite sample. Looks wrong on a plot
            even though the expectation is correct. Newcomers get confused
            and start "fixing" it with abs().
          - Bad signal for adaptive KL control (noisy -> LR oscillates).

    k3 — SCHULMAN UNBIASED
        KL_k3 ≈ E[ r - log(r) - 1 ]   where r = exp(ref_logp - policy_logp)

        Properties:
          + ALWAYS >= 0 (r - log(r) - 1 has minimum 0 at r=1).
          + Unbiased (same expectation as k1, different finite-sample
            distribution).
          + Lower variance than k1 in practice.
          + Stable signal for adaptive KL / LR control.
          - Slightly more compute than k1 (one exp + one add).

        See John Schulman's blog post "Approximating KL Divergence"
        (http://joschu.net/blog/kl-approx.html) for the derivation and
        variance analysis. This is what GRPO/DAPO/etc. actually use.

    DEFAULT: k3. Use k1 only for debugging / comparison / when you want
    to confirm your k3 implementation matches the naive version in
    expectation (run both over many steps, compare means — they should
    agree to within ~sampling noise).

    HINTS
    -----
    Both branches reduce to: `(per_token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)`

    - k1:   per_token_kl = policy_logp - ref_logp
    - k3:   log_r = ref_logp - policy_logp
            r = torch.exp(log_r)
            per_token_kl = r - log_r - 1

    GOTCHA
    ------
    For k3, clamp log_r before exp() if you're seeing nan/inf in training:
        log_r = (ref_logp - policy_logp).clamp(-10, 10)
    `rl_step` clamps the POLICY ratio to [-5, 5] but that's a different
    ratio (policy/old, not ref/policy). For the KL ratio, -10/+10
    is looser and sufficient for stability without distorting typical values.
    """
    if estimator == "k1":
        per_token_kl = policy_logp - ref_logp
    elif estimator == "k3":
        log_r = (ref_logp - policy_logp).clamp(-10, 10)
        r = torch.exp(log_r)
        per_token_kl = r - log_r - 1.0
    else:
        raise ValueError(f"unknown KL estimator: {estimator!r}")

    return (per_token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


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
) -> torch.Tensor:
    """Policy gradient loss for the configured variant.

    Args:
        cfg: RLConfig — `loss_type`, `clip_low`, `clip_high`, `clip_ratio`,
             `clip_cispo`, `dg_eta`, `max_new_tokens` are all read.
        policy_logp: [B*G, S-1] under current policy (with grad).
        old_logp: [B*G, S-1] under rollout policy (no grad).
        advantages: [B*G] (no grad).
        mask: [B*G, S-1] response token mask.
        token_count: [B*G] number of unmasked tokens per sequence.

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
                   weight and multiply policy_logp directly. Numerator is
                   ratio.clamp(max=clip_cispo).

    - "dg"       : Delight-gated PG. Compute "delight" = eta * adv * surprisal
                   (surprisal = -policy_logp.detach()), pass through sigmoid
                   to get a per-token gate, multiply against the policy
                   gradient term.

    - "dg_cispo" : DG and CISPO combined. Use the clamped importance weight
                   from CISPO AND the sigmoid gate from DG.

    GOTCHAS
    -------
    - Clamp log_ratio to [-5, 5] before exp() to avoid overflow on outliers
      (this matters at temperature > 0.7).
    - Advantages must be unsqueezed for broadcasting: a = advantages.unsqueeze(1)
    - Don't forget the negative sign on the final loss (we're maximizing).

    """
    if cfg.loss_type == "rloo":
        mean_logp = (policy_logp * mask).sum(dim=1) / token_count  # (B*G,)
        loss = -(advantages * mean_logp).mean()
    # sequence-level ratio
    elif cfg.loss_type == "gspo":
        seq_logp = (policy_logp * mask).sum(dim=1) / token_count  # (B*G,)
        old_seq_logp = (old_logp * mask).sum(dim=1) / token_count
        seq_log_ratio = (seq_logp - old_seq_logp).clamp(-5, 5)  # clamp for stability
        seq_ratio = torch.exp(seq_log_ratio)  # (B*G,)
        # we support asymmetric clipping for GSPO as well
        seq_clipped = torch.clamp(seq_ratio, 1.0 - cfg.clip_low, 1.0 + cfg.clip_high)
        loss = -torch.min(advantages * seq_ratio, advantages * seq_clipped).mean()
    # all token-level variants
    else:
        log_ratio = (policy_logp - old_logp).clamp(-5, 5)  # clamp for stability
        ratio = torch.exp(log_ratio)  # (B*G, S-1)
        clipped = torch.clamp(ratio, 1 - cfg.clip_low, 1 + cfg.clip_high)
        a = advantages.unsqueeze(1)  # (B*G, 1)
        obj = torch.min(ratio * a, clipped * a)
        # token-level normalization
        if cfg.loss_type == "dapo":
            loss = -(obj * mask).sum() / mask.sum().clamp(min=1.0)
        elif cfg.loss_type == "grpo":
            per_seq = (obj * mask).sum(dim=1) / token_count
            loss = -per_seq.mean()
        elif cfg.loss_type == "dr_grpo":
            per_seq = (obj * mask).sum(dim=1) / cfg.max_new_tokens
            loss = -per_seq.mean()
        elif cfg.loss_type == "cispo":
            raise NotImplementedError("cispo loss not yet implemented")
        elif cfg.loss_type == "dg":
            raise NotImplementedError("dg loss not yet implemented")
        elif cfg.loss_type == "dg_cispo":
            raise NotImplementedError("dg_cispo loss not yet implemented")
        else:
            raise ValueError(f"unknown loss: {cfg.loss_type!r}")

    return loss


# =============================================================================
# 5. FULL RL STEP — collect, multi-epoch optimize, adaptive LR
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
        micro_batches: list of pre-collected micro-batches. Each is a dict with
                       keys: full_ids, plen, adv, old_logp, ref_logp, mask,
                       token_count, responses, rewards. Built by the trainer's
                       rollout_phase.
        model: trainable policy (already DDP-wrapped if distributed).
        ref_model: frozen reference (LoRA: base model with disable_adapter;
                   full FT: separate frozen copy).
        optimizer: AdamW or similar, already constructed.
        stats: TrainingStats instance to log into.
        step: current step number (for logging cadence).
        kl_ema: rolling KL EMA from previous step (for adaptive LR).

    Returns:
        (metrics_dict, new_kl_ema). metrics_dict has loss/reward/kl/clip_frac.

    PHASES
    ------
    1. Multi-epoch optimization over the collected micro-batches.
       Loop `cfg.optim_epochs` times (1 for RLOO). Each epoch:
         a. zero_grad
         b. for each micro-batch: forward pass through model -> policy_logp,
            compute_kl, compute_loss, scale by 1/len(micro_batches), backward
         c. (only on the LAST epoch) accumulate clip stats, entropy, etc.
         d. clip_grad_norm_ + optimizer.step

    2. Adaptive LR (if cfg.use_adaptive_lr):
       - Update kl_ema with the per-step mean KL (capped at kl_target * 10).
       - If kl_ema > kl_target * kl_factor: shrink LR (keep above min_lr).
       - If kl_ema < kl_target / kl_factor: grow LR (keep below max_lr).

    3. Stats logging (push into TrainingStats).

    GOTCHAS
    -------
    - The forward pass MUST be inside the inner loop (per micro-batch),
      not amortised across epochs. Multi-epoch means re-doing the forward
      with the latest weights each epoch.
    - Loss is divided by len(micro_batches) so that .backward() correctly
      averages across grad accumulation.
    - clip_frac counting differs between sequence-level (GSPO) and
      token-level (everything else) — see the branching below.
    - For RLOO, optim_epochs=1 should be set by the config, should add a warning
      (the trainer should do this).

    """
    model.train()
    n = len(micro_batches)
    n_epochs = cfg.optim_epochs  # RLOO must have optim_epochs=1 — enforced by validate_rl_config

    # Stats accumulators — only populated on the last epoch
    tot_kl = tot_pg = tot_kl_loss = tot_ent = 0.0
    tot_clip = tot_tok = 0.0

    for epoch in range(n_epochs):
        optimizer.zero_grad(set_to_none=True)
        last = (epoch == n_epochs - 1)

        for mb in micro_batches:
            policy_logp, _, entropy = compute_token_logprobs(
                model, mb["full_ids"], mb["plen"], cfg.temperature,
                return_entropy=last,
                pad_token_id=mb.get("pad_token_id"),
                stop_token_id=mb.get("stop_token_id"),
                response_lengths=mb.get("response_lengths"),
            )
            kl = compute_kl(policy_logp, mb["ref_logp"], mb["mask"])
            pg_loss = compute_loss(
                cfg, policy_logp, mb["old_logp"],
                mb["adv"], mb["mask"], mb["token_count"],
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

                    # Clip fraction — differs between sequence-level and token-level
                    if cfg.loss_type == "gspo":
                        seq_lp = (policy_logp * mb["mask"]).sum(dim=1) / mb["token_count"]
                        old_seq_lp = (mb["old_logp"] * mb["mask"]).sum(dim=1) / mb["token_count"]
                        ratio = torch.exp((seq_lp - old_seq_lp).clamp(-5, 5))
                        tot_clip += ((ratio - 1.0).abs() > cfg.clip_high).float().sum()
                        tot_tok += float(len(ratio))
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
    with record_function("metrics_dict"):
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
        }

    if cfg.use_adaptive_lr:
        pass  # skipped for now — add later

    return metrics, kl_ema
