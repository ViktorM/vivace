"""Plain dataclasses used across algos and the trainer.

These are deliberately data-only — no methods, no ABCs, no factories.

`RolloutBatch` is documentation only — `rollout_phase` hands `rl_step` plain
dicts and nothing constructs it. `RLConfig` has two string fields (`loss_type`,
`adv_type`) that drive composable dispatch in
`vivace/algos/policy_gradient.py`. `SFTConfig` is a small companion
for the optional SFT warmup path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """What a rollout produces. Keep this flat and explicit.

    Shapes:
        prompt_ids:      [B, P]            (padded prompts)
        response_ids:    [B, G, T]         (G = group_size, T = max_new_tokens)
        response_mask:   [B, G, T]         (1 for real tokens, 0 for padding)
        old_log_probs:   [B, G, T]         (log-probs under the rollout policy)
        ref_log_probs:   [B, G, T] or None (log-probs under frozen reference, optional)
        rewards:         [B, G]            (scalar reward per sequence)
        answers:         list[str]         (ground-truth answers, length B)
    """

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor | None
    rewards: torch.Tensor
    answers: list[str]


@dataclass
class RLConfig:
    """Composable RL hyperparameters.

    The two string fields below are CONFIG SWITCHES — `vivace/algos/policy_gradient.py`
    has if/elif ladders that dispatch on them. Composability comes from the
    fact that any `loss_type` × any `adv_type` is a valid combination.
    """

    # --- Algorithm switches ---
    loss_type: str = "grpo"   # grpo | dr_grpo | dapo | gspo | cispo | rloo (dg | dg_cispo: stubs)
    adv_type: str = "grpo"    # grpo (z-scored) | dr_grpo (no std) | rloo (leave-one-out)

    # --- Training ---
    steps: int = 100               # unread; the trainer runs TrainerConfig.num_steps
    batch_size: int = 1
    group_size: int = 8
    lr: float = 1e-6
    warmup_steps: int = 10
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    optim_epochs: int = 2          # RLOO: validate_rl_config forces 1

    # --- LR schedule (post-warmup cosine) ---
    eta_min_ratio: float = 0.2     # cosine floor as fraction of peak lr (peak * ratio = eta_min)
    lr_restart: bool = False       # True: cosine to mid, linear ramp eta_min→peak over `warmup_steps`, cosine again

    # --- AdamW (PyTorch defaults; paper recipes sometimes override) ---
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999      # MiniMax-M1 / CISPO paper uses 0.95
    adam_eps: float = 1e-8         # MiniMax-M1 / CISPO paper uses 1e-15

    # --- Adaptive LR — inert: rl_step's block is `pass` ---
    use_adaptive_lr: bool = False
    kl_target: float = 0.05
    kl_factor: float = 1.5         # how far from target triggers adjustment
    kl_ema: float = 0.2
    lr_factor: float = 1.5         # how much to scale LR
    min_lr: float = 1e-7
    max_lr: float = 2e-5

    # --- Generation ---
    max_new_tokens: int = 192        # max response length (HF generate / vLLM max_tokens)
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = -1                  # -1 = disabled; set e.g. 50 for top-k sampling

    # --- Clipping ---
    clip_low: float = 0.2          # epsilon_low
    clip_high: float = 0.2         # epsilon_high (set > clip_low for DAPO), but asymmetric can be used with other algorithms
    clip_ratio: float = 0.25       # unread by any loss (compute_loss uses clip_low/clip_high); kept because yamls set it
    clip_cispo_high: float = 5.0   # CISPO upper IS-weight cap. Paper tunes this side.
    clip_cispo_low: float = 0.0    # CISPO lower IS-weight floor. 0.0 disables lower clipping.
    # Canonical CISPO keeps all token gradients. Eq. 7 / PPO-style dropping is opt-in.
    cispo_use_token_mask: bool = False
    # token = previous DAPO-style token mean; sequence = equal weight per response;
    # hybrid = average of both, which is more robust to long negative samples.
    cispo_normalization: str = "hybrid"

    # --- KL ---
    kl_coef: float = 0.02          # KL regularization coefficient (some implementations call this kl_beta)
    adv_eps: float = 1e-4          # std floor of the grpo z-score

    # --- DG (delight-gating) — stub; dg_eta unread ---
    dg_eta: float = 1.0

    # --- Adaptive sampling (DAPO-style) ---
    # Default ON since the 0.5B/gsm8k pilots showed it's strictly helpful for
    # every RLOO-family algorithm (and required for GRPO to learn at all).
    # See docs/pilot_findings.md for the cross-algo evidence.
    adaptive_sampling: bool = True
    oversample_factor: float = 2.0
    reward_threshold: float = 1.8  # unread
    min_reward_spread: float = 0.5 # alive_rate metric only; the filter keeps the top-B groups by spread

    # --- Memory / numerics ---
    # Chunk entropy along the time axis in `compute_token_logprobs` to bound
    # the [B, T, V] temporaries to ~C/T of full size. 0 = single-shot.
    entropy_chunk_size: int = 64
    # True keeps entropy in the autograd graph (needed for entropy-bonus
    # loss variants). False uses a no_grad pre-alloc path for extra memory
    # savings — backprop through entropy then silently yields zero gradients.
    entropy_grad: bool = False

    # --- Logging (unread; trainer mirrors its own log_interval here) ---
    log_interval: int = 10
    preview_interval: int = 100

    @property
    def uses_clipping(self) -> bool:
        return self.loss_type != "rloo"


def validate_rl_config(cfg: RLConfig) -> None:
    """Warn+fix known-bad combos; raise on invalid CISPO settings."""
    import warnings

    if cfg.loss_type == "rloo" and cfg.optim_epochs != 1:
        warnings.warn(
            f"RLOO requires optim_epochs=1 (got {cfg.optim_epochs}). "
            "Multi-epoch optimization breaks REINFORCE's unbiasedness. Overriding."
        )
        cfg.optim_epochs = 1

    if cfg.adaptive_sampling and cfg.oversample_factor <= 1.0:
        warnings.warn(
            f"adaptive_sampling=True but oversample_factor={cfg.oversample_factor} <= 1.0. "
            "Nothing to filter. Setting adaptive_sampling=False."
        )
        cfg.adaptive_sampling = False

    if cfg.clip_cispo_low < 0.0:
        raise ValueError(f"clip_cispo_low must be >= 0.0, got {cfg.clip_cispo_low}")
    if cfg.clip_cispo_low > 1.0:
        raise ValueError(f"clip_cispo_low must be <= 1.0, got {cfg.clip_cispo_low}")
    if cfg.clip_cispo_high <= 0.0:
        raise ValueError(f"clip_cispo_high must be > 0.0, got {cfg.clip_cispo_high}")
    if cfg.clip_cispo_high < 1.0:
        raise ValueError(f"clip_cispo_high must be >= 1.0, got {cfg.clip_cispo_high}")
    if cfg.clip_cispo_low > cfg.clip_cispo_high:
        raise ValueError(
            f"clip_cispo_low must be <= clip_cispo_high, got "
            f"{cfg.clip_cispo_low} > {cfg.clip_cispo_high}"
        )

    if cfg.cispo_normalization not in {"token", "sequence", "hybrid"}:
        raise ValueError(
            f"cispo_normalization must be token | sequence | hybrid, "
            f"got {cfg.cispo_normalization!r}"
        )


# --- Preset factories — unused; values predate the v1 recipes ---

def grpo_config(**kw) -> RLConfig:
    return RLConfig(loss_type="grpo", adv_type="grpo", **kw)

def dr_grpo_config(**kw) -> RLConfig:
    return RLConfig(loss_type="dr_grpo", adv_type="dr_grpo", **kw)

def gspo_config(**kw) -> RLConfig:
    return RLConfig(loss_type="gspo", adv_type="grpo", lr=2e-6, **kw)

def dapo_config(**kw) -> RLConfig:
    return RLConfig(
        loss_type="dapo", adv_type="dr_grpo",
        clip_high=0.28, kl_coef=0.01,
        adaptive_sampling=True, **kw,
    )

def rloo_config(**kw) -> RLConfig:
    return RLConfig(
        loss_type="rloo", adv_type="rloo",
        optim_epochs=1, kl_coef=0.01, **kw,
    )

def cispo_config(**kw) -> RLConfig:
    return RLConfig(loss_type="cispo", adv_type="rloo", kl_coef=0.04, **kw)

def dg_config(**kw) -> RLConfig:
    return RLConfig(loss_type="dg", adv_type="rloo", kl_coef=0.02, **kw)

def dg_cispo_config(**kw) -> RLConfig:
    return RLConfig(
        loss_type="dg_cispo", adv_type="rloo",
        dg_eta=1.0, clip_cispo_high=5.0, kl_coef=0.04, **kw,
    )


@dataclass
class SFTConfig:
    """Settings for the optional SFT warmup path."""

    lr: float = 1e-5
    steps: int = 200
    batch_size: int = 4
    warmup_steps: int = 5
    grad_clip: float = 1.0
    n_examples: int = 2000
