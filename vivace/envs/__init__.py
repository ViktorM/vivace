"""Env registry — name → class lookup for `cfg.env_name` / `cfg.eval_envs`.

When adding a new env: import its class here and add a row to `ENV_REGISTRY`.
The trainer dispatches purely on the registry, so configs can reference envs
by string name without import gymnastics.
"""

from __future__ import annotations

from vivace.envs.aime import AIME2024Env, AIME2025Env
from vivace.envs.base import Env, Example
from vivace.envs.gsm8k import GSM8KEnv
from vivace.envs.math import MATHEnv
from vivace.envs.math500 import MATH500Env

ENV_REGISTRY: dict[str, type[Env]] = {
    "gsm8k":   GSM8KEnv,
    "math":    MATHEnv,
    "math500": MATH500Env,
    "aime24":  AIME2024Env,
    "aime25":  AIME2025Env,
}

# Per-env preset kwargs. Lets a yaml reference a variant by name without
# wiring a full kwargs dict through the trainer config. Keep this small —
# new variants belong here only when they're worth a stable name.
ENV_PRESETS: dict[str, dict] = {
    # 1.5B-Instruct hits format=99% in <50 steps, then accuracy plateaus / regresses
    # because the format reward dominates the gradient. This preset rebalances:
    # correctness 5x default, format weights crushed, so the gradient targets
    # answer-correctness specifically.
    "math_strict": dict(
        corpus="hendrycks",
        reward_overrides={
            "correct_bonus": 5.0,
            "strict_format_bonus": 0.05,
            "soft_format_bonus": 0.05,
            "xmlcount_max": 0.05,
        },
    ),
}


def make_env(name: str, **kwargs) -> Env:
    """Instantiate an env by registry name.

    Resolves `name` first against `ENV_PRESETS` (a config-only alias mapping to
    a class + preset kwargs), then falls back to `ENV_REGISTRY` (raw class).
    Caller-supplied **kwargs override the preset.
    """
    if name in ENV_PRESETS:
        preset = ENV_PRESETS[name]
        # All current presets target MATHEnv; if that changes, store cls per preset.
        return MATHEnv(**{**preset, **kwargs})
    if name not in ENV_REGISTRY:
        raise ValueError(
            f"unknown env {name!r}; known: {sorted(set(ENV_REGISTRY) | set(ENV_PRESETS))}"
        )
    return ENV_REGISTRY[name](**kwargs)


def register_env(name: str, cls: type[Env], *, overwrite: bool = False) -> None:
    """Register a custom env class so it's reachable by `cfg.env_name=<name>`.

    Call this once at import time from your own package before launching
    a training run. The trainer's name → class lookup will find it.

    >>> from vivace.envs import register_env
    >>> from my_lib import MyEnv
    >>> register_env("my_env", MyEnv)
    """
    if not overwrite and name in ENV_REGISTRY:
        raise ValueError(
            f"env {name!r} is already registered as {ENV_REGISTRY[name].__name__}; "
            "pass overwrite=True to replace it"
        )
    ENV_REGISTRY[name] = cls


__all__ = ["Env", "Example", "ENV_REGISTRY", "make_env", "register_env"]
