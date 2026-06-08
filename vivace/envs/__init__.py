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


def make_env(name: str, **kwargs) -> Env:
    """Instantiate an env by registry name.

    Per-env kwargs (corpus, reward_overrides, n_eval, ...) come from the yaml
    config via `env_kwargs` / `eval_env_kwargs`. See `TrainerConfig`.
    """
    if name not in ENV_REGISTRY:
        raise ValueError(f"unknown env {name!r}; known: {sorted(ENV_REGISTRY)}")
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
