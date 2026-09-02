"""--set typing and the allocator-mode peek in vivace/scripts/train.py."""
import sys

import pytest

from vivace.scripts.train import _apply_set_overrides, _peek_mode_from_argv, build_trainer_config, load_config

COLO = "vivace/configs/gsm8k/dapo_0.5b_colo.yaml"
DISAGG = "vivace/configs/gsm8k/dapo_2x4090.yaml"


def _cfg(*overrides):
    d = load_config(COLO)
    _apply_set_overrides(d, list(overrides))
    return build_trainer_config(d)


def test_set_keeps_strings_for_str_fields_and_wraps_scalars_for_list_fields():
    c = _cfg("wandb_group=off", "wandb_run_name=2026-09-01", "eval_envs=math500",
             "lora_target_modules=q_proj", "rl.lr=8.0e-5", "seed=7")
    assert c.wandb_group == "off" and c.wandb_run_name == "2026-09-01"   # yaml 1.1 would give False / date
    assert c.eval_envs == ["math500"] and c.lora_target_modules == ("q_proj",)  # not char-split
    assert c.rl.lr == 8e-5 and c.seed == 7


def test_set_list_literals_and_null_still_parse():
    c = _cfg("eval_envs=[gsm8k,math500]", "wandb_group=null")
    assert c.eval_envs == ["gsm8k", "math500"] and c.wandb_group is None


def test_peek_mode_honours_mode_flag_and_set(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--config", COLO, "--mode", "disaggregated"])
    assert _peek_mode_from_argv() == "disaggregated"
    monkeypatch.setattr(sys, "argv", ["train", "--config", DISAGG, "--set", "mode=colocated"])
    assert _peek_mode_from_argv() == "colocated"
    monkeypatch.setattr(sys, "argv", ["train", "--config", DISAGG])
    assert _peek_mode_from_argv() == "disaggregated"


def test_build_trainer_config_validates_rl_before_any_model_load():
    with pytest.raises(ValueError):
        _cfg("rl.group_size=1")
