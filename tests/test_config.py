"""Tests for src/config.py — YAML loading, merging, CLI arg generation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    build_train_command,
    config_to_cli_args,
    load_config,
    load_yaml,
    merge_configs,
)


@pytest.fixture
def base_yaml(tmp_path: Path) -> Path:
    data = {
        "policy": {"path": "lerobot/smolvla_base", "optimizer_lr": 1e-3},
        "peft": {"method_type": "LORA", "r": 32},
        "batch_size": 8,
        "steps": 20000,
    }
    p = tmp_path / "base.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture
def override_yaml(tmp_path: Path) -> Path:
    data = {"peft": {"r": 64}, "steps": 5000}
    p = tmp_path / "override.yaml"
    p.write_text(yaml.dump(data))
    return p


class TestLoadYaml:
    def test_loads_valid_yaml(self, base_yaml: Path):
        config = load_yaml(base_yaml)
        assert config["batch_size"] == 8
        assert config["policy"]["path"] == "lerobot/smolvla_base"

    def test_empty_file_returns_empty_dict(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert load_yaml(p) == {}


class TestMergeConfigs:
    def test_shallow_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert merge_configs(base, override) == {"a": 1, "b": 99}

    def test_deep_merge(self):
        base = {"policy": {"path": "base", "lr": 1e-3}}
        override = {"policy": {"lr": 1e-4}}
        result = merge_configs(base, override)
        assert result == {"policy": {"path": "base", "lr": 1e-4}}

    def test_new_keys_added(self):
        base = {"a": 1}
        override = {"b": 2}
        assert merge_configs(base, override) == {"a": 1, "b": 2}

    def test_does_not_mutate_originals(self):
        base = {"policy": {"path": "orig"}}
        override = {"policy": {"path": "new"}}
        merge_configs(base, override)
        assert base["policy"]["path"] == "orig"


class TestLoadConfig:
    def test_base_only(self, base_yaml: Path):
        config = load_config(base_yaml)
        assert config["peft"]["r"] == 32

    def test_with_override(self, base_yaml: Path, override_yaml: Path):
        config = load_config(base_yaml, override_yaml)
        assert config["peft"]["r"] == 64  # overridden
        assert config["peft"]["method_type"] == "LORA"  # preserved from base
        assert config["steps"] == 5000  # overridden
        assert config["batch_size"] == 8  # preserved from base


class TestConfigToCliArgs:
    def test_flat_values(self):
        config = {"batch_size": 8, "steps": 20000}
        args = config_to_cli_args(config)
        assert "--batch_size=8" in args
        assert "--steps=20000" in args

    def test_nested_values(self):
        config = {"policy": {"path": "lerobot/smolvla_base"}}
        args = config_to_cli_args(config)
        assert "--policy.path=lerobot/smolvla_base" in args

    def test_boolean_values(self):
        config = {"wandb": {"enable": True}}
        args = config_to_cli_args(config)
        assert "--wandb.enable=true" in args

    def test_none_becomes_null(self):
        config = {"policy": {"output_features": None}}
        args = config_to_cli_args(config)
        assert "--policy.output_features=null" in args

    def test_list_values(self):
        config = {"targets": ["q_proj", "v_proj"]}
        args = config_to_cli_args(config)
        assert "--targets=q_proj,v_proj" in args


class TestBuildTrainCommand:
    def test_starts_with_lerobot_train(self):
        cmd = build_train_command({"batch_size": 8})
        assert cmd[0] == "lerobot-train"
        assert "--batch_size=8" in cmd

    def test_full_config(self, base_yaml: Path):
        config = load_config(base_yaml)
        cmd = build_train_command(config)
        assert cmd[0] == "lerobot-train"
        assert "--policy.path=lerobot/smolvla_base" in cmd
        assert "--peft.r=32" in cmd
