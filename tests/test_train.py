"""Tests for scripts/train.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.train import _cast_value, _set_nested, main, parse_cli_overrides
from src.config import build_train_command, load_config


class TestParseCliOverrides:
    """Tests for CLI argument parsing."""

    def test_defaults(self):
        base, override, overrides = parse_cli_overrides([])
        assert base == "configs/base.yaml"
        assert override is None
        assert overrides == {}

    def test_config_flag(self):
        base, _, _ = parse_cli_overrides(["--config", "other.yaml"])
        assert base == "other.yaml"

    def test_config_equals(self):
        base, _, _ = parse_cli_overrides(["--config=other.yaml"])
        assert base == "other.yaml"

    def test_override_flag(self):
        _, override, _ = parse_cli_overrides(["--override", "ablation.yaml"])
        assert override == "ablation.yaml"

    def test_override_equals(self):
        _, override, _ = parse_cli_overrides(["--override=ablation.yaml"])
        assert override == "ablation.yaml"

    def test_key_value_simple(self):
        _, _, overrides = parse_cli_overrides(["--steps=5000"])
        assert overrides == {"steps": 5000}

    def test_key_value_dotted(self):
        _, _, overrides = parse_cli_overrides(["--policy.optimizer_lr=2e-4"])
        assert overrides == {"policy": {"optimizer_lr": 2e-4}}

    def test_multiple_overrides(self):
        _, _, overrides = parse_cli_overrides([
            "--steps=5000",
            "--batch_size=16",
            "--wandb.enable=false",
        ])
        assert overrides["steps"] == 5000
        assert overrides["batch_size"] == 16
        assert overrides["wandb"]["enable"] is False

    def test_combined_args(self):
        base, override, overrides = parse_cli_overrides([
            "--config", "custom.yaml",
            "--override", "ablation.yaml",
            "--steps=100",
        ])
        assert base == "custom.yaml"
        assert override == "ablation.yaml"
        assert overrides == {"steps": 100}


class TestCastValue:
    """Tests for value type casting."""

    def test_int(self):
        assert _cast_value("5000") == 5000

    def test_float(self):
        assert _cast_value("1e-3") == 1e-3

    def test_bool_true(self):
        assert _cast_value("true") is True

    def test_bool_false(self):
        assert _cast_value("false") is False

    def test_null(self):
        assert _cast_value("null") is None

    def test_string(self):
        assert _cast_value("some_string") == "some_string"


class TestSetNested:
    """Tests for nested dict construction from dotted keys."""

    def test_simple_key(self):
        d: dict = {}
        _set_nested(d, "steps", 100)
        assert d == {"steps": 100}

    def test_dotted_key(self):
        d: dict = {}
        _set_nested(d, "policy.lr", 0.001)
        assert d == {"policy": {"lr": 0.001}}

    def test_deep_dotted_key(self):
        d: dict = {}
        _set_nested(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}


class TestBuildTrainCommand:
    """Tests that build_train_command produces expected output from base config."""

    def test_base_config(self):
        config = load_config("configs/base.yaml")
        cmd = build_train_command(config)
        assert cmd[0] == "lerobot-train"
        assert "--policy.pretrained_path=lerobot/smolvla_base" in cmd
        assert "--steps=20000" in cmd
        assert "--batch_size=8" in cmd
        assert "--wandb.enable=true" in cmd

    def test_cli_overrides_applied(self):
        config = load_config("configs/base.yaml")
        # Simulate applying a CLI override
        from src.config import merge_configs
        overrides = {"steps": 5000, "batch_size": 16}
        config = merge_configs(config, overrides)
        cmd = build_train_command(config)
        assert "--steps=5000" in cmd
        assert "--batch_size=16" in cmd
        # Original values should be gone
        assert "--steps=20000" not in cmd
        assert "--batch_size=8" not in cmd


class TestMain:
    """Integration tests for main() with subprocess mocked."""

    @patch("scripts.train.subprocess.run")
    def test_main_runs_command(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0)
        exit_code = main([])
        assert exit_code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "lerobot-train"

    @patch("scripts.train.subprocess.run")
    def test_main_with_overrides(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0)
        exit_code = main(["--steps=5000"])
        assert exit_code == 0
        cmd = mock_run.call_args[0][0]
        assert "--steps=5000" in cmd
        assert "--steps=20000" not in cmd

    @patch("scripts.train.subprocess.run")
    def test_main_returns_exit_code(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=1)
        assert main([]) == 1

    @patch("scripts.train.subprocess.run")
    def test_main_nested_override(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0)
        main(["--policy.optimizer_lr=2e-4"])
        cmd = mock_run.call_args[0][0]
        assert "--policy.optimizer_lr=0.0002" in cmd
