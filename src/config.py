"""Config loading, merging, and lerobot-train CLI argument generation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base. Override values win at leaf level."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    base_path: str | Path,
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load base config and optionally merge an override config on top."""
    config = load_yaml(base_path)
    if override_path is not None:
        override = load_yaml(override_path)
        config = merge_configs(config, override)
    return config


def _flatten(d: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dict into dot-separated key-value pairs."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten(v, key))
        else:
            items.append((key, v))
    return items


def config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert a config dict to lerobot-train CLI arguments.

    Nested dicts become dot-separated flags:
        {"policy": {"path": "x"}} -> ["--policy.path=x"]

    Booleans become true/false strings. None values are 'null'.
    Lists are formatted as JSON arrays (draccus format).
    """
    import json

    args: list[str] = []
    for key, value in _flatten(config):
        if isinstance(value, bool):
            args.append(f"--{key}={str(value).lower()}")
        elif isinstance(value, list):
            # draccus expects JSON array syntax: '[0,1,2]'
            args.append(f"--{key}={json.dumps(value)}")
        elif value is None:
            args.append(f"--{key}=null")
        else:
            args.append(f"--{key}={value}")
    return args


def build_train_command(config: dict[str, Any]) -> list[str]:
    """Build full lerobot-train command from config dict."""
    return ["lerobot-train"] + config_to_cli_args(config)
