"""Wrapper around lerobot-train that loads YAML configs and applies CLI overrides."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config import build_train_command, load_config, merge_configs


def parse_cli_overrides(args: list[str]) -> tuple[str, str | None, dict[str, Any]]:
    """Parse CLI arguments into base config path, override path, and key=value overrides.

    Returns (base_path, override_path, overrides_dict) where overrides_dict
    uses nested dicts for dot-separated keys (e.g. --policy.lr=1e-3).
    """
    base_path = "configs/base.yaml"
    override_path: str | None = None
    overrides: dict[str, Any] = {}

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--config" and i + 1 < len(args):
            base_path = args[i + 1]
            i += 2
        elif arg.startswith("--config="):
            base_path = arg.split("=", 1)[1]
            i += 1
        elif arg == "--override" and i + 1 < len(args):
            override_path = args[i + 1]
            i += 2
        elif arg.startswith("--override="):
            override_path = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            # Convert value to appropriate type
            value = _cast_value(value)
            # Build nested dict from dot-separated key
            _set_nested(overrides, key, value)
            i += 1
        else:
            i += 1

    return base_path, override_path, overrides


def _cast_value(value: str) -> Any:
    """Cast a string CLI value to the appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated key."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _resolve_episode_fraction(config: dict[str, Any]) -> dict[str, Any]:
    """Convert dataset.episode_fraction to explicit episode list for lerobot-train.

    If dataset.episode_fraction is set (e.g., 0.25), loads dataset metadata to
    compute a stratified subset of episodes, then replaces it with dataset.episodes.
    """
    ds = config.get("dataset", {})
    fraction = ds.pop("episode_fraction", None)
    if fraction is None:
        return config

    import random

    from src.dataset import stratified_split

    # Load episode-task mapping from the dataset
    repo_id = ds.get("repo_id", "lerobot/libero_10_image")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id)
        # Build episodes_per_task mapping from HF Dataset episodes
        episodes_per_task: dict[str, list[int]] = {}
        for i in range(len(dataset.meta.episodes)):
            ep = dataset.meta.episodes[i]
            ep_idx = ep["episode_index"]
            task_list = ep.get("tasks", [])
            task = task_list[0] if task_list else f"task_{ep_idx}"
            episodes_per_task.setdefault(task, []).append(int(ep_idx))
    except Exception:
        # Fallback: assume 379 episodes, no stratification
        all_eps = list(range(379))
        random.seed(42)
        random.shuffle(all_eps)
        n = max(1, int(len(all_eps) * fraction))
        ds["episodes"] = sorted(all_eps[:n])
        return config

    # Use stratified split to get a representative subset
    # val_fraction = 1 - fraction gives us a train set of size fraction
    train_eps, _ = stratified_split(episodes_per_task, val_fraction=1.0 - fraction, seed=42)
    ds["episodes"] = sorted(train_eps)
    print(f"Data ablation: using {len(train_eps)}/{sum(len(v) for v in episodes_per_task.values())} "
          f"episodes ({fraction:.0%})")
    return config


def main(argv: list[str] | None = None) -> int:
    """Load config, apply overrides, and run lerobot-train."""
    if argv is None:
        argv = sys.argv[1:]

    base_path, override_path, overrides = parse_cli_overrides(argv)
    config = load_config(base_path, override_path)

    if overrides:
        config = merge_configs(config, overrides)

    # Handle episode_fraction -> episodes conversion
    config = _resolve_episode_fraction(config)

    cmd = build_train_command(config)
    print(" ".join(cmd))

    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
