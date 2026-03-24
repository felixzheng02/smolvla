"""Dataset loading, analysis, train/val splitting, and episode subsetting."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetStats:
    """Summary statistics for a LeRobot dataset."""

    repo_id: str
    total_episodes: int
    total_frames: int
    fps: int | None
    state_dim: int
    action_dim: int
    image_keys: list[str]
    image_shapes: dict[str, tuple[int, ...]]
    tasks: dict[str, int]  # task_description -> episode_count
    episodes_per_task: dict[str, list[int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.repo_id}",
            f"Episodes: {self.total_episodes}",
            f"Frames: {self.total_frames}",
            f"FPS: {self.fps}",
            f"State dim: {self.state_dim}",
            f"Action dim: {self.action_dim}",
            f"Cameras: {self.image_keys}",
            f"Image shapes: {self.image_shapes}",
            f"Tasks ({len(self.tasks)}):",
        ]
        for task, count in self.tasks.items():
            lines.append(f"  - [{count} eps] {task}")
        return "\n".join(lines)


def compute_stats(
    meta: dict[str, Any],
    episode_tasks: dict[int, str],
) -> DatasetStats:
    """Compute dataset stats from metadata and episode-task mapping.

    Args:
        meta: Dataset metadata dict (from info.json or dataset.meta).
            Expected keys: repo_id, total_episodes, total_frames, fps,
            features (with observation.state, action, observation.images.*).
        episode_tasks: Mapping of episode_index -> language_instruction/task string.
    """
    features = meta.get("features", {})

    # Extract state/action dims from feature shapes
    state_dim = _get_feature_dim(features, "observation.state")
    action_dim = _get_feature_dim(features, "action")

    # Find image keys and shapes
    image_keys = sorted(
        k for k in features if k.startswith("observation.images.")
    )
    image_shapes = {}
    for k in image_keys:
        shape = features[k].get("shape") or features[k].get("shapes", {}).get("camera", ())
        image_shapes[k] = tuple(shape) if shape else ()

    # Count episodes per task
    task_counts: Counter[str] = Counter(episode_tasks.values())

    # Group episodes by task
    episodes_per_task: dict[str, list[int]] = {}
    for ep_idx, task in sorted(episode_tasks.items()):
        episodes_per_task.setdefault(task, []).append(ep_idx)

    return DatasetStats(
        repo_id=meta.get("repo_id", "unknown"),
        total_episodes=meta.get("total_episodes", len(episode_tasks)),
        total_frames=meta.get("total_frames", 0),
        fps=meta.get("fps"),
        state_dim=state_dim,
        action_dim=action_dim,
        image_keys=image_keys,
        image_shapes=image_shapes,
        tasks=dict(task_counts.most_common()),
        episodes_per_task=episodes_per_task,
    )


def _get_feature_dim(features: dict, key: str) -> int:
    """Extract dimension from a feature spec (shape field or dtype)."""
    feat = features.get(key, {})
    shape = feat.get("shape")
    if shape:
        # shape is e.g. [7] for action or [8] for state
        return int(shape[0]) if len(shape) == 1 else int(np.prod(shape))
    return 0


def stratified_split(
    episodes_per_task: dict[str, list[int]],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split episodes into train/val sets, stratified by task.

    Ensures at least 1 val episode per task (if task has ≥2 episodes).
    Returns sorted (train_indices, val_indices).
    """
    rng = np.random.default_rng(seed)
    train_eps: list[int] = []
    val_eps: list[int] = []

    for _task, episodes in sorted(episodes_per_task.items()):
        eps = np.array(episodes)
        rng.shuffle(eps)
        n_val = max(1, int(len(eps) * val_fraction))
        if len(eps) < 2:
            # Too few episodes — put in train only
            train_eps.extend(eps.tolist())
        else:
            val_eps.extend(eps[:n_val].tolist())
            train_eps.extend(eps[n_val:].tolist())

    return sorted(train_eps), sorted(val_eps)


def subset_episodes(
    episode_indices: list[int],
    fraction: float,
    seed: int = 42,
) -> list[int]:
    """Randomly sample a fraction of episode indices."""
    if fraction >= 1.0:
        return episode_indices
    rng = np.random.default_rng(seed)
    n = max(1, int(len(episode_indices) * fraction))
    chosen = rng.choice(episode_indices, size=n, replace=False)
    return sorted(chosen.tolist())


def save_split(
    train_eps: list[int],
    val_eps: list[int],
    path: str | Path,
) -> None:
    """Save train/val split to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": train_eps, "val": val_eps}, indent=2))


def load_split(path: str | Path) -> tuple[list[int], list[int]]:
    """Load train/val split from JSON."""
    data = json.loads(Path(path).read_text())
    return data["train"], data["val"]
