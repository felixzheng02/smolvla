"""Tests for src/dataset.py — stats, splitting, subsetting."""

import json

import numpy as np
import pytest

from src.dataset import (
    DatasetStats,
    compute_stats,
    load_split,
    save_split,
    stratified_split,
    subset_episodes,
)

# -- Fixtures: mock metadata resembling lerobot/libero_10_image --


@pytest.fixture
def mock_meta() -> dict:
    return {
        "repo_id": "lerobot/libero_10_image",
        "total_episodes": 379,
        "total_frames": 101469,
        "fps": 10,
        "features": {
            "observation.state": {"shape": [8], "dtype": "float32"},
            "action": {"shape": [7], "dtype": "float32"},
            "observation.images.image": {"shape": [256, 256, 3], "dtype": "uint8"},
            "observation.images.wrist_image": {"shape": [256, 256, 3], "dtype": "uint8"},
        },
    }


@pytest.fixture
def mock_episode_tasks() -> dict[int, str]:
    """10 tasks, ~38 episodes each."""
    tasks = {}
    task_names = [f"task_{i}: do something {i}" for i in range(10)]
    for ep in range(379):
        tasks[ep] = task_names[ep % 10]
    return tasks


@pytest.fixture
def mock_episodes_per_task(mock_episode_tasks: dict) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for ep, task in sorted(mock_episode_tasks.items()):
        result.setdefault(task, []).append(ep)
    return result


class TestComputeStats:
    def test_basic_fields(self, mock_meta, mock_episode_tasks):
        stats = compute_stats(mock_meta, mock_episode_tasks)
        assert stats.repo_id == "lerobot/libero_10_image"
        assert stats.total_episodes == 379
        assert stats.total_frames == 101469
        assert stats.fps == 10

    def test_dimensions(self, mock_meta, mock_episode_tasks):
        stats = compute_stats(mock_meta, mock_episode_tasks)
        assert stats.state_dim == 8
        assert stats.action_dim == 7

    def test_image_keys(self, mock_meta, mock_episode_tasks):
        stats = compute_stats(mock_meta, mock_episode_tasks)
        assert "observation.images.image" in stats.image_keys
        assert "observation.images.wrist_image" in stats.image_keys
        assert stats.image_shapes["observation.images.image"] == (256, 256, 3)

    def test_task_counts(self, mock_meta, mock_episode_tasks):
        stats = compute_stats(mock_meta, mock_episode_tasks)
        assert len(stats.tasks) == 10
        # Each task should have ~38 episodes (379 / 10)
        for count in stats.tasks.values():
            assert 37 <= count <= 38

    def test_summary_string(self, mock_meta, mock_episode_tasks):
        stats = compute_stats(mock_meta, mock_episode_tasks)
        s = stats.summary()
        assert "379" in s
        assert "101469" in s
        assert "State dim: 8" in s


class TestStratifiedSplit:
    def test_no_overlap(self, mock_episodes_per_task):
        train, val = stratified_split(mock_episodes_per_task, val_fraction=0.1)
        assert set(train).isdisjoint(set(val))

    def test_complete_coverage(self, mock_episodes_per_task):
        train, val = stratified_split(mock_episodes_per_task, val_fraction=0.1)
        all_eps = set()
        for eps in mock_episodes_per_task.values():
            all_eps.update(eps)
        assert set(train) | set(val) == all_eps

    def test_val_fraction_approximate(self, mock_episodes_per_task):
        train, val = stratified_split(mock_episodes_per_task, val_fraction=0.2)
        total = len(train) + len(val)
        # Each task contributes ~20% to val
        assert 0.15 < len(val) / total < 0.30

    def test_at_least_one_val_per_task(self, mock_episodes_per_task):
        train, val = stratified_split(mock_episodes_per_task, val_fraction=0.05)
        val_set = set(val)
        # Each task with ≥2 eps should have at least 1 in val
        for task, eps in mock_episodes_per_task.items():
            if len(eps) >= 2:
                assert any(e in val_set for e in eps), f"No val episode for {task}"

    def test_deterministic(self, mock_episodes_per_task):
        t1, v1 = stratified_split(mock_episodes_per_task, seed=42)
        t2, v2 = stratified_split(mock_episodes_per_task, seed=42)
        assert t1 == t2 and v1 == v2

    def test_different_seed_gives_different_split(self, mock_episodes_per_task):
        _, v1 = stratified_split(mock_episodes_per_task, seed=42)
        _, v2 = stratified_split(mock_episodes_per_task, seed=99)
        assert v1 != v2

    def test_single_episode_task(self):
        """Task with 1 episode goes to train."""
        eps_per_task = {"single": [0], "normal": [1, 2, 3, 4, 5]}
        train, val = stratified_split(eps_per_task, val_fraction=0.2)
        assert 0 in train
        assert 0 not in val


class TestSubsetEpisodes:
    def test_full_fraction(self):
        eps = list(range(100))
        assert subset_episodes(eps, 1.0) == eps

    def test_half_fraction(self):
        eps = list(range(100))
        result = subset_episodes(eps, 0.5, seed=42)
        assert len(result) == 50
        assert all(e in eps for e in result)

    def test_deterministic(self):
        eps = list(range(100))
        r1 = subset_episodes(eps, 0.25, seed=42)
        r2 = subset_episodes(eps, 0.25, seed=42)
        assert r1 == r2

    def test_minimum_one(self):
        eps = [0, 1, 2]
        result = subset_episodes(eps, 0.01)
        assert len(result) >= 1


class TestSaveLoadSplit:
    def test_roundtrip(self, tmp_path):
        train = [0, 2, 4, 6]
        val = [1, 3, 5]
        path = tmp_path / "split.json"
        save_split(train, val, path)
        loaded_train, loaded_val = load_split(path)
        assert loaded_train == train
        assert loaded_val == val

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "split.json"
        save_split([1], [2], path)
        assert path.exists()
