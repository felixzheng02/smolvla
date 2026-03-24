#!/usr/bin/env python3
"""Part 1: Dataset analysis — load, compute stats, create train/val split."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_yaml
from src.dataset import compute_stats, save_split, stratified_split


def load_lerobot_dataset(repo_id: str):
    """Load a LeRobot dataset from HuggingFace Hub."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id)


def extract_meta(dataset) -> dict:
    """Extract metadata dict from a LeRobotDataset."""
    info = dataset.meta.info
    return {
        "repo_id": info.get("repo_id", str(dataset.repo_id)),
        "total_episodes": info.get("total_episodes", dataset.num_episodes),
        "total_frames": info.get("total_frames", dataset.num_frames),
        "fps": info.get("fps"),
        "features": info.get("features", {}),
    }


def extract_episode_tasks(dataset) -> dict[int, str]:
    """Map episode_index -> task string from dataset metadata.

    LeRobot v0.4+: meta.episodes is an HF Dataset with 'tasks' column (list[str]).
    meta.tasks is a pandas DataFrame (index=task_desc, column=task_index).
    """
    tasks_map = {}
    episodes = dataset.meta.episodes
    for i in range(len(episodes)):
        ep = episodes[i]
        ep_idx = ep["episode_index"]
        # 'tasks' column contains list of task descriptions
        task_list = ep.get("tasks", [])
        tasks_map[int(ep_idx)] = task_list[0] if task_list else f"task_{ep_idx}"
    return tasks_map


def main():
    parser = argparse.ArgumentParser(description="Analyze a LeRobot dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Config file to read dataset.repo_id from",
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Override dataset repo_id")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    # Resolve repo_id
    config = load_yaml(args.config)
    repo_id = args.repo_id or config.get("dataset", {}).get("repo_id", "lerobot/libero_10_image")

    print(f"Loading dataset: {repo_id}")
    dataset = load_lerobot_dataset(repo_id)

    # Compute and print stats
    meta = extract_meta(dataset)
    episode_tasks = extract_episode_tasks(dataset)
    stats = compute_stats(meta, episode_tasks)
    print("\n" + stats.summary())

    # Create and save train/val split
    train_eps, val_eps = stratified_split(
        stats.episodes_per_task,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"\nSplit: {len(train_eps)} train / {len(val_eps)} val episodes")

    out = Path(args.output_dir)
    split_path = out / "split.json"
    save_split(train_eps, val_eps, split_path)
    print(f"Saved split to {split_path}")


if __name__ == "__main__":
    main()
