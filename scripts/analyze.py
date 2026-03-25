#!/usr/bin/env python3
"""Part 1: Dataset analysis — load, compute stats, create train/val split, export episode videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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


def save_episode_video(dataset, episode_index: int, output_path: Path, fps: int = 10) -> None:
    """Export one episode as a side-by-side video (front + wrist cameras).

    Uses imageio to write MP4. Each frame shows both camera views side by side
    with episode/frame info overlaid.
    """
    import imageio.v3 as iio

    # Find frame range for this episode
    ep_mask = dataset.hf_dataset["episode_index"]
    indices = [i for i, ep in enumerate(ep_mask) if ep == episode_index]
    if not indices:
        print(f"Episode {episode_index} not found")
        return

    # Get task description
    ep_meta = dataset.meta.episodes[episode_index]
    task_list = ep_meta.get("tasks", [])
    task_desc = task_list[0] if task_list else f"episode {episode_index}"

    print(f"Exporting episode {episode_index}: {task_desc} ({len(indices)} frames)")

    frames = []
    for idx in indices:
        item = dataset[idx]
        # Images come as tensors (C, H, W) in [0, 1]
        front = item["observation.images.image"]
        wrist = item["observation.images.wrist_image"]

        # Convert to numpy uint8 (H, W, C)
        if hasattr(front, "numpy"):
            front = front.permute(1, 2, 0).numpy()
            wrist = wrist.permute(1, 2, 0).numpy()
        front = (np.clip(front, 0, 1) * 255).astype(np.uint8)
        wrist = (np.clip(wrist, 0, 1) * 255).astype(np.uint8)

        # Side by side
        frame = np.concatenate([front, wrist], axis=1)
        frames.append(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, np.stack(frames), fps=fps, codec="h264")
    print(f"Saved {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


def main():
    """Load a LeRobot dataset, print summary statistics, and save a stratified train/val split."""
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
    # Video export options
    parser.add_argument("--save-video", type=int, nargs="*", default=None,
                        metavar="EPISODE",
                        help="Export episode videos. No args = one per task. "
                             "Specify episode indices to export specific ones.")
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

    # Export episode videos
    if args.save_video is not None:
        fps = meta.get("fps", 10) or 10
        video_dir = out / "videos" / "dataset"
        if args.save_video:
            # Specific episodes requested
            episodes = args.save_video
        else:
            # One episode per task (first episode of each)
            episodes = [eps[0] for eps in stats.episodes_per_task.values()]

        for ep_idx in episodes:
            task = episode_tasks.get(ep_idx, "unknown")
            safe_task = task[:60].replace(" ", "_").replace("/", "_")
            video_path = video_dir / f"ep{ep_idx:03d}_{safe_task}.mp4"
            save_episode_video(dataset, ep_idx, video_path, fps=fps)


if __name__ == "__main__":
    main()
