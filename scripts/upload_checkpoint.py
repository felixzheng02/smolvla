#!/usr/bin/env python3
"""Upload a SmolVLA checkpoint to HuggingFace Hub.

Uploads the pretrained_model/ directory (LoRA adapter weights + configs)
from a training checkpoint. Excludes training_state/ (optimizer, scheduler).

Usage:
    python scripts/upload_checkpoint.py --checkpoint outputs/train/base --repo-id user/smolvla-libero10
    python scripts/upload_checkpoint.py --checkpoint outputs/train/base/checkpoints/050000 --repo-id user/smolvla-libero10
    python scripts/upload_checkpoint.py --checkpoint outputs/train/base --repo-id user/smolvla-libero10 --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def resolve_pretrained_model(checkpoint_path: str) -> tuple[Path, str]:
    """Find the pretrained_model directory and step label from a checkpoint path.

    Returns (pretrained_model_path, step_label).
    """
    cp = Path(checkpoint_path)
    if not cp.exists():
        sys.exit(f"Error: path does not exist: {cp}")

    # Direct pretrained_model path
    if cp.name == "pretrained_model" and cp.is_dir():
        step = cp.parent.name
        return cp, step

    # Checkpoint dir containing pretrained_model/
    pm = cp / "pretrained_model"
    if pm.is_dir():
        step = cp.name
        return pm, step

    # Training output root — find last or highest checkpoint
    ckpts = cp / "checkpoints"
    if ckpts.is_dir():
        last = ckpts / "last" / "pretrained_model"
        if last.is_dir():
            real = (ckpts / "last").resolve()
            step = real.name if real.name.isdigit() else "last"
            return last, step
        numbered = sorted(
            [d for d in ckpts.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if numbered:
            pm = numbered[-1] / "pretrained_model"
            if pm.is_dir():
                return pm, numbered[-1].name

    sys.exit(
        f"Error: no pretrained_model found under {cp}\n"
        f"Expected structure: .../checkpoints/STEP/pretrained_model/"
    )


def main():
    parser = argparse.ArgumentParser(description="Upload SmolVLA checkpoint to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (dir or training output root)")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g., user/smolvla-libero10)")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--commit-message", type=str, default=None, help="Custom commit message")
    parser.add_argument("--revision", type=str, default="main", help="Branch to upload to (default: main)")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    pretrained_path, step_label = resolve_pretrained_model(args.checkpoint)

    print(f"Checkpoint:  {pretrained_path}")
    print(f"Step:        {step_label}")
    print(f"Repo:        {args.repo_id}")
    print(f"Private:     {args.private}")
    print()

    # List files to upload
    files = sorted(pretrained_path.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Files to upload ({len(files)}):")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(pretrained_path)}  ({size_kb:.1f} KB)")
    print()

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    commit_message = args.commit_message or f"Upload SmolVLA LoRA checkpoint (step {step_label})"

    url = api.upload_folder(
        folder_path=str(pretrained_path),
        repo_id=args.repo_id,
        commit_message=commit_message,
        revision=args.revision,
    )

    print(f"Uploaded to: https://huggingface.co/{args.repo_id}")
    print(f"Commit:      {url}")


if __name__ == "__main__":
    main()
