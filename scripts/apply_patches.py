#!/usr/bin/env python3
"""Apply local patches to the installed lerobot package.

Adds per-action-dimension loss logging to SmolVLAPolicy.forward(),
which logs loss_action/{dx,dy,dz,drot1,drot2,drot3,gripper} to wandb
during training.

Usage:
    python scripts/apply_patches.py

Idempotent: safe to run multiple times.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PATCH_MARKER = "loss_action/"
ANCHOR_LINE = 'loss_dict["losses_after_rm_padding"]'

PATCH_BLOCK = '''
        # Per-action-dimension losses for wandb diagnostics
        _actual_dim = batch["action"].shape[-1]
        _per_dim = losses[:, :, :_actual_dim].mean(dim=(0, 1))
        _dim_names = ["dx", "dy", "dz", "drot1", "drot2", "drot3", "gripper"]
        for _i, _v in enumerate(_per_dim.tolist()):
            _name = _dim_names[_i] if _i < len(_dim_names) else f"dim_{_i}"
            loss_dict[f"loss_action/{_name}"] = _v
'''


def find_modeling_file() -> Path:
    """Locate the installed modeling_smolvla.py via importlib."""
    spec = importlib.util.find_spec("lerobot.policies.smolvla.modeling_smolvla")
    if spec is None or spec.origin is None:
        print("ERROR: Cannot find lerobot.policies.smolvla.modeling_smolvla")
        print("Is lerobot installed? Run: pip install 'lerobot[smolvla,peft]'")
        sys.exit(1)
    return Path(spec.origin)


def check_version() -> None:
    """Warn if lerobot version differs from the tested version."""
    try:
        import lerobot
        version = getattr(lerobot, "__version__", "unknown")
        if version != "0.4.4":
            print(f"WARNING: lerobot version is {version}, patch tested on 0.4.4")
    except ImportError:
        pass


def apply_patch(filepath: Path) -> bool:
    """Apply the per-action-dim loss patch. Returns True if patch was applied."""
    content = filepath.read_text()

    if PATCH_MARKER in content:
        print(f"Patch already applied: {filepath}")
        return False

    lines = content.split("\n")
    anchor_idx = None
    for i, line in enumerate(lines):
        if ANCHOR_LINE in line:
            anchor_idx = i
            break

    if anchor_idx is None:
        print(f"ERROR: Cannot find anchor line containing: {ANCHOR_LINE}")
        print(f"File: {filepath}")
        print("The lerobot version may be incompatible with this patch.")
        sys.exit(1)

    # Insert the patch block after the anchor line
    patch_lines = PATCH_BLOCK.rstrip("\n").split("\n")
    for j, pl in enumerate(patch_lines):
        lines.insert(anchor_idx + 1 + j, pl)

    filepath.write_text("\n".join(lines))

    # Verify
    if PATCH_MARKER not in filepath.read_text():
        print("ERROR: Patch verification failed")
        sys.exit(1)

    print(f"Patch applied: {filepath}")
    return True


def main() -> int:
    check_version()
    filepath = find_modeling_file()
    print(f"Target: {filepath}")
    apply_patch(filepath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
