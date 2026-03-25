#!/usr/bin/env python3
"""Wrapper that applies OOD observation patches before running lerobot-eval.

Called by scripts/evaluate.py for OOD evaluation modes. Reads OOD config
from the path in the OOD_CONFIG environment variable, monkey-patches the
relevant lerobot functions, then runs lerobot-eval with the remaining CLI args.

Supported modes:
    paraphrased: Replace task instructions with semantic paraphrases.
    visual: Add Gaussian noise and/or brightness shift to camera observations.
"""

from __future__ import annotations

import json
import os
import sys


def _apply_paraphrase_patch(paraphrases: dict[str, str]) -> None:
    """Monkey-patch add_envs_task to replace instructions with paraphrases."""
    import lerobot.envs.utils as env_utils

    _original = env_utils.add_envs_task

    def patched_add_envs_task(env, observation):
        observation = _original(env, observation)
        tasks = observation.get("task", [])
        observation["task"] = [paraphrases.get(t, t) for t in tasks]
        return observation

    env_utils.add_envs_task = patched_add_envs_task


def _apply_visual_patch(noise_std: float = 0.0, brightness_shift: float = 0.0) -> None:
    """Monkey-patch preprocess_observation to add visual perturbations."""
    import torch
    import lerobot.envs.utils as env_utils

    _original = env_utils.preprocess_observation

    def patched_preprocess(observation):
        processed = _original(observation)
        for key in list(processed.keys()):
            if key.startswith("observation.images."):
                img = processed[key]
                if noise_std > 0:
                    img = img + torch.randn_like(img) * noise_std
                if brightness_shift != 0:
                    img = img + brightness_shift
                processed[key] = torch.clamp(img, 0.0, 1.0)
        return processed

    env_utils.preprocess_observation = patched_preprocess


def main():
    config_path = os.environ.get("OOD_CONFIG")
    if not config_path:
        print("Error: OOD_CONFIG environment variable not set", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        ood_config = json.load(f)

    mode = ood_config.get("mode")

    if mode == "paraphrased":
        paraphrases = ood_config["paraphrases"]
        _apply_paraphrase_patch(paraphrases)
        print(f"OOD: paraphrased instructions ({len(paraphrases)} mappings)")

    elif mode == "visual":
        noise_std = ood_config.get("noise_std", 0.05)
        brightness_shift = ood_config.get("brightness_shift", 0.0)
        _apply_visual_patch(noise_std=noise_std, brightness_shift=brightness_shift)
        print(f"OOD: visual perturbation (noise_std={noise_std}, brightness={brightness_shift})")

    else:
        print(f"Error: unknown OOD mode '{mode}'", file=sys.stderr)
        sys.exit(1)

    # Run lerobot-eval with remaining CLI args
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main
    lerobot_eval_main()


if __name__ == "__main__":
    main()
