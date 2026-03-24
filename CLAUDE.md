# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ndimensions labs SmolVLA 1-Day Challenge** — fine-tune SmolVLA (450M param VLA model) on LIBERO manipulation tasks using LoRA, evaluate in simulation (ID + OOD), run ablations. Spec: `ndims_AI_short.pdf`.

**Stack:** Python 3.11, PyTorch, LeRobot framework (`lerobot[smolvla,peft]`), robosuite + MuJoCo (LIBERO sim), wandb.

## Environment Setup

```bash
# All work uses the 'smolvla' mamba environment
mamba activate smolvla

# Or prefix commands with:
mamba run -n smolvla <command>
```

## Commands

```bash
# Run all tests
mamba run -n smolvla python -m pytest tests/ -v

# Run a single test file
mamba run -n smolvla python -m pytest tests/test_config.py -v

# Run a single test
mamba run -n smolvla python -m pytest tests/test_config.py::TestMergeConfigs::test_deep_merge -v

# Part 1: Dataset analysis
mamba run -n smolvla python scripts/analyze.py

# Part 2: Training (single command, reproducible)
mamba run -n smolvla python scripts/train.py --config configs/base.yaml

# Part 3: Plot training diagnostics
mamba run -n smolvla python scripts/plot.py --run outputs/train/main

# Part 4: Evaluation
mamba run -n smolvla python scripts/evaluate.py --checkpoint outputs/train/main --mode id
mamba run -n smolvla python scripts/evaluate.py --checkpoint outputs/train/main --mode ood-instructions
mamba run -n smolvla python scripts/evaluate.py --checkpoint outputs/train/main --mode ood-cross-suite

# Part 5: Ablation (runs all variants)
mamba run -n smolvla python scripts/ablate.py --base configs/base.yaml --ablations configs/ablations/

# Direct lerobot-train equivalent:
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/libero_10_image \
  --peft.method_type=LORA --peft.r=32 \
  --policy.optimizer_lr=1e-3 \
  --batch_size=8 --steps=20000 \
  --output_dir=outputs/train/main
```

## Key Decisions

- **Dataset:** `lerobot/libero_10_image` — 379 episodes, 10 tasks, 101k frames, Franka Panda, 8D state + 7D action + 2x 256×256 cameras
- **Base checkpoint:** `lerobot/smolvla_base` (450M params)
- **Fine-tuning:** LoRA via LeRobot's `--peft.*` flags (rank=32, LR=1e-3, default target modules: q/v proj + action/state proj)
- **Simulator:** LIBERO/robosuite (CPU physics, no GPU needed for eval)
- **OOD conditions:** (1) paraphrased instructions, (2) cross-suite transfer to `libero_spatial`
- **Ablations:** dataset size (25%/50%/100%), training steps (5k/10k/20k), LoRA rank (8/32/64)

## Architecture

**Two-layer design:** `scripts/` are thin entry points (parse args → call `src/` → save outputs). `src/` contains reusable modules.

- `src/config.py` — YAML loading, deep-merge of base + override configs, conversion to `lerobot-train` CLI args. Ablation configs in `configs/ablations/` only specify overrides; `merge_configs()` handles the rest.
- `src/dataset.py` — Dataset stats computation from metadata dicts + episode-task mappings (no LeRobot dependency). Stratified train/val splitting, episode subsetting for data-size ablations. Split persistence via JSON.
- `scripts/analyze.py` — The only file that imports LeRobot directly (lazy import). Extracts metadata from `LeRobotDataset` objects and feeds it to `src/dataset.py` functions.

**Not yet built:** `src/evaluator.py` (LIBERO rollouts), `src/ood.py` (paraphrases + cross-suite), `src/plotting.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/ablate.py`, `scripts/plot.py`.

## Config System

`configs/base.yaml` has all training defaults. Ablation configs (e.g., `configs/ablations/rank_8.yaml`) only specify what differs. `src/config.py:load_config(base, override)` deep-merges them. `config_to_cli_args()` flattens nested dicts to dot-separated `lerobot-train` flags (e.g., `policy.path` → `--policy.path=...`).

## SmolVLA Architecture Notes

**SmolVLA** = SigLIP vision encoder (frozen) + SmolLM2-360M language decoder + Flow Matching action expert (~100M params). Outputs action chunks (50 steps × action_dim). Uses 64 visual tokens per frame via PixelShuffle.

**LoRA targets** (default in LeRobot): `q_proj`/`v_proj` in the LM expert, plus `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`. LR should be 10× higher than full fine-tuning (1e-3 vs 1e-4).

**LeRobot dataset format:** Parquet + MP4 (AV1). Fields: `observation.images.image`, `observation.images.wrist_image`, `observation.state` (8D), `action` (7D), `language_instruction`, `episode_index`, `frame_index`.

**LIBERO** runs on robosuite → MuJoCo. CPU-based physics. Set `MUJOCO_GL=egl` for headless GPU rendering or `MUJOCO_GL=osmesa` for CPU rendering.

## Code Conventions

- `outputs/` is ephemeral (gitignored); `results/` is for submission artifacts
- Heavy dependencies (LeRobot, robosuite) are lazy-imported only in `scripts/`, keeping `src/` unit-testable with mocks
- Tests use pytest fixtures with mock metadata matching `lerobot/libero_10_image` structure

## Deliverables Checklist

- [ ] GitHub repo with scripts, configs, results, README
- [ ] Training reproducible via single command
- [ ] Fine-tuned checkpoint on HuggingFace Hub
- [ ] Training plots + evaluation results
- [ ] Short video: successes + failures
- [ ] ~1-page writeup
