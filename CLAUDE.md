# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ndimensions labs SmolVLA 1-Day Challenge** — fine-tune SmolVLA (450M param VLA model) on LIBERO manipulation tasks using LoRA, evaluate in simulation (ID + OOD), run ablations. Spec: `ndims_AI_short.pdf`.

**Stack:** Python, PyTorch, LeRobot framework (`lerobot[smolvla,peft]`), robosuite + MuJoCo (LIBERO sim), wandb.

## Key Decisions

- **Dataset:** `lerobot/libero_10_image` — 379 episodes, 10 tasks, 101k frames, Franka Panda, 8D state + 7D action + 2x 256×256 cameras
- **Base checkpoint:** `lerobot/smolvla_base` (450M params)
- **Fine-tuning:** LoRA via LeRobot's `--peft.*` flags (rank=32, LR=1e-3, default target modules: q/v proj + action/state proj)
- **Simulator:** LIBERO/robosuite (CPU physics, no GPU needed for eval)
- **OOD conditions:** (1) paraphrased instructions, (2) cross-suite transfer to `libero_spatial`
- **Ablations:** dataset size (25%/50%/100%), training steps (5k/10k/20k), LoRA rank (8/32/64)

## Project Structure

```
scripts/                       # Entry points (one per challenge part)
  analyze.py                   # Part 1: dataset EDA → stats + train/val split
  train.py                     # Part 2: launches lerobot-train with config
  evaluate.py                  # Part 4: ID + OOD evaluation, video recording
  ablate.py                    # Part 5: ablation grid (train + eval per variant)
  plot.py                      # Part 3: diagnostic plots from training logs

src/                           # Reusable modules imported by scripts
  config.py                    # Load YAML configs, merge overrides, map to lerobot-train args
  dataset.py                   # Dataset loading, stats, train/val split, subsetting
  evaluator.py                 # LIBERO env setup, policy rollout, metrics, video capture
  ood.py                       # OOD handlers: instruction paraphrases + cross-suite setup
  plotting.py                  # Loss curves, eval bar charts, ablation comparison plots

configs/
  base.yaml                    # Full training config (all defaults)
  ablations/                   # Override-only configs (merged on top of base)
    rank_8.yaml, rank_64.yaml, steps_5k.yaml, steps_10k.yaml, data_25pct.yaml, data_50pct.yaml

outputs/                       # gitignored — checkpoints, wandb logs, intermediates
results/                       # Submission artifacts — plots, eval JSONs, videos
```

## Commands

```bash
# Install
pip install -r requirements.txt

# Part 1: Dataset analysis
python scripts/analyze.py

# Part 2: Training (single command, reproducible)
python scripts/train.py --config configs/base.yaml

# Part 3: Plot training diagnostics
python scripts/plot.py --run outputs/train/main

# Part 4: Evaluation
python scripts/evaluate.py --checkpoint outputs/train/main --mode id
python scripts/evaluate.py --checkpoint outputs/train/main --mode ood-instructions
python scripts/evaluate.py --checkpoint outputs/train/main --mode ood-cross-suite

# Part 5: Ablation (runs all variants)
python scripts/ablate.py --base configs/base.yaml --ablations configs/ablations/

# Direct lerobot-train (equivalent to scripts/train.py):
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/libero_10_image \
  --peft.method_type=LORA --peft.r=32 \
  --policy.optimizer_lr=1e-3 \
  --batch_size=8 --steps=20000 \
  --output_dir=outputs/train/main
```

## Architecture Notes

**SmolVLA** = SigLIP vision encoder (frozen) + SmolLM2-360M language decoder + Flow Matching action expert (~100M params). Outputs action chunks (50 steps × action_dim). Uses 64 visual tokens per frame via PixelShuffle.

**LoRA targets** (default in LeRobot): `q_proj`/`v_proj` in the LM expert, plus `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`. LR should be 10× higher than full fine-tuning (1e-3 vs 1e-4).

**LeRobot dataset format:** Parquet + MP4 (AV1). Fields: `observation.images.image`, `observation.images.wrist_image`, `observation.state` (8D), `action` (7D), `language_instruction`, `episode_index`, `frame_index`.

**LIBERO** runs on robosuite → MuJoCo. CPU-based physics. Set `MUJOCO_GL=egl` for headless GPU rendering or `MUJOCO_GL=osmesa` for CPU rendering.

## Code Conventions

- Scripts are thin entry points: parse args → call `src/` modules → save outputs
- Configs are override-only YAML: ablation configs only specify what differs from `base.yaml`; `src/config.py` handles merging
- `outputs/` is ephemeral (gitignored); `results/` is for submission artifacts

## Deliverables Checklist

- [ ] GitHub repo with scripts, configs, results, README
- [ ] Training reproducible via single command
- [ ] Fine-tuned checkpoint on HuggingFace Hub
- [ ] Training plots + evaluation results
- [ ] Short video: successes + failures
- [ ] ~1-page writeup
