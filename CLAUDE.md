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
# Run all tests (128 tests)
mamba run -n smolvla python -m pytest tests/ -v

# Run a single test file
mamba run -n smolvla python -m pytest tests/test_config.py -v

# Run a single test
mamba run -n smolvla python -m pytest tests/test_config.py::TestMergeConfigs::test_deep_merge -v

# Part 1: Dataset analysis
mamba run -n smolvla python scripts/analyze.py

# Part 2: Training (supports --config, --override, and --key=value overrides)
mamba run -n smolvla python scripts/train.py --config configs/base.yaml
mamba run -n smolvla python scripts/train.py --override configs/ablations/rank_8.yaml
mamba run -n smolvla python scripts/train.py --steps=5000 --peft.r=16

# Part 3: Plot training diagnostics
mamba run -n smolvla python scripts/plot.py --run outputs/train/main
mamba run -n smolvla python scripts/plot.py --results results/eval/id_results.json

# Part 4: Evaluation (three modes)
mamba run -n smolvla python scripts/evaluate.py --checkpoint outputs/train/main --mode id
mamba run -n smolvla python scripts/evaluate.py --checkpoint outputs/train/main --mode ood-instructions
mamba run -n smolvla python scripts/evaluate.py --checkpoint outputs/train/main --mode ood-cross-suite

# Part 5: Ablation grid (trains + evaluates all variants)
mamba run -n smolvla python scripts/ablate.py --base configs/base.yaml --ablations configs/ablations/
mamba run -n smolvla python scripts/ablate.py --skip-training --skip-eval  # replot from existing results
```

## Key Decisions

- **Dataset:** `lerobot/libero_10_image` — 379 episodes, 10 tasks, 101k frames, Franka Panda, 8D state + 7D action + 2x 256×256 cameras
- **Base checkpoint:** `lerobot/smolvla_base` (450M params)
- **Fine-tuning:** LoRA via LeRobot's `--peft.*` flags (rank=32, LR=1e-3, default target modules: q/v proj + action/state proj)
- **Critical flags:** `policy.load_vlm_weights=true` (loads pretrained weights), `policy.push_to_hub=false`, `eval_freq=0` (eval env creation OOMs; evaluate separately after training)
- **Simulator:** LIBERO/robosuite 1.4.1 (CPU physics, no GPU needed for eval). robosuite 1.5.x is incompatible with libero.
- **OOD conditions:** (1) paraphrased instructions, (2) cross-suite transfer to `libero_spatial`
- **Ablations:** dataset size (25%/50%/100%), training steps (5k/10k/20k), LoRA rank (8/32/64)
- **Performance:** ~17 steps/sec at batch_size=2, ~5 steps/sec at batch_size=8 (RTX 5080, LoRA)

## Architecture

**Two-layer design:** `scripts/` are thin CLI entry points (parse args → call `src/` → save outputs). `src/` contains reusable, unit-testable modules with no heavy dependencies.

### Config pipeline (`src/config.py` → `scripts/train.py`)
`configs/base.yaml` has all training defaults. Ablation configs in `configs/ablations/` only specify overrides. The merge chain is: base YAML → override YAML → CLI `--key=value` args. `config_to_cli_args()` flattens nested dicts to dot-separated `lerobot-train` flags. `scripts/train.py` adds its own CLI override parsing (`_cast_value`, `_set_nested`) on top, then calls `lerobot-train` via subprocess.

### Dataset analysis (`src/dataset.py` → `scripts/analyze.py`)
`src/dataset.py` works with plain dicts/lists (no LeRobot dependency): `compute_stats()` takes metadata + episode-task mapping, `stratified_split()` ensures per-task val representation, `subset_episodes()` enables data-size ablations. `scripts/analyze.py` is the only file that lazy-imports `LeRobotDataset` to extract metadata, then delegates to `src/dataset.py`.

### Evaluation (`src/evaluator.py` + `src/ood.py` → `scripts/evaluate.py`)
`src/evaluator.py` defines a Protocol-based `Env` interface and generic `run_rollout()`/`run_evaluation()` functions that take any env + policy callable — fully testable with mock objects. `EvalResults` aggregates per-task and overall success rates with JSON persistence. `src/ood.py` provides `paraphrase_instruction()` (lookup dict for 10 LIBERO-10 tasks + synonym fallback) and `get_cross_suite_config()` for transfer targets. `scripts/evaluate.py` handles the heavy LIBERO/robosuite/SmolVLA imports.

### Ablation (`scripts/ablate.py`)
Discovers all YAML files in `configs/ablations/`, runs `scripts/train.py` + `scripts/evaluate.py` for each variant plus a baseline, collects `*_results.json` files, and generates comparison plots via `src/plotting.py`. Supports `--skip-training` and `--skip-eval` to rerun from existing checkpoints/results.

### Plotting (`src/plotting.py` → `scripts/plot.py`)
Three plot types: `plot_loss_curve()` (JSON lines or CSV logs), `plot_eval_results()` (bar chart), `plot_ablation_comparison()` (grouped bars). All use `matplotlib.use("Agg")` for headless rendering.

## SmolVLA Architecture Notes

**SmolVLA** = SigLIP vision encoder (frozen) + SmolLM2-360M language decoder + Flow Matching action expert (~100M params). Outputs action chunks (50 steps × action_dim). Uses 64 visual tokens per frame via PixelShuffle.

**LoRA targets** (default in LeRobot): `q_proj`/`v_proj` in the LM expert, plus `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`. LR should be 10× higher than full fine-tuning (1e-3 vs 1e-4).

**LeRobot dataset format:** Parquet + MP4 (AV1). Fields: `observation.images.image`, `observation.images.wrist_image`, `observation.state` (8D), `action` (7D), `language_instruction`, `episode_index`, `frame_index`.

**LIBERO** runs on robosuite → MuJoCo. CPU-based physics. Set `MUJOCO_GL=egl` for headless GPU rendering or `MUJOCO_GL=osmesa` for CPU rendering.

## Code Conventions

- `outputs/` is ephemeral (gitignored); `results/` is for submission artifacts
- Heavy dependencies (LeRobot, robosuite) are lazy-imported only in `scripts/`, keeping `src/` unit-testable with mocks
- Tests use pytest class-based organization with fixtures providing mock metadata matching `lerobot/libero_10_image` structure
- All `src/` modules use `from __future__ import annotations` and type hints

## Deliverables Checklist

- [ ] GitHub repo with scripts, configs, results, README
- [ ] Training reproducible via single command
- [ ] Fine-tuned checkpoint on HuggingFace Hub
- [ ] Training plots + evaluation results
- [ ] Short video: successes + failures
- [ ] ~1-page writeup
