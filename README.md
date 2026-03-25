# SmolVLA LIBERO Fine-Tuning

Fine-tune [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (450M param Vision-Language-Action model) on the [LIBERO-10](https://huggingface.co/datasets/lerobot/libero_10_image) manipulation benchmark using LoRA, evaluate in simulation (ID + OOD), and run ablation studies.

Built on the [LeRobot](https://github.com/huggingface/lerobot) framework with [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)/[robosuite](https://github.com/ARISE-Initiative/robosuite) simulation.

Videos and training plots available at: [Google Drive](https://drive.google.com/drive/folders/1k_Bfd2FuBnf3YklDII-0YyqxEuCz7nLz?usp=sharing).


## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/smolvla-libero.git
cd smolvla-libero
```

### 2. Create the conda environment

```bash
mamba create -n smolvla python=3.11 -y
mamba activate smolvla
```

### 3. Install dependencies

```bash
mamba install cmake cxx-compiler -y
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -r requirements.txt
pip install robosuite==1.4.1
```

### 4. Apply local patches

This project adds per-action-dimension loss logging (`loss_action/{dx,dy,dz,drot1,drot2,drot3,gripper}`) to SmolVLA's training loop. The patch must be applied to the installed lerobot package:

```bash
python scripts/apply_patches.py
```

This is idempotent — safe to run multiple times. It locates the installed `modeling_smolvla.py` automatically and inserts 7 lines into `SmolVLAPolicy.forward()`.

### 5. Verify the install

```bash
# Run tests (no GPU required, ~30s)
PYTHONPATH=. mamba run -n smolvla python -m pytest tests/ -v
```

All 162 tests should pass. If imports fail, check that `robosuite==1.4.1` is installed (not 1.5.x).

### 6. Set up Weights & Biases (optional but recommended)

Training logs per-action-dimension losses and other diagnostics to wandb.

```bash
wandb login
```

To disable wandb, pass `--wandb.enable=false` to `train.py`, or set `WANDB_MODE=disabled`.

## Reproduce Training

### Base run (LoRA rank=64, 50k steps)

```bash
PYTHONPATH=. mamba run -n smolvla python scripts/train.py --config configs/base.yaml
```

This runs `lerobot-train` as a subprocess with:
- **Model:** `lerobot/smolvla_base` (452M params, ~3.5M trainable with LoRA rank=64)
- **Dataset:** `lerobot/libero_10_image` (379 episodes, 10 tasks, auto-downloaded from HuggingFace)
- **Hyperparams:** batch_size=32, lr=1e-3, cosine decay to 1e-4, 50k steps
- **Checkpoints:** Saved every 2k steps to `outputs/train/base/checkpoints/`
- **Time:** ~7.5 hours on RTX 5080 at ~1.8 steps/sec

**Important:** `lerobot-train` requires the output directory to not exist. If you need to re-run, either delete `outputs/train/base/` first or change `output_dir` via CLI override:

```bash
# Delete previous run
rm -rf outputs/train/base

# Or use a different output directory
mamba run -n smolvla python scripts/train.py --output_dir=outputs/train/base_v2
```

### Training with ablation overrides

```bash
# LoRA rank ablations
mamba run -n smolvla python scripts/train.py --override configs/ablations/rank_32.yaml
```

## Evaluation

### In-distribution evaluation

Evaluate on the same LIBERO-10 tasks used in training:

```bash
mamba run -n smolvla MUJOCO_GL=egl python scripts/evaluate.py \
  --checkpoint outputs/train/base \
  --mode id \
  --num-episodes 10
```

### Evaluate a specific checkpoint step

```bash
mamba run -n smolvla MUJOCO_GL=egl python scripts/evaluate.py \
  --checkpoint outputs/train/base/checkpoints/010000 \
  --mode id \
  --num-episodes 10
```

Results are stored in `results/eval/{run_name}/step_{N}/{mode}/` with:
- `eval_info.json` — raw lerobot-eval output
- `{mode}_results.json` — concise per-task summary
- `videos/` — episode rollout videos

### OOD evaluation: paraphrased instructions

Replace task instructions with semantic paraphrases (e.g., "put both the alphabet soup and the tomato sauce in the basket" → "place the soup can and the sauce container into the basket"):

```bash
mamba run -n smolvla MUJOCO_GL=egl python scripts/evaluate.py \
  --checkpoint outputs/train/base \
  --mode ood-paraphrased \
  --num-episodes 10
```

### OOD evaluation: visual perturbations

Add Gaussian noise and/or brightness shift to camera observations:

```bash
# Default: noise_std=0.05
mamba run -n smolvla MUJOCO_GL=egl python scripts/evaluate.py \
  --checkpoint outputs/train/base \
  --mode ood-visual \
  --num-episodes 10
```

## Other Scripts

### Dataset analysis

```bash
mamba run -n smolvla python scripts/analyze.py
```

### Export episode videos

```bash
# One episode per task (side-by-side front + wrist camera)
mamba run -n smolvla python scripts/analyze.py --save-video
```


## Project Structure

```
.
├── configs/
│   ├── base.yaml                 # Base training config (LoRA rank=64, 50k steps, batch=32)
│   └── ablations/                # Single-variable override configs
│       ├── rank_8.yaml           #   LoRA rank 8
│       ├── rank_32.yaml          #   LoRA rank 32
│       ├── rank_128.yaml         #   LoRA rank 128
│       ├── steps_5k.yaml         #   5k training steps
│       ├── steps_10k.yaml        #   10k training steps
│       ├── data_25pct.yaml       #   25% of training episodes
│       └── data_50pct.yaml       #   50% of training episodes
├── src/                          # Library modules (no heavy deps, unit-testable)
│   ├── config.py                 #   YAML loading, deep merge, CLI flag generation
│   ├── dataset.py                #   Dataset stats, stratified splitting, subsetting
│   ├── ood.py                    #   OOD paraphrases and cross-suite transfer configs
│   └── plotting.py               #   Loss curves, eval bar charts
├── scripts/                      # CLI entry points (thin wrappers, lazy-import heavy deps)
│   ├── train.py                  #   Training via lerobot-train subprocess
│   ├── evaluate.py               #   ID + OOD evaluation via lerobot-eval
│   ├── ood_eval_wrapper.py       #   Monkey-patches lerobot-eval for OOD modes
│   ├── analyze.py                #   Dataset analysis and video export
│   ├── plot.py                   #   Generate training/eval plots
│   └── ablate.py                 #   Ablation grid (train + eval + plot)
├── tests/                        # 162 pytest tests (no GPU required)
├── results/                      # Evaluation results and plots (checked in)
│   └── eval/{run_name}/step_{N}/{mode}/
└── outputs/                      # Training outputs (gitignored)
    └── train/{run_name}/checkpoints/
```