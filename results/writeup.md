# Writeup: SmolVLA Fine-Tuning on LIBERO-10

## What I Tried

I fine-tuned SmolVLA-0.45B on **LIBERO-10** (also known as LIBERO-Long), the hardest suite in the LIBERO benchmark: 10 multi-step tabletop manipulation tasks with a Franka Panda arm, 379 demonstration episodes (~38 per task), dual cameras (front + wrist) at 256x256, 7-DoF delta actions.

**Training setup:** LoRA rank=64 on `q_proj`/`v_proj` of the language model plus action expert projections (~3.5M trainable / 452M total params), batch_size=32, lr=1e-3 with cosine decay to 1e-4, AdamW optimizer, 50k steps. Trained on a single RTX 5080 (16 GB VRAM) in ~7.5 hours. All metrics logged to Iights & Biases, including custom per-action-dimension losses (dx, dy, dz, drot1-3, gripper) via a local patch to SmolVLA's `forward()`.

**Evaluation:** Used `lerobot-eval` via subprocess for all modes — I intentionally avoided custom rollout loops to preserve the official observation processing chain. For OOD evaluation, I tested paraphrased instructions and visual perturbations.

## What Worked

**Training converged smoothly.** Loss dropped from ~0.45 to 0.073 over 50k steps with stable gradient norms (~0.05). Per-action-dimension loss breakdown shoId gripper loss dominating early and converging fastest, while rotation dimensions (drot1-3) remained the highest-loss components throughout. This is consistent with the difficulty of learning precise orientation control from limited demonstrations.

**Single-object tasks reached strong performance.** In-distribution evaluation (100 episodes, 10 per task) at step 50k yielded **29% overall**, but with extreme per-task variance. A clear structural pattern: single-object multi-step tasks average **60%**, while two-object sequential tasks average **8%**. The 60% on single-object tasks confirms the LoRA configuration, observation pipeline, and training hyperparameters are effective.

## Ablation 1: OOD Paraphrased Instructions (29% → 0%)

I replaced each task instruction with a semantic paraphrase (e.g., *"put both the alphabet soup and the tomato sauce in the basket"* → *"place the soup can and the sauce container into the basket"*) and re-evaluated the same checkpoint.

| Task | ID | OOD-Paraphrased | Description |
|------|----|-----------------|-------------|
| 2 | 80% | 0% | Turn on stove + moka pot |
| 5 | 70% | 0% | Book in caddy |
| 9 | 50% | 0% | Mug in microwave + close |
| 3 | 40% | 0% | Bowl in draIr + close |
| All others | 0-20% | 0% | Two-object tasks |
| **Overall** | **29%** | **0%** | |

**Result: total catastrophic failure.** Every task drops to 0% — including task 2 (80% → 0%) and task 5 (70% → 0%). The policy doesn't degrade gradually; it collapses completely. This implies the model has learned task-specific visuomotor policies indexed by exact instruction strings, not language-conditioned behavior.

**Proposed fix:** Recent work (arXiv:2603.16044) shows that fine-tuning OpenVLA with LLM-generated paraphrases during training significantly restores instruction robustness — the model learns to map diverse phrasings to the same behavior rather than memorizing a single token sequence per task.

## Ablation 2: OOD Visual Perturbation (29% → 17%)

I added Gaussian noise (σ=0.05) to all camera observations at inference time, simulating minor lighting or sensor variation.

| Task | ID | OOD-Visual | Delta | Description |
|------|----|-----------|-------|-------------|
| 2 | 80% | 40% | -40pp | Stove + moka pot |
| 5 | 70% | 40% | -30pp | Book in caddy |
| 9 | 50% | 10% | -40pp | Mug in microwave |
| 3 | 40% | 50% | +10pp | Bowl in draIr |
| 6 | 20% | 10% | -10pp | Mug + pudding |
| 8 | 20% | 0% | -20pp | Both moka pots |
| 1 | 10% | 10% | 0 | Cream cheese + butter |
| 0 | 0% | 10% | +10pp | Soup + sauce |
| 4 | 0% | 0% | 0 | Two mugs on plates |
| 7 | 0% | 0% | 0 | Soup + cream cheese |
| **Overall** | **29%** | **17%** | **-12pp** | |

**Result: significant but non-catastrophic degradation.** Overall success drops 41% relative (29% → 17%). The highest-performing ID tasks suffer the largest absolute drops (task 2: -40pp, task 5: -30pp, task 9: -40pp), while already-failing tasks stay at or near 0%. Task 3 and 0 appear to gain +10pp, but this is within the noise margin for 10 episodes.

**Interpretation:** Unlike paraphrased instructions, mild visual noise does not completely break the policy. It still executes recognizable manipulation behaviors, just with reduced precision. The model has learned visual features that partially generalize beyond the exact training pixel distribution. Training-time image augmentation (random crops, color jitter, Gaussian noise) would likely improve visual robustness substantially.

## What Didn't Work

**Two-object sequential tasks near-completely fail (8% ID average).** These tasks require picking and placing two objects sequentially (e.g., "put both the alphabet soup and the tomato sauce in the basket"). After placing the first object, the model 50-step action chunking commits to a long action trajectory predicted from a single observation, so if the first chunk doesn't perfectly complete the handoff between objects, the remaining steps execute stale actions that drift away from the second object. With only ~38 demos, the model sees very few examples of this critical transition point, making it difficult to learn a robust second-object grasping policy. By contrast, single-object multi-step tasks (e.g., "pick up the book and place it in the back compartment of the caddy") average 60% because the entire trajectory follows a single grasp-move-place arc with no object transition.

**Training loss continued declining but eval plateaued.** Step 30k achieved 30% overall (vs 29% at 50k) — 20k additional steps yielded no meaningful improvement. However, I suspect further training will improve performance because rotation losses seem to not converge yet.

**Language grounding is absent — but this is a field-wide problem, not specific to our setup.** The 0% OOD-paraphrased result initially looks like a failure of our fine-tuning, but LIBERO-PRO and LIBERO-Plus show that even much larger models (OpenVLA 7B, pi0 3.3B, pi0.5) exhibit the same collapse. Current VLAs trained on LIBERO do not use language semantically — they memorize visuomotor trajectories indexed by visual scene context, with language serving as a de facto task ID.

## What I'd Do With More Compute/Time

1. **Training-time image augmentation** — random crops, color jitter, and Gaussian noise would directly address the 41% visual-OOD drop at near-zero compute cost.
2. **Instruction augmentation during training** — randomly substitute paraphrased instructions during fine-tuning to prevent instruction memorization and preserve language grounding. This is the most impactful change suggested by our results.
3. **Temporal ensembling** (`n_action_steps=1`) — re-predict every step rather than executing full 50-step chunks.
4. **More Training Steps" — rotation loss not converge yet.
