#!/usr/bin/env python3
"""Part 4: Policy evaluation — ID and OOD conditions in LIBERO simulation.

ID evaluation uses lerobot-eval directly (the official LeRobot evaluation pipeline).
OOD evaluation uses ood_eval_wrapper.py which monkey-patches lerobot-eval to modify
observations before the policy sees them.

Modes:
    id              — Same tasks as training (lerobot-eval directly).
    ood-paraphrased — Same tasks, rephrased language instructions.
    ood-visual      — Same tasks, Gaussian noise + brightness shift on camera images.
    ood-cross-suite — Different LIBERO suite (e.g., libero_spatial).

Output directory structure:
    results/eval/{run_name}/step_{N}/{mode}/
        eval_info.json      — raw lerobot-eval output
        videos/             — episode videos
        {mode}_results.json — concise summary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src.ood import get_cross_suite_config, get_paraphrase_map


# Rename map: LIBERO env produces observation.images.image2 (wrist camera)
# but the training dataset used observation.images.wrist_image.
RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}


def resolve_checkpoint(checkpoint_path: str) -> tuple[str, str, str]:
    """Resolve checkpoint path and extract run name + step.

    Returns (pretrained_model_path, run_name, step_label).
    """
    cp = Path(checkpoint_path)

    # Extract run name and step from path structure
    run_name = "unknown"
    step_label = "last"

    # Walk up to find the training output root (parent of "checkpoints/")
    for parent in [cp] + list(cp.parents):
        if (parent / "checkpoints").is_dir():
            run_name = parent.name
            break
        if parent.name == "checkpoints":
            run_name = parent.parent.name
            # The directory after checkpoints/ is the step
            rel = cp.relative_to(parent)
            step_label = rel.parts[0] if rel.parts else "last"
            break

    # If pointing directly to a numbered checkpoint dir
    if cp.name.isdigit():
        step_label = cp.name
    elif cp.parent.name.isdigit():
        step_label = cp.parent.name

    if not cp.exists():
        sys.exit(f"Error: checkpoint path does not exist: {cp}")

    # Resolve to pretrained_model path
    pretrained = cp / "pretrained_model"
    if pretrained.exists():
        return str(pretrained), run_name, step_label

    ckpts = cp / "checkpoints"
    if ckpts.exists():
        last = ckpts / "last" / "pretrained_model"
        if last.exists():
            return str(last), run_name, "last"
        numbered = sorted(
            [d for d in ckpts.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if numbered:
            pm = numbered[-1] / "pretrained_model"
            if pm.exists():
                return str(pm), run_name, numbered[-1].name

    # No pretrained_model found — likely wrong path
    sys.exit(
        f"Error: no pretrained_model found under {cp}\n"
        f"Available checkpoints: {', '.join(d.name for d in sorted(cp.parent.iterdir()) if d.is_dir()) if cp.parent.exists() else 'N/A'}"
    )


def _lerobot_eval_args(
    checkpoint: str,
    task_suite: str,
    n_episodes: int,
    output_dir: str,
    batch_size: int = 1,
) -> list[str]:
    """Build CLI args for lerobot-eval (without the python/module prefix)."""
    return [
        f"--policy.path={checkpoint}",
        f"--env.type=libero",
        f"--env.task={task_suite}",
        f"--env.obs_type=pixels_agent_pos",
        f"--env.observation_height=256",
        f"--env.observation_width=256",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--policy.device=cuda",
        f"--output_dir={output_dir}",
        f"--rename_map={json.dumps(RENAME_MAP)}",
    ]


def _eval_env() -> dict[str, str]:
    """Environment variables for eval subprocesses."""
    return {**os.environ, "MUJOCO_GL": "egl"}


def run_lerobot_eval(
    checkpoint: str,
    task_suite: str,
    n_episodes: int,
    output_dir: str,
    batch_size: int = 1,
) -> dict:
    """Run lerobot-eval directly and return the results dict."""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        *_lerobot_eval_args(checkpoint, task_suite, n_episodes, output_dir, batch_size),
    ]
    print(f"Running: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, env=_eval_env())
    if result.returncode != 0:
        print(f"lerobot-eval failed with exit code {result.returncode}")
        return {}

    return _load_eval_info(output_dir)


def run_ood_eval(
    checkpoint: str,
    task_suite: str,
    n_episodes: int,
    output_dir: str,
    batch_size: int,
    ood_config: dict,
) -> dict:
    """Run lerobot-eval via ood_eval_wrapper.py with observation patches.

    Args:
        ood_config: Dict with 'mode' key ('paraphrased' or 'visual') and
                    mode-specific parameters (paraphrases dict, noise_std, etc.).
    """
    # Write OOD config to temp file
    fd, config_path = tempfile.mkstemp(suffix=".json", prefix="ood_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ood_config, f)

        cmd = [
            sys.executable, "scripts/ood_eval_wrapper.py",
            *_lerobot_eval_args(checkpoint, task_suite, n_episodes, output_dir, batch_size),
        ]
        env = {**_eval_env(), "OOD_CONFIG": config_path}

        mode_label = ood_config.get("mode", "ood")
        print(f"Running OOD ({mode_label}): {' '.join(cmd[:3])}...")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"OOD eval failed with exit code {result.returncode}")
            return {}
    finally:
        os.unlink(config_path)

    return _load_eval_info(output_dir)


def _load_eval_info(output_dir: str) -> dict:
    """Load eval_info.json from an eval output directory."""
    eval_info_path = Path(output_dir) / "eval_info.json"
    if eval_info_path.exists():
        with open(eval_info_path) as f:
            return json.load(f)
    return {}


def get_task_descriptions(task_suite: str) -> dict[int, str]:
    """Look up task descriptions from LIBERO benchmark registry.

    Returns {task_id: description}. Falls back to empty dict on import failure.
    """
    try:
        from libero.libero import benchmark
        bm = benchmark.get_benchmark(task_suite)()
        return {i: bm.get_task(i).language for i in range(bm.n_tasks)}
    except Exception:
        return {}


def build_summary(info: dict, mode: str, task_suite: str) -> dict:
    """Build a concise summary dict from raw lerobot-eval output.

    Structure:
        mode, suite, overall_success_rate, n_episodes, eval_time_s, avg_episode_time_s,
        tasks: [{task_id, description, success_rate, n_episodes, successes, failures}]
    """
    overall = info.get("overall", {})
    per_task = info.get("per_task", [])
    descriptions = get_task_descriptions(task_suite)

    n_episodes = overall.get("n_episodes", 0)
    eval_s = overall.get("eval_s", 0.0)

    tasks = []
    for task_info in per_task:
        tid = task_info.get("task_id", 0)
        metrics = task_info.get("metrics", {})
        successes = metrics.get("successes", [])
        n = len(successes)
        rate = sum(successes) / n if n else 0.0
        success_ids = [i for i, s in enumerate(successes) if s]
        failure_ids = [i for i, s in enumerate(successes) if not s]
        tasks.append({
            "task_id": tid,
            "description": descriptions.get(tid, ""),
            "success_rate": round(rate, 4),
            "n_episodes": n,
            "successes": success_ids,
            "failures": failure_ids,
        })

    return {
        "mode": mode,
        "suite": task_suite,
        "overall_success_rate": round(overall.get("pc_success", 0.0) / 100.0, 4),
        "n_episodes": n_episodes,
        "eval_time_s": round(eval_s, 1),
        "avg_episode_time_s": round(eval_s / n_episodes, 1) if n_episodes else 0.0,
        "tasks": tasks,
    }


def print_results(summary: dict) -> None:
    """Print a concise one-line-per-task results table."""
    mode = summary["mode"]
    sr = summary["overall_success_rate"]
    n = summary["n_episodes"]
    t = summary["eval_time_s"]

    print(f"\n  {mode}  |  {sr:.1%} success  |  {n} eps  |  {t:.0f}s")
    print(f"  {'─' * 56}")
    for task in summary["tasks"]:
        tid = task["task_id"]
        desc = task["description"]
        rate = task["success_rate"]
        wins = len(task["successes"])
        total = task["n_episodes"]
        label = f"{desc[:48]}" if desc else f"task {tid}"
        print(f"  {tid:>2}  {rate:>5.0%}  ({wins}/{total})  {label}")


def save_summary(summary: dict, output_dir: Path) -> None:
    """Save the concise results JSON."""
    mode = summary["mode"]
    out_path = output_dir / f"{mode}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy in LIBERO")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--mode",
        choices=["id", "ood-paraphrased", "ood-visual", "ood-cross-suite"],
        required=True,
    )
    parser.add_argument("--num-episodes", type=int, default=20,
                        help="Episodes per task")
    parser.add_argument("--output-dir", type=str, default="results/eval",
                        help="Root output directory")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (auto-derived from checkpoint path if omitted)")
    parser.add_argument("--source-suite", type=str, default="libero_10")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Parallel envs per task")
    # OOD-visual parameters
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Gaussian noise std for ood-visual (default: 0.05)")
    parser.add_argument("--brightness-shift", type=float, default=0.0,
                        help="Brightness shift for ood-visual (default: 0.0)")
    # OOD-paraphrased parameters
    parser.add_argument("--paraphrase-variant", type=int, default=0,
                        help="Which paraphrase variant to use (0 or 1)")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    checkpoint, auto_run_name, step_label = resolve_checkpoint(args.checkpoint)
    run_name = args.run_name or auto_run_name
    step_dir = f"step_{step_label}" if step_label.isdigit() else step_label

    # Build mode-specific output dir suffix for ood-visual with non-default params
    mode_dir = args.mode
    if args.mode == "ood-visual" and (args.noise_std != 0.05 or args.brightness_shift != 0.0):
        mode_dir = f"ood-visual_n{args.noise_std}_b{args.brightness_shift}"

    # results/eval/{run_name}/{step}/{mode}/
    eval_dir = Path(args.output_dir) / run_name / step_dir / mode_dir

    print(f"Checkpoint: {checkpoint}")
    print(f"Run: {run_name}  Step: {step_label}  Mode: {args.mode}")
    print(f"Output: {eval_dir}")

    task_suite = args.source_suite
    info = {}

    if args.mode == "id":
        info = run_lerobot_eval(
            checkpoint=checkpoint,
            task_suite=task_suite,
            n_episodes=args.num_episodes,
            output_dir=str(eval_dir),
            batch_size=args.batch_size,
        )

    elif args.mode == "ood-paraphrased":
        paraphrases = get_paraphrase_map(task_suite, variant=args.paraphrase_variant)
        print(f"Paraphrase mappings ({len(paraphrases)}):")
        for orig, para in paraphrases.items():
            print(f"  {orig[:50]}...")
            print(f"    -> {para[:50]}...")
        info = run_ood_eval(
            checkpoint=checkpoint,
            task_suite=task_suite,
            n_episodes=args.num_episodes,
            output_dir=str(eval_dir),
            batch_size=args.batch_size,
            ood_config={"mode": "paraphrased", "paraphrases": paraphrases},
        )

    elif args.mode == "ood-visual":
        info = run_ood_eval(
            checkpoint=checkpoint,
            task_suite=task_suite,
            n_episodes=args.num_episodes,
            output_dir=str(eval_dir),
            batch_size=args.batch_size,
            ood_config={
                "mode": "visual",
                "noise_std": args.noise_std,
                "brightness_shift": args.brightness_shift,
            },
        )

    elif args.mode == "ood-cross-suite":
        cross_config = get_cross_suite_config(task_suite)
        target = cross_config["targets"][0]
        task_suite = target["suite"]
        print(f"Cross-suite: {args.source_suite} -> {task_suite}")
        info = run_lerobot_eval(
            checkpoint=checkpoint,
            task_suite=task_suite,
            n_episodes=args.num_episodes,
            output_dir=str(eval_dir),
            batch_size=args.batch_size,
        )

    if info:
        summary = build_summary(info, args.mode, task_suite)
        print_results(summary)
        save_summary(summary, eval_dir)


if __name__ == "__main__":
    main()
