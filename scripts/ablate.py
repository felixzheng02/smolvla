#!/usr/bin/env python3
"""Part 5: Ablation study — run training + evaluation for each ablation variant."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.plotting import plot_ablation_comparison


def discover_ablation_configs(ablations_dir: str | Path) -> dict[str, Path]:
    """Find all YAML ablation configs in a directory.

    Returns dict of ablation_name -> config_path, sorted by name.
    """
    ablations_dir = Path(ablations_dir)
    configs = {}
    for p in sorted(ablations_dir.glob("*.yaml")):
        name = p.stem  # e.g., "rank_8" from "rank_8.yaml"
        configs[name] = p
    return configs


def run_training(
    base_config: str,
    override_config: str | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Run scripts/train.py with given configs. Returns exit code."""
    cmd = [sys.executable, "scripts/train.py", "--config", str(base_config)]
    if override_config:
        cmd.extend(["--override", str(override_config)])
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"Training: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode


def collect_eval_results(output_dir: str | Path) -> dict[str, float]:
    """Load evaluation results from an output directory.

    Looks for *_results.json files and extracts overall success rates.
    """
    output_dir = Path(output_dir)
    results = {}
    for p in sorted(output_dir.glob("*_results.json")):
        data = json.loads(p.read_text())
        mode = data.get("mode", p.stem)
        results[mode] = data.get("overall_success_rate", 0.0)
    return results


def run_ablation_grid(
    base_config: str,
    ablation_configs: dict[str, Path],
    eval_mode: str = "id",
    num_episodes: int = 20,
    skip_training: bool = False,
    skip_eval: bool = False,
) -> dict[str, dict[str, float]]:
    """Run training + evaluation for each ablation variant.

    Returns {ablation_name: {metric_name: value}}.
    """
    all_results: dict[str, dict[str, float]] = {}

    # Also include baseline (no override)
    variants = {"baseline": None, **ablation_configs}

    for name, override_path in variants.items():
        config = load_config(base_config, override_path)
        output_dir = config.get("output_dir", f"outputs/train/{name}")

        # Train
        if not skip_training:
            exit_code = run_training(base_config, str(override_path) if override_path else None)
            if exit_code != 0:
                print(f"WARNING: Training failed for {name} (exit code {exit_code})")
                all_results[name] = {"train_failed": 1.0}
                continue

        # Evaluate
        if not skip_eval:
            eval_dir = Path(f"results/eval/{name}")
            eval_cmd = [
                sys.executable, "scripts/evaluate.py",
                "--checkpoint", str(output_dir),
                "--mode", eval_mode,
                "--num-episodes", str(num_episodes),
                "--output-dir", str(eval_dir),
            ]
            print(f"Evaluating {name}...")
            subprocess.run(eval_cmd)
            all_results[name] = collect_eval_results(eval_dir)
        else:
            # Try to load existing results
            eval_dir = Path(f"results/eval/{name}")
            if eval_dir.exists():
                all_results[name] = collect_eval_results(eval_dir)

    return all_results


def main():
    """Discover ablation configs, run the train+eval grid, and generate comparison plots."""
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--base", default="configs/base.yaml", help="Base config path")
    parser.add_argument("--ablations", default="configs/ablations/", help="Ablation configs dir")
    parser.add_argument("--eval-mode", default="id", help="Evaluation mode")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--skip-training", action="store_true", help="Skip training, use existing checkpoints")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval, use existing results")
    parser.add_argument("--output-dir", default="results/ablation", help="Output dir for plots")
    args = parser.parse_args()

    # Discover configs
    ablation_configs = discover_ablation_configs(args.ablations)
    print(f"Found {len(ablation_configs)} ablation configs: {list(ablation_configs.keys())}")

    # Run grid
    results = run_ablation_grid(
        base_config=args.base,
        ablation_configs=ablation_configs,
        eval_mode=args.eval_mode,
        num_episodes=args.num_episodes,
        skip_training=args.skip_training,
        skip_eval=args.skip_eval,
    )

    # Save raw results
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Plot comparison
    if results:
        plot_path = out / "ablation_comparison.png"
        plot_ablation_comparison(results, plot_path)
        print(f"Plot saved to {plot_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    for name, metrics in sorted(results.items()):
        metrics_str = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        print(f"  {name}: {metrics_str}")


if __name__ == "__main__":
    main()
