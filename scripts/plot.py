from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.plotting import plot_eval_results, plot_loss_curve


def main() -> None:
    """Generate plots from training logs and/or evaluation results."""
    parser = argparse.ArgumentParser(description="Generate training & eval plots.")
    parser.add_argument("--run", type=Path, default=None,
                        help="Path to training output directory (contains train.log).")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to evaluation results JSON file.")
    args = parser.parse_args()

    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        log_path = args.run / "train.log"
        if log_path.exists():
            plot_loss_curve(log_path, out_dir / "loss_curve.png")
            print(f"Saved loss curve to {out_dir / 'loss_curve.png'}")
        else:
            print(f"Warning: {log_path} not found, skipping loss curve.")

    if args.results is not None:
        with open(args.results) as f:
            data = json.load(f)
        plot_eval_results(data, out_dir / "eval_results.png")
        print(f"Saved eval results to {out_dir / 'eval_results.png'}")


if __name__ == "__main__":
    main()
