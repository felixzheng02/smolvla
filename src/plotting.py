"""Plotting utilities: loss curves, eval bar charts, and ablation comparisons.

All functions use matplotlib with the 'Agg' backend for headless rendering
and write output directly to PNG files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_lerobot_log(log_path: Path) -> list[dict]:
    """Parse lerobot training log lines.

    Handles three formats:
    - lerobot text: ``step:50 smpl:400 ep:1 epch:0.00 loss:0.644 grdn:0.115 lr:4e-05 updt_s:0.1 data_s:0.1``
    - JSON lines
    - CSV (only for .csv extension)

    All key:value pairs from text logs are captured. Common field renames
    are applied for consistency (grdn->grad_norm, updt_s->update_s, etc.).
    """
    import re

    log_path = Path(log_path)
    records: list[dict] = []

    with open(log_path) as f:
        first_line = f.readline().strip()

    if first_line.startswith("{"):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    elif str(log_path).endswith(".csv"):
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({k: float(v) for k, v in row.items()})
    else:
        # Generic key:value parser for lerobot log format
        kv_pattern = re.compile(r"(\w+):(\S+)")
        renames = {"grdn": "grad_norm", "updt_s": "update_s", "data_s": "dataloading_s",
                    "smpl": "samples", "ep": "episodes", "epch": "epochs"}
        with open(log_path) as f:
            for line in f:
                pairs = kv_pattern.findall(line)
                if not pairs or not any(k == "step" for k, _ in pairs):
                    continue
                record: dict = {}
                for k, v in pairs:
                    k = renames.get(k, k)
                    v_str = v.replace("K", "e3").replace("M", "e6")
                    try:
                        num = float(v_str)
                        record[k] = int(num) if num == int(num) and "." not in v and "e" not in v.lower() else num
                    except ValueError:
                        record[k] = v
                if "step" in record:
                    records.append(record)

    return records


def plot_loss_curve(log_path: Path, output_path: Path) -> None:
    """Plot training loss (and optionally val_loss / lr) vs step from a log file."""
    records = parse_lerobot_log(log_path)

    if not records:
        fig, ax = plt.subplots()
        ax.set_title("Training Loss (no data)")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, label="train_loss")

    if "val_loss" in records[0]:
        val_losses = [r["val_loss"] for r in records]
        ax.plot(steps, val_losses, label="val_loss", linestyle="--")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()

    # Optional secondary axis for learning rate
    if "lr" in records[0]:
        ax2 = ax.twinx()
        ax2.plot(steps, [r["lr"] for r in records], color="gray", alpha=0.4, label="lr")
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="center right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eval_results(
    results: dict[str, float], output_path: Path, title: str = ""
) -> None:
    """Plot a bar chart of evaluation success rates per condition."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if results:
        names = list(results.keys())
        values = list(results.values())
        bars = ax.bar(names, values)
        ax.set_ylim(0, max(1.0, max(values) * 1.1))
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Success Rate")
    ax.set_title(title or "Evaluation Results")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_comparison(
    results: dict[str, dict[str, float]], output_path: Path
) -> None:
    """Plot a grouped bar chart comparing ablations across metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if not results:
        ax.set_title("Ablation Comparison (no data)")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    ablation_names = list(results.keys())
    metric_names = list(next(iter(results.values())).keys())
    n_ablations = len(ablation_names)
    n_metrics = len(metric_names)

    x = np.arange(n_metrics)
    width = 0.8 / max(n_ablations, 1)

    for i, abl in enumerate(ablation_names):
        vals = [results[abl].get(m, 0.0) for m in metric_names]
        offset = (i - (n_ablations - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=abl)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Value")
    ax.set_title("Ablation Comparison")
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
