from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.plotting import plot_ablation_comparison, plot_eval_results, plot_loss_curve


class TestPlotLossCurve:
    """Tests for plot_loss_curve."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_basic(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        self._write_jsonl(log, [
            {"step": 0, "loss": 1.5},
            {"step": 100, "loss": 1.0},
            {"step": 200, "loss": 0.8},
        ])
        out = tmp_path / "loss.png"
        plot_loss_curve(log, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_val_loss_and_lr(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        self._write_jsonl(log, [
            {"step": 0, "loss": 1.5, "val_loss": 1.6, "lr": 1e-4},
            {"step": 100, "loss": 1.0, "val_loss": 1.1, "lr": 5e-5},
        ])
        out = tmp_path / "loss.png"
        plot_loss_curve(log, out)
        assert out.exists()

    def test_csv_format(self, tmp_path: Path) -> None:
        log = tmp_path / "train.csv"
        log.write_text("step,loss\n0,1.5\n100,1.0\n200,0.8\n")
        out = tmp_path / "loss.png"
        plot_loss_curve(log, out)
        assert out.exists()

    def test_single_point(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        self._write_jsonl(log, [{"step": 0, "loss": 1.5}])
        out = tmp_path / "loss.png"
        plot_loss_curve(log, out)
        assert out.exists()

    def test_empty_log(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("")
        out = tmp_path / "loss.png"
        plot_loss_curve(log, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        self._write_jsonl(log, [{"step": 0, "loss": 1.0}])
        out = tmp_path / "sub" / "dir" / "loss.png"
        plot_loss_curve(log, out)
        assert out.exists()


class TestPlotEvalResults:
    """Tests for plot_eval_results."""

    def test_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "eval.png"
        plot_eval_results({"ID": 0.85, "OOD-paraphrase": 0.72}, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_title(self, tmp_path: Path) -> None:
        out = tmp_path / "eval.png"
        plot_eval_results({"A": 0.5}, out, title="My Title")
        assert out.exists()

    def test_empty_results(self, tmp_path: Path) -> None:
        out = tmp_path / "eval.png"
        plot_eval_results({}, out)
        assert out.exists()

    def test_single_entry(self, tmp_path: Path) -> None:
        out = tmp_path / "eval.png"
        plot_eval_results({"only": 0.99}, out)
        assert out.exists()


class TestPlotAblationComparison:
    """Tests for plot_ablation_comparison."""

    def test_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "ablation.png"
        data = {
            "rank_8": {"ID": 0.80, "OOD": 0.65},
            "rank_32": {"ID": 0.87, "OOD": 0.73},
        }
        plot_ablation_comparison(data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_ablation(self, tmp_path: Path) -> None:
        out = tmp_path / "ablation.png"
        plot_ablation_comparison({"only": {"m1": 0.5, "m2": 0.6}}, out)
        assert out.exists()

    def test_empty_results(self, tmp_path: Path) -> None:
        out = tmp_path / "ablation.png"
        plot_ablation_comparison({}, out)
        assert out.exists()

    def test_many_ablations(self, tmp_path: Path) -> None:
        out = tmp_path / "ablation.png"
        data = {f"run_{i}": {"acc": i * 0.1, "f1": i * 0.05} for i in range(5)}
        plot_ablation_comparison(data, out)
        assert out.exists()
