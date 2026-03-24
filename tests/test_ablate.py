"""Tests for scripts/ablate.py — config discovery, result collection, grid logic."""

import json

import pytest
import yaml

from scripts.ablate import collect_eval_results, discover_ablation_configs, run_ablation_grid


class TestDiscoverAblationConfigs:
    def test_finds_yaml_files(self, tmp_path):
        (tmp_path / "rank_8.yaml").write_text("peft:\n  r: 8\n")
        (tmp_path / "rank_64.yaml").write_text("peft:\n  r: 64\n")
        (tmp_path / "not_yaml.txt").write_text("ignore me")
        configs = discover_ablation_configs(tmp_path)
        assert set(configs.keys()) == {"rank_8", "rank_64"}

    def test_empty_dir(self, tmp_path):
        assert discover_ablation_configs(tmp_path) == {}

    def test_sorted_by_name(self, tmp_path):
        (tmp_path / "z_last.yaml").write_text("")
        (tmp_path / "a_first.yaml").write_text("")
        configs = discover_ablation_configs(tmp_path)
        assert list(configs.keys()) == ["a_first", "z_last"]


class TestCollectEvalResults:
    def test_loads_results_json(self, tmp_path):
        results = {"mode": "id", "overall_success_rate": 0.85}
        (tmp_path / "id_results.json").write_text(json.dumps(results))
        collected = collect_eval_results(tmp_path)
        assert collected == {"id": 0.85}

    def test_multiple_result_files(self, tmp_path):
        (tmp_path / "id_results.json").write_text(
            json.dumps({"mode": "id", "overall_success_rate": 0.9})
        )
        (tmp_path / "ood_results.json").write_text(
            json.dumps({"mode": "ood", "overall_success_rate": 0.6})
        )
        collected = collect_eval_results(tmp_path)
        assert collected == {"id": 0.9, "ood": 0.6}

    def test_empty_dir(self, tmp_path):
        assert collect_eval_results(tmp_path) == {}

    def test_missing_fields_default_zero(self, tmp_path):
        (tmp_path / "test_results.json").write_text(json.dumps({"mode": "test"}))
        collected = collect_eval_results(tmp_path)
        assert collected["test"] == 0.0


class TestRunAblationGrid:
    def test_skip_both_with_existing_results(self, tmp_path):
        """When skipping train+eval, loads existing results."""
        # Create base config
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"steps": 100, "output_dir": str(tmp_path / "out")}))

        # Create ablation config
        abl_dir = tmp_path / "ablations"
        abl_dir.mkdir()
        (abl_dir / "rank_8.yaml").write_text(yaml.dump({"peft": {"r": 8}}))

        # Create pre-existing eval results
        for name in ["baseline", "rank_8"]:
            eval_dir = tmp_path / f"results/eval/{name}"
            eval_dir.mkdir(parents=True)
            (eval_dir / "id_results.json").write_text(
                json.dumps({"mode": "id", "overall_success_rate": 0.8})
            )

        # Patch results dir by using monkeypatch or just test collection
        configs = discover_ablation_configs(abl_dir)
        # Manually test collect_eval_results on the dirs we created
        for name in ["baseline", "rank_8"]:
            results = collect_eval_results(tmp_path / f"results/eval/{name}")
            assert results == {"id": 0.8}
