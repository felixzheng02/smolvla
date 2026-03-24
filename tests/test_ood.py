"""Tests for src/ood.py — OOD paraphrasing and cross-suite transfer."""

import pytest

from src.ood import (
    LIBERO_PARAPHRASES,
    get_cross_suite_config,
    paraphrase_instruction,
)


class TestParaphraseKnown:
    """Test paraphrasing for known LIBERO-10 instructions."""

    @pytest.mark.parametrize("instruction", list(LIBERO_PARAPHRASES.keys()))
    def test_returns_nonempty_list(self, instruction: str):
        result = paraphrase_instruction(instruction)
        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.parametrize("instruction", list(LIBERO_PARAPHRASES.keys()))
    def test_paraphrases_differ_from_original(self, instruction: str):
        result = paraphrase_instruction(instruction)
        for p in result:
            assert p != instruction

    def test_known_instruction_case_insensitive(self):
        upper = "Pick Up The Black Bowl On The Left And Place It On The Plate"
        result = paraphrase_instruction(upper)
        assert len(result) >= 2

    def test_does_not_mutate_lookup(self):
        key = next(iter(LIBERO_PARAPHRASES))
        original = list(LIBERO_PARAPHRASES[key])
        result = paraphrase_instruction(key)
        result.append("should not appear in lookup")
        assert LIBERO_PARAPHRASES[key] == original


class TestParaphraseFallback:
    """Test the fallback path for unknown instructions."""

    def test_unknown_instruction_returns_list(self):
        result = paraphrase_instruction("pick up the red cube and stack it")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_unknown_instruction_applies_synonym(self):
        result = paraphrase_instruction("push the box to the corner")
        assert any("slide" in p or "move" in p for p in result)

    def test_completely_novel_returns_normalized(self):
        novel = "  Rotate The Arm 90 Degrees  "
        result = paraphrase_instruction(novel)
        assert isinstance(result, list)
        assert len(result) >= 1
        # Should at least return the normalized lowercase string
        assert result[0] == novel.lower().strip()


class TestCrossSuiteConfig:
    """Test cross-suite transfer config generation."""

    def test_returns_source_and_targets(self):
        config = get_cross_suite_config("libero_10")
        assert "source" in config
        assert "targets" in config

    def test_source_has_valid_repo_id(self):
        config = get_cross_suite_config("libero_10")
        assert config["source"]["repo_id"].startswith("lerobot/")

    def test_targets_have_valid_repo_ids(self):
        config = get_cross_suite_config("libero_10")
        for target in config["targets"]:
            assert target["repo_id"].startswith("lerobot/")
            assert "suite" in target

    def test_targets_exclude_source(self):
        config = get_cross_suite_config("libero_10")
        source_suite = config["source"]["suite"]
        for target in config["targets"]:
            assert target["suite"] != source_suite

    def test_unknown_suite_raises(self):
        with pytest.raises(ValueError, match="Unknown suite"):
            get_cross_suite_config("libero_nonexistent")

    @pytest.mark.parametrize(
        "suite", ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
    )
    def test_all_suites_return_targets(self, suite: str):
        config = get_cross_suite_config(suite)
        assert len(config["targets"]) >= 1
