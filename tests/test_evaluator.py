"""Tests for src/evaluator.py — rollouts, metrics, result persistence."""

import json

import pytest

from src.evaluator import EvalResults, RolloutResult, run_evaluation, run_rollout


# -- Mock env and policy --


class MockEnv:
    """Env that succeeds after a fixed number of steps."""

    def __init__(self, success_at: int = 5, max_reward: float = 1.0):
        self.success_at = success_at
        self.max_reward = max_reward
        self._step = 0

    def reset(self) -> dict:
        self._step = 0
        return {"state": [0.0] * 8, "image": [[0] * 3] * 4}

    def step(self, action):
        self._step += 1
        done = self._step >= self.success_at
        success = done
        reward = self.max_reward if success else 0.0
        info = {"success": success}
        obs = {"state": [float(self._step)] * 8, "image": [[self._step] * 3] * 4}
        return obs, reward, done, info

    def close(self):
        pass


class MockFailEnv(MockEnv):
    """Env that never succeeds."""

    def step(self, action):
        self._step += 1
        done = self._step >= self.success_at
        info = {"success": False}
        obs = {"state": [0.0] * 8}
        return obs, 0.0, done, info


def dummy_policy(obs: dict):
    """Policy that returns a fixed action."""
    return [0.0] * 7


class TestRolloutResult:
    def test_success_result(self):
        r = RolloutResult(success=True, steps=10, total_reward=1.0)
        assert r.success is True
        assert r.steps == 10

    def test_failure_result(self):
        r = RolloutResult(success=False, steps=400, total_reward=0.0)
        assert r.success is False


class TestRunRollout:
    def test_successful_rollout(self):
        env = MockEnv(success_at=5)
        result = run_rollout(env, dummy_policy, max_steps=100)
        assert result.success is True
        assert result.steps == 5
        assert result.total_reward == 1.0

    def test_timeout_rollout(self):
        env = MockEnv(success_at=999)
        result = run_rollout(env, dummy_policy, max_steps=10)
        assert result.success is False
        assert result.steps == 10

    def test_failing_env(self):
        env = MockFailEnv(success_at=5)
        result = run_rollout(env, dummy_policy, max_steps=100)
        assert result.success is False
        assert result.steps == 5

    def test_custom_success_fn(self):
        env = MockEnv(success_at=3)
        # Custom success: always True
        result = run_rollout(
            env, dummy_policy, max_steps=100,
            success_fn=lambda obs, info: True,
        )
        assert result.success is True

    def test_record_video(self):
        env = MockEnv(success_at=3)
        result = run_rollout(env, dummy_policy, max_steps=100, record_video=True)
        # Should have frames from steps with "image" in obs
        assert len(result.frames) > 0

    def test_no_video_by_default(self):
        env = MockEnv(success_at=3)
        result = run_rollout(env, dummy_policy, max_steps=100)
        assert result.frames == []


class TestRunEvaluation:
    def test_single_task(self):
        tasks = {"task_a": MockEnv(success_at=5)}
        results = run_evaluation(
            tasks, dummy_policy, mode="id", num_episodes=3, max_steps=100,
        )
        assert results.mode == "id"
        assert "task_a" in results.task_results
        assert len(results.task_results["task_a"]) == 3
        assert results.overall_success_rate == 1.0

    def test_multiple_tasks(self):
        tasks = {
            "success_task": MockEnv(success_at=3),
            "fail_task": MockFailEnv(success_at=3),
        }
        results = run_evaluation(
            tasks, dummy_policy, mode="test", num_episodes=4, max_steps=100,
        )
        assert results.per_task_success_rate["success_task"] == 1.0
        assert results.per_task_success_rate["fail_task"] == 0.0
        assert results.overall_success_rate == 0.5

    def test_empty_tasks(self):
        results = run_evaluation({}, dummy_policy, mode="test", num_episodes=5)
        assert results.overall_success_rate == 0.0
        assert results.summary()["total_episodes"] == 0


class TestEvalResults:
    def test_summary(self):
        results = EvalResults(mode="id")
        results.task_results["t1"] = [
            RolloutResult(success=True, steps=5),
            RolloutResult(success=False, steps=10),
        ]
        s = results.summary()
        assert s["mode"] == "id"
        assert s["overall_success_rate"] == 0.5
        assert s["per_task"]["t1"] == 0.5
        assert s["total_episodes"] == 2

    def test_save_and_load(self, tmp_path):
        results = EvalResults(mode="ood")
        results.task_results["t1"] = [
            RolloutResult(success=True, steps=5),
            RolloutResult(success=True, steps=3),
        ]
        path = tmp_path / "results.json"
        results.save(path)

        loaded = json.loads(path.read_text())
        assert loaded["mode"] == "ood"
        assert loaded["overall_success_rate"] == 1.0
        assert loaded["per_task"]["t1"] == 1.0

    def test_save_creates_parent_dirs(self, tmp_path):
        results = EvalResults(mode="test")
        path = tmp_path / "nested" / "dir" / "results.json"
        results.save(path)
        assert path.exists()

    def test_empty_results(self):
        results = EvalResults(mode="empty")
        assert results.overall_success_rate == 0.0
        assert results.summary()["total_episodes"] == 0
