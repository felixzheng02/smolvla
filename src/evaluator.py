"""LIBERO policy evaluation: rollouts, metrics, and video recording."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class Env(Protocol):
    """Minimal environment interface for rollouts."""

    def reset(self) -> dict[str, Any]: ...
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict]: ...
    def close(self) -> None: ...


@dataclass
class RolloutResult:
    """Result of a single rollout episode."""

    success: bool
    steps: int
    total_reward: float = 0.0
    frames: list[Any] = field(default_factory=list)  # for video


@dataclass
class EvalResults:
    """Aggregated evaluation results across tasks and episodes."""

    mode: str
    task_results: dict[str, list[RolloutResult]] = field(default_factory=dict)

    @property
    def per_task_success_rate(self) -> dict[str, float]:
        rates = {}
        for task, results in self.task_results.items():
            if results:
                rates[task] = sum(r.success for r in results) / len(results)
            else:
                rates[task] = 0.0
        return rates

    @property
    def overall_success_rate(self) -> float:
        all_results = [r for rs in self.task_results.values() for r in rs]
        if not all_results:
            return 0.0
        return sum(r.success for r in all_results) / len(all_results)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "overall_success_rate": self.overall_success_rate,
            "per_task": self.per_task_success_rate,
            "total_episodes": sum(len(rs) for rs in self.task_results.values()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))


def run_rollout(
    env: Env,
    policy_fn: Callable[[dict[str, Any]], Any],
    max_steps: int = 400,
    success_fn: Callable[[dict[str, Any], dict], bool] | None = None,
    record_video: bool = False,
) -> RolloutResult:
    """Execute a single rollout episode.

    Args:
        env: Environment with reset/step interface.
        policy_fn: Maps observation dict to action.
        max_steps: Maximum steps before timeout.
        success_fn: Optional function (obs, info) -> bool to check success.
            If None, checks info["success"] from env.step.
        record_video: If True, store rendered frames.
    """
    obs = env.reset()
    total_reward = 0.0
    frames: list[Any] = []
    success = False

    for step_idx in range(max_steps):
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if record_video and "image" in obs:
            frames.append(obs["image"])

        # Check success
        if success_fn is not None:
            success = success_fn(obs, info)
        else:
            success = bool(info.get("success", False))

        if done:
            break

    return RolloutResult(
        success=success,
        steps=step_idx + 1,
        total_reward=total_reward,
        frames=frames if record_video else [],
    )


def run_evaluation(
    tasks: dict[str, Env],
    policy_fn: Callable[[dict[str, Any]], Any],
    mode: str,
    num_episodes: int = 20,
    max_steps: int = 400,
    success_fn: Callable[[dict[str, Any], dict], bool] | None = None,
    record_video: bool = False,
) -> EvalResults:
    """Evaluate a policy across multiple tasks and episodes.

    Args:
        tasks: Mapping of task_name -> environment.
        policy_fn: Maps observation dict to action.
        mode: Evaluation mode label (e.g., "id", "ood-instructions").
        num_episodes: Episodes per task.
        max_steps: Max steps per episode.
        success_fn: Optional custom success checker.
        record_video: Record frames from rollouts.
    """
    results = EvalResults(mode=mode)

    for task_name, env in tasks.items():
        task_rollouts: list[RolloutResult] = []
        for _ in range(num_episodes):
            rollout = run_rollout(
                env=env,
                policy_fn=policy_fn,
                max_steps=max_steps,
                success_fn=success_fn,
                record_video=record_video,
            )
            task_rollouts.append(rollout)
        results.task_results[task_name] = task_rollouts

    return results


def save_video(frames: list[Any], path: str | Path, fps: int = 10) -> None:
    """Save frames as MP4 video using matplotlib animation."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    if not frames:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(frame_idx: int):
        im.set_data(frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(str(path), fps=fps, writer="ffmpeg")
    plt.close(fig)
