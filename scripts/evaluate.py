#!/usr/bin/env python3
"""Part 4: Policy evaluation — ID and OOD conditions in LIBERO simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluator import EvalResults, run_evaluation, save_video
from src.ood import get_cross_suite_config, paraphrase_instruction


def load_policy(checkpoint_path: str):
    """Load a fine-tuned SmolVLA policy from checkpoint."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.cuda()
    return policy


def make_policy_fn(policy):
    """Wrap a SmolVLA policy into a callable obs -> action."""
    import torch

    def policy_fn(obs: dict) -> any:
        with torch.no_grad():
            return policy.select_action(obs)

    return policy_fn


def create_libero_envs(task_suite: str = "libero_10") -> dict[str, any]:
    """Create LIBERO environments for a task suite.

    Returns dict of task_name -> env.
    """
    import robosuite as suite
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[task_suite]
    envs = {}

    for i in range(task_suite_obj.n_tasks):
        task = task_suite_obj.get_task(i)
        env = suite.make(
            "LIBERO_Kitchen_Tabletop_Manipulation",
            robots="Panda",
            bddl_file=task.bddl_file,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=256,
            camera_widths=256,
        )
        envs[task.name] = env

    return envs


def create_paraphrased_envs(
    base_envs: dict[str, any],
    task_instructions: dict[str, str],
) -> dict[str, any]:
    """Wrap existing envs with paraphrased instructions for OOD eval.

    Returns dict of "task_name (paraphrase N)" -> env.
    Note: the actual instruction override happens at inference time
    by modifying the language_instruction in the observation.
    """
    ood_envs = {}
    for task_name, env in base_envs.items():
        instruction = task_instructions.get(task_name, task_name)
        paraphrases = paraphrase_instruction(instruction)
        for i, para in enumerate(paraphrases[:2]):  # max 2 per task
            ood_envs[f"{task_name} (paraphrase {i + 1})"] = env
    return ood_envs


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy in LIBERO")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--mode",
        choices=["id", "ood-instructions", "ood-cross-suite"],
        required=True,
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/eval")
    parser.add_argument("--source-suite", type=str, default="libero_10")
    args = parser.parse_args()

    out = Path(args.output_dir)

    # Load policy
    print(f"Loading policy from {args.checkpoint}")
    policy = load_policy(args.checkpoint)
    policy_fn = make_policy_fn(policy)

    if args.mode == "id":
        print(f"Creating {args.source_suite} environments for ID evaluation")
        envs = create_libero_envs(args.source_suite)
    elif args.mode == "ood-instructions":
        print("Creating environments with paraphrased instructions")
        base_envs = create_libero_envs(args.source_suite)
        # TODO: extract real task instructions from dataset metadata
        task_instructions = {name: name for name in base_envs}
        envs = create_paraphrased_envs(base_envs, task_instructions)
    elif args.mode == "ood-cross-suite":
        cross_config = get_cross_suite_config(args.source_suite)
        target = cross_config["targets"][0]  # first transfer target
        print(f"Cross-suite eval: {args.source_suite} -> {target['suite']}")
        envs = create_libero_envs(target["suite"])

    # Run evaluation
    print(f"Running {args.num_episodes} episodes per task ({len(envs)} tasks)")
    results = run_evaluation(
        tasks=envs,
        policy_fn=policy_fn,
        mode=args.mode,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        record_video=args.record_video,
    )

    # Print and save results
    summary = results.summary()
    print(f"\nOverall success rate: {summary['overall_success_rate']:.2%}")
    print("Per-task success rates:")
    for task, rate in summary["per_task"].items():
        print(f"  {task}: {rate:.2%}")

    results.save(out / f"{args.mode}_results.json")
    print(f"\nResults saved to {out / f'{args.mode}_results.json'}")

    # Save videos if recorded
    if args.record_video:
        for task_name, rollouts in results.task_results.items():
            for i, rollout in enumerate(rollouts):
                if rollout.frames:
                    video_path = out / "videos" / f"{task_name}_ep{i}.mp4"
                    save_video(rollout.frames, video_path)

    # Close environments
    for env in envs.values():
        env.close()


if __name__ == "__main__":
    main()
