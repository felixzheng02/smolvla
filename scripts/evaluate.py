#!/usr/bin/env python3
"""Part 4: Policy evaluation — ID and OOD conditions in LIBERO simulation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.evaluator import EvalResults, run_evaluation, save_video
from src.ood import get_cross_suite_config, paraphrase_instruction


def load_policy(checkpoint_path: str):
    """Load a fine-tuned SmolVLA policy from checkpoint.

    Handles both PEFT (adapter) and full checkpoints. For PEFT:
    1. Reads adapter_config.json to find base model path
    2. Loads base model
    3. Applies PEFT adapter weights
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(checkpoint_path)
    adapter_config = checkpoint / "adapter_config.json"

    if adapter_config.exists():
        # PEFT checkpoint — load base model then apply adapter
        import json

        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(checkpoint))
        base_path = peft_config.base_model_name_or_path
        if not base_path:
            raise ValueError("No base_model_name_or_path in adapter config")

        print(f"  Base model: {base_path}")
        policy = SmolVLAPolicy.from_pretrained(base_path)
        policy = PeftModel.from_pretrained(policy, str(checkpoint), config=peft_config)
    else:
        # Full checkpoint
        policy = SmolVLAPolicy.from_pretrained(str(checkpoint))

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

    Returns dict of task_name -> (env, init_states, language_instruction).
    """
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[task_suite](0)
    envs = {}

    for i in range(suite.get_num_tasks()):
        task = suite.get_task(i)
        bddl_file = suite.get_task_bddl_file_path(i)
        init_states = suite.get_task_init_states(i)

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        envs[task.name] = {
            "env": env,
            "init_states": init_states,
            "language": task.language,
        }

    return envs


def create_paraphrased_envs(
    base_envs: dict[str, any],
    task_instructions: dict[str, str],
) -> dict[str, any]:
    """Wrap existing envs with paraphrased instructions for OOD eval."""
    ood_envs = {}
    for task_name, env_info in base_envs.items():
        instruction = task_instructions.get(task_name, task_name)
        paraphrases = paraphrase_instruction(instruction)
        for i, para in enumerate(paraphrases[:2]):
            key = f"{task_name} (paraphrase {i + 1})"
            ood_envs[key] = {
                "env": env_info["env"],
                "init_states": env_info["init_states"],
                "language": para,
            }
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

    # Set EGL rendering for headless GPU
    os.environ.setdefault("MUJOCO_GL", "egl")

    out = Path(args.output_dir)

    # Load policy
    print(f"Loading policy from {args.checkpoint}")
    policy = load_policy(args.checkpoint)
    policy_fn = make_policy_fn(policy)

    if args.mode == "id":
        print(f"Creating {args.source_suite} environments for ID evaluation")
        env_configs = create_libero_envs(args.source_suite)
    elif args.mode == "ood-instructions":
        print("Creating environments with paraphrased instructions")
        env_configs = create_libero_envs(args.source_suite)
        task_instructions = {name: cfg["language"] for name, cfg in env_configs.items()}
        env_configs = create_paraphrased_envs(env_configs, task_instructions)
    elif args.mode == "ood-cross-suite":
        cross_config = get_cross_suite_config(args.source_suite)
        target = cross_config["targets"][0]
        print(f"Cross-suite eval: {args.source_suite} -> {target['suite']}")
        env_configs = create_libero_envs(target["suite"])

    # Extract just the envs for run_evaluation
    envs = {name: cfg["env"] for name, cfg in env_configs.items()}

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
    for cfg in env_configs.values():
        cfg["env"].close()


if __name__ == "__main__":
    main()
