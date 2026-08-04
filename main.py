"""Minimal random-policy example for the Continual World v3 sequence."""

from __future__ import annotations

import argparse

from continualworld import TASK_SEQUENCES, get_cl_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random policy over a Continual World v3 task sequence."
    )
    parser.add_argument("--sequence", choices=TASK_SEQUENCES, default="CW10")
    parser.add_argument("--steps-per-task", type=int, default=200)
    parser.add_argument("--episode-horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = get_cl_env(
        args.sequence,
        args.steps_per_task,
        episode_horizon=args.episode_horizon,
        seed=args.seed,
    )
    env.action_space.seed(args.seed)

    episodes = 0
    successful_episodes = 0
    previous_task_index = None
    try:
        while not env.exhausted:
            _, reset_info = env.reset()
            if reset_info["sequence_index"] != previous_task_index:
                print(
                    f"task {reset_info['sequence_index'] + 1}/{env.num_tasks}: "
                    f"{reset_info['task_name']}"
                )
                previous_task_index = reset_info["sequence_index"]
            done = False
            final_info = reset_info
            while not done:
                action = (
                    env.action_space.sample()
                )  # The example policy is purely random.
                _, _, terminated, truncated, final_info = env.step(action)
                done = terminated or truncated

            episodes += 1
            successful_episodes += int(final_info["episode_success"])
    finally:
        env.close()

    print(
        f"finished {args.sequence}: {env.total_steps} transitions, "
        f"{successful_episodes}/{episodes} successful episodes"
    )


if __name__ == "__main__":
    main()
