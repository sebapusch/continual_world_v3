"""Minimal random-policy example for the Continual World v3 sequence."""

from __future__ import annotations

import argparse

import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from continualworld import TASK_SEQUENCES, get_cl_env
from rl.algorithms.sac import SACLearner
from rl.datasets.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random policy over a Continual World v3 task sequence."
    )
    parser.add_argument("--sequence", choices=TASK_SEQUENCES, default="CW10")
    parser.add_argument("--steps-per-task", type=int, default=100_000)
    parser.add_argument("--episode-horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--training-starts', type=int, default=1000)
    parser.add_argument('--gradient-update-interval', type=int, default=1000)
    parser.add_argument('--gradient-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--total-timesteps', type=int, default=100_000)

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

    num_episodes = 0
    num_success = 0
    previous_task_index = 0
    current_timestep = 0
    success = False

    sac = SACLearner(
        seed=args.seed,
        observations=jnp.array(env.observation_space.sample()[np.newaxis]),
        actions=jnp.array(env.action_space.sample()[np.newaxis]),
    )

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, args.buffer_size)

    progress_bar = tqdm(total=args.total_timesteps)

    try:
        while current_timestep <= args.total_timesteps:
            observation, reset_info = env.reset(seed=args.seed)

            if reset_info["sequence_index"] != previous_task_index:
                break

            collected_timesteps = 0
            not_started = collected_timesteps < args.training_starts
            while (not_started or collected_timesteps < args.gradient_update_interval
                   and current_timestep + collected_timesteps < args.total_timesteps):

                action = env.action_space.sample() if not_started else np.array(sac.sample_actions(observation))

                next_observation, reward, terminated, truncated, info = env.step(action)
                mask = 0.0 if terminated else 1.0
                done = float(terminated or truncated)

                replay_buffer.insert(observation, action, reward, mask, done, next_observation)

                success = success or info['success']

                if done:
                    observation, reset_info = env.reset(seed=args.seed)

                    num_episodes += 1
                    if success:
                        num_success += 1

                    success = False
                else:
                    observation = next_observation

                collected_timesteps += 1

            current_timestep += collected_timesteps
            progress_bar.update(collected_timesteps)

            if (current_timestep - args.training_starts) % args.gradient_update_interval == 0:
                for i in range(args.gradient_steps):
                    sac.update(replay_buffer.sample(batch_size=args.batch_size))
    finally:
        env.close()


    print(
        f"finished {args.sequence}: {env.total_steps} transitions, "
        f"{num_success}/{num_episodes} successful episodes"
    )


if __name__ == "__main__":
    main()
