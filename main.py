from __future__ import annotations

import argparse
import os
from typing import Literal

import jax.numpy as jnp
import numpy as np

from continualworld.envs import META_WORLD_TIME_HORIZON
from continualworld.interproc.interproc_evaluator import InterProcEvaluator
from continualworld import TASK_SEQUENCES, get_cl_env, ContinualWorldEnv
from continualworld.utils.eval import Evaluator, StandardEvaluator
from continualworld.utils.logger import Logger, TerminalLogger, WandbLogger
from rl.algorithms.sac import SACLearner
from rl.datasets.replay_buffer import ReplayBuffer


def _should_train(
    timestep: int, training_starts: int, gradient_update_interval: int
) -> bool:
    """Return whether a gradient-update block is due at this timestep."""
    return (
        timestep >= training_starts
        and (timestep - training_starts) % gradient_update_interval == 0
    )

def _should_eval(
    timestep: int, eval_starts: int, eval_interval: int
) -> bool:
    return (
        timestep >= eval_starts
        and (timestep - eval_starts) % eval_interval == 0
    )

def _get_loggers(args: argparse.Namespace) -> list[Logger]:
    loggers: list[Logger] = [TerminalLogger()]
    if args.wandb_project is not None:
        loggers.append(WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        ))

    return loggers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", choices=TASK_SEQUENCES, default="CW10")
    parser.add_argument("--steps-per-task", type=int, default=1_000_000)
    parser.add_argument("--episode-horizon", type=int, default=META_WORLD_TIME_HORIZON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--training-starts', type=int, default=4_000)
    parser.add_argument('--gradient-update-interval', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=10_000)
    parser.add_argument('--gradient-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--max-grad-norm', type=int, default=1.0)

    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--temp-lr', type=float, default=3e-4)
    parser.add_argument('--target-entropy', type=float, default=None)

    parser.add_argument('--eval-cpu-frac', type=float, default=None)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-name', type=str, default=None)

    return parser.parse_args()


def make_evaluator(
        env: ContinualWorldEnv,
        loggers: list[Logger],
        seed: int,
        mode: Literal['all', 'current', 'back'] = 'all',
        eval_cpu_frac: float | None = None,
        num_episodes: int = 15,
) -> Evaluator:
    if eval_cpu_frac is not None:
        assert 0.0 < eval_cpu_frac < 1.0, "Invalid fraction of CPUs to use for evaluation"

        available_cpus = sorted(os.sched_getaffinity(0))

        n_eval = int(len(available_cpus) * eval_cpu_frac)

        eval_cpus = available_cpus[-n_eval:]
        train_cpus = available_cpus[:-n_eval]

        print(f'Restricting train loop to {len(train_cpus)} CPUs')
        os.sched_setaffinity(0, train_cpus)

        return InterProcEvaluator(
            eval_cpus,
            env,
            loggers,
            seed,
            mode,
            num_episodes,
            True,
        )

    return StandardEvaluator(
        env,
        loggers,
        seed,
        mode,
        num_episodes,
    )


def main() -> None:
    args = parse_args()
    env = get_cl_env(
        ['push-v3'],
        args.steps_per_task,
        episode_horizon=args.episode_horizon,
        seed=args.seed,
        render_mode='rgb_array',
    )
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    num_episodes = 0
    num_success = 0
    current_timestep = 0
    success = False

    sac = SACLearner(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        temp_lr=args.temp_lr,
        seed=args.seed,
        observations=jnp.array(env.observation_space.sample()[np.newaxis]),
        actions=jnp.array(env.action_space.sample()[np.newaxis]),
        hidden_dims=(400, 400, 400),
        tau=0.005,
        target_entropy=-np.prod(env.action_space.shape[0]).item(),
        max_grad_norm=args.max_grad_norm,
    )

    replay_buffer = ReplayBuffer(
        env.observation_space,  # type: ignore[arg-type]
        env.action_space,       # type: ignore[arg-type]
        args.buffer_size,
    )

    loggers = _get_loggers(args)

    evaluator = make_evaluator(
        env,
        loggers,
        args.seed + 42,
        'current',
        args.eval_cpu_frac,
    )

    try:
        observation, _ = env.reset(seed=args.seed)
        while current_timestep < args.total_timesteps and not env.exhausted:
            if current_timestep < args.training_starts:
                action = env.action_space.sample()
            else:
                action = np.asarray(sac.sample_actions(observation))

            next_observation, reward, terminated, truncated, info = env.step(action)
            mask = 0.0 if terminated else 1.0
            done = terminated or truncated
            replay_buffer.insert(
                observation,
                action,
                reward,
                mask,
                float(done),
                next_observation,
            )

            success = success or bool(info["success"])
            current_timestep += 1
            for logger in loggers:
                logger.increase_timestep()

            if _should_train(
                current_timestep,
                args.training_starts,
                args.gradient_update_interval,
            ):
                print(f'training ({current_timestep}/{args.total_timesteps})')
                for _ in range(args.gradient_steps):
                    sac.update(replay_buffer.sample(batch_size=args.batch_size))

            if _should_eval(
                current_timestep,
                args.training_starts,
                args.eval_interval,
            ) and not env.exhausted:
                print("evaluating")
                evaluator.evaluate(current_timestep, sac)

            if done:
                num_episodes += 1
                if success:
                    num_success += 1
                success = False

                if not env.exhausted:
                    observation, _ = env.reset()
            else:
                observation = next_observation
    finally:
        evaluator.close()
        for logger in loggers:
            logger.close()
        env.close()

    print(
        f"finished {args.sequence}: {env.total_steps} transitions, "
        f"{num_success}/{num_episodes} successful episodes"
    )


if __name__ == "__main__":
    main()
