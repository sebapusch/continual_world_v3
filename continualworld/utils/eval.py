from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from gymnasium import Env

from continualworld.envs import META_WORLD_TIME_HORIZON
from continualworld.utils.logger import Logger
from continualworld import ContinualWorldEnv
from rl.algorithms.sac import SACLearner


EvalMode = Literal['all', 'current', 'back']


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
            self,
            timestep: int,
            agent: SACLearner,
    ):
        ...

    def close(self) -> None:
        pass


class StandardEvaluator(Evaluator):
    def __init__(
            self,
            env: ContinualWorldEnv,
            loggers: list[Logger],
            seed: int,
            mode: Literal['all', 'current', 'back'] = 'all',
            num_episodes: int = 15,
    ) -> None:
        self.env = env
        self.loggers = loggers
        self.seed = seed
        self.mode = mode
        self.num_episodes = num_episodes

    def evaluate(
            self,
            timestep: int,
            agent: SACLearner,
    ):
        evaluate(
            timestep,
            agent,
            self.env,
            self.loggers,
            self.seed,
            self.mode,          # type: ignore
            self.num_episodes
        )


def _get_env_list(env: ContinualWorldEnv, mode: EvalMode) -> tuple[list[Env], list[str]]:
    match mode:
        case 'all':
            return env.test_envs, env.tasks
        case 'current':
            current = env.current_test_env
            if current is None:
                raise ValueError('No current test env')
            return [current], [env.tasks[env.current_task_index]]
        case 'back':
            return env.test_envs[:env.current_task_index], env.tasks[:env.current_task_index]


def make_video(
        timestep: int,
        agent: SACLearner,
        env: ContinualWorldEnv,
        loggers: list[Logger],
        seed: int,
        mode: EvalMode = 'all',
):
    for logger in loggers:
        logger.set_timestep(timestep)

    envs, env_names = _get_env_list(env, mode)

    for i, env in enumerate(envs):
        frames = []

        observation, _ = env.reset(seed=seed)

        for _ in range(META_WORLD_TIME_HORIZON):
            action = np.asarray(
                agent.sample_actions(observation, deterministic=True)
            )
            observation, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            frames.append(np.moveaxis(frame, -1, 0).astype(np.uint8))

            if terminated or truncated:
                break

        frames  = np.array(frames)
        for logger in loggers:
            if logger.supports_video:
                logger.log_video(f'video/{env_names[i]}', frames)


def evaluate(
        timestep: int,
        agent: SACLearner,
        env: ContinualWorldEnv,
        loggers: list[Logger],
        seed: int,
        mode: EvalMode = 'all',
        num_episodes: int = 15,
) -> None:
    for logger in loggers:
        logger.set_timestep(timestep)

    envs, env_names = _get_env_list(env, mode)

    for i, eval_env in enumerate(envs):
        episode = 0
        episodic_returns = np.zeros(num_episodes)
        num_success = 0

        observation, info = eval_env.reset(seed=seed)

        success = False
        while episode < num_episodes:
            action = np.asarray(
                agent.sample_actions(observation, deterministic=True)
            )

            next_observation, reward, terminated, truncated, info = eval_env.step(action)

            episodic_returns[episode] += reward

            success = success or info['success']
            done = terminated or truncated

            if done:
                episode += 1
                if success:
                    num_success += 1

                observation, info = eval_env.reset()
                success = False
            else:
                observation = next_observation

        for logger in loggers:
            logger.log(metric=f'eval/{env_names[i]}/avg_episodic_return', value=episodic_returns.mean())
            logger.log(metric=f'eval/{env_names[i]}/success_rate', value=num_success / num_episodes)

    for logger in loggers:
        logger.flush()
