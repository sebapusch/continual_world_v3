from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from continualworld.utils.logger import Logger
from continualworld import ContinualWorldEnv
from rl.algorithms.sac import SACLearner


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
            self,
            timestep: int,
            agent: SACLearner,
    ):
        ...

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


def evaluate(
        timestep: int,
        agent: SACLearner,
        env: ContinualWorldEnv,
        loggers: list[Logger],
        seed: int,
        mode: Literal['all', 'current', 'back'] = 'all',
        num_episodes: int = 15,
) -> None:
    for logger in loggers:
        logger.set_timestep(timestep)

    match mode:
        case 'all':
            env_names = env.tasks
            envs = env.test_envs
        case 'current':
            env_names = [env.tasks[env.current_task_index]]
            envs = [env.current_test_env]
        case 'back':
            env_names = env.tasks[:env.current_task_index]
            envs = env.test_envs[:env.current_task_index]
        case _:
            raise ValueError(f'Unknown mode: {mode}')

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
            else:
                observation = next_observation

        eval_env.close()

        for logger in loggers:
            logger.log(metric=f'eval/{env_names[i]}/avg_episodic_return', value=episodic_returns.mean())
            logger.log(metric=f'eval/{env_names[i]}/success_rate', value=num_success / num_episodes)



