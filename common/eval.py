from abc import abstractmethod, ABC
from typing import Literal

import numpy as np

from continualworld import ContinualWorldEnv
from rl.algorithms.sac import SACLearner


class Logger(ABC):
    def __init__(self) -> None:
        self._timestep: int = 0

    def increase_timestep(self) -> None:
        self._timestep += 1

    @abstractmethod
    def log(self, metric: str, value: float) -> None:
        ...


class TerminalLogger(Logger):
    def log(self, metric: str, value: float) -> None:
        print(f'[{self._timestep}] {metric}: {value:.2f}')


def evaluate(
        agent: SACLearner,
        env: ContinualWorldEnv,
        loggers: list[Logger],
        seed: int,
        mode: Literal['all', 'current', 'back'] = 'all',
        num_episodes: int = 15,
) -> None:
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



