import sys

import multiprocess as mp
import os
import time
from typing import Literal

from continualworld import ContinualWorldEnv
from continualworld.utils.eval import Evaluator, evaluate, make_video
from continualworld.utils.logger import Logger
from rl.algorithms.sac import SACLearner


def eval_process(
        cpu_ids: list[int],
        queue: mp.Queue,
        env: ContinualWorldEnv,
        loggers: list[Logger],
        seed: int,
        mode: Literal['all', 'current', 'back'] = 'all',
        num_episodes: int = 15,
        video: bool = False,
) -> None:
    os.sched_setaffinity(0, cpu_ids)
    print(f'Starting evaluation process with {len(cpu_ids)} cpus')
    sys.stdout.flush()

    # rebuild test environments due to render mode pickle problem
    env.build_test_envs()

    try:
        while True:
            message = queue.get()
            if message is None:
                break

            timestep, agent = message
            assert isinstance(agent, SACLearner), f'Received invalid agent type {type(agent)}, expected {SACLearner}'
            assert isinstance(timestep, int), f'Received invalid timestep type {type(timestep)}, expected int'

            start_time = time.time()
            evaluate(
                timestep=timestep,
                agent=agent,
                env=env,
                loggers=loggers,
                seed=seed,
                mode=mode,
                num_episodes=num_episodes,
            )
            if video:
                make_video(
                    timestep=timestep,
                    agent=agent,
                    env=env,
                    loggers=loggers,
                    seed=seed,
                    mode=mode,
                )

            print(f'Completed evaluation at timestep {timestep} in {time.time() - start_time:.1f} seconds')
            sys.stdout.flush()

    except KeyboardInterrupt:
        print('Received keyboard interrupt, stopping evaluation process')
    finally:
        for eval_env in env.test_envs:
            eval_env.close()
        for logger in loggers:
            logger.close()


class InterProcEvaluator(Evaluator):
    def __init__(
            self,
            cpus: list[int],
            env: ContinualWorldEnv,
            loggers: list[Logger],
            seed: int,
            mode: Literal['all', 'current', 'back'] = 'all',
            num_episodes: int = 15,
            video: bool = False,
    ) -> None:
        mp.set_start_method("spawn", force=True)

        super().__init__()
        self.cpus = cpus
        self.queue = mp.Queue()
        self.eval_proc = mp.Process(
            target=eval_process,
            args=(cpus, self.queue, env, loggers, seed, mode, num_episodes, video),
        )
        self.eval_proc.start()

    def evaluate(
            self,
            timestep: int,
            agent: SACLearner,
    ):
        self.queue.put((timestep, agent))

    def close(self) -> None:
        self.queue.put(None)
        self.eval_proc.join()
        self.queue.close()
