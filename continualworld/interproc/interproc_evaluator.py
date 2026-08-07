import sys

import multiprocess as mp
import os
import time

from logging import Logger
from typing import Literal

from continualworld import ContinualWorldEnv
from continualworld.utils.eval import Evaluator, evaluate
from rl.algorithms.sac import SACLearner


def eval_process(
        cpu_ids: list[int],
        queue: mp.Queue,
        env: ContinualWorldEnv,
        loggers: list[Logger],
        seed: int,
        mode: Literal['all', 'current', 'back'] = 'all',
        num_episodes: int = 15,
) -> None:
    os.sched_setaffinity(0, cpu_ids)
    print(f'Starting evaluation process with {len(cpu_ids)} cpus')
    sys.stdout.flush()

    try:
        while True:
            timestep, agent = queue.get()
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
            print(f'Completed evaluation at timestep {timestep} in {time.time() - start_time:.1f} seconds')
            sys.stdout.flush()

    except KeyboardInterrupt:
        print('Received keyboard interrupt, stopping evaluation process')


class InterProcEvaluator(Evaluator):
    def __init__(
            self,
            cpus: list[int],
            env: ContinualWorldEnv,
            loggers: list[Logger],
            seed: int,
            mode: Literal['all', 'current', 'back'] = 'all',
            num_episodes: int = 15,
    ) -> None:
        mp.set_start_method("spawn", force=True)

        super().__init__()
        self.cpus = cpus
        self.queue = mp.Queue()
        self.eval_proc = mp.Process(
            target=eval_process,
            args=(cpus, self.queue, env, loggers, seed, mode, num_episodes),
        )
        self.eval_proc.start()


    def evaluate(
            self,
            timestep: int,
            agent: SACLearner,
    ):
        self.queue.put((timestep, agent))


