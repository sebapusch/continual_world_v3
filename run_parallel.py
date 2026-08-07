import os
import multiprocessing as mp
from argparse import ArgumentParser
from time import sleep

import numpy as np


def evaluator(cpu_ids: list[int], queue: mp.Queue):
    os.sched_setaffinity(0, cpu_ids)
    print("Evaluator CPUs:", os.sched_getaffinity(0))

    item = queue.get()
    print("got", item)
    # evaluation loop ...


def main(
        eval_cpu_frac: float,
):
    assert 0.0 > eval_cpu_frac > 1.0, "Invalid fraction of CPUs to use for evaluation"

    available_cpus = sorted(os.sched_getaffinity(0))

    n_eval = int(len(available_cpus) * eval_cpu_frac)

    eval_cpus = available_cpus[-n_eval:]
    train_cpus = available_cpus[:-n_eval]

    # Restrict main training process
    os.sched_setaffinity(0, train_cpus)

    # Start evaluator
    queue = mp.Queue()

    eval_proc = mp.Process(
        target=evaluator,
        args=(eval_cpus, queue),
    )
    eval_proc.start()

    sleep(10)
    queue.put(np.array([1, 2, 3]))

    eval_proc.join()
    print("Training CPUs:", os.sched_getaffinity(0))




if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--eval-cpu-frac", type=float, default=0.1)
    main(
        **vars(args.parse_args())
    )