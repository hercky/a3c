import multiprocessing as mp
import os
import random

import numpy as np

def set_random_seed(seed):
    """
    Sets the random seed
    """
    random.seed(seed)
    np.random.seed(seed)

def set_seed_and_run(process_idx, run_func):
    set_random_seed(np.random.randint(0, 2 ** 32))
    run_func(process_idx)


def run_async(n_process, run_func):
    """Run experiments asynchronously.

    Args:
      n_process (int): number of processes
      run_func: function that will be run in parallel
    """

    processes = []

    for process_idx in range(n_process):
        processes.append(mp.Process(target=set_seed_and_run, args=(
            process_idx, run_func)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()


def test_generator(prc_id):
    for i in xrange(3):
        print prc_id, np.random.randint(0, 100)


run_async(5,test_generator)     