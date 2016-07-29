import multiprocessing as mp
import os
import random
import tempfile
import json
from collections import OrderedDict

import numpy as np

def set_seed_and_run(process_idx, run_func):
    """
    sets random seed and run the process
    """
    seed = np.random.randint(0, 2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    run_func(process_idx)

def dqn_phi(screens):
    """Phi (feature extractor) of DQN for ALE
    Args:
      screens: List of N screen objects. Each screen object must be
      numpy.ndarray whose dtype is numpy.uint8.
    Returns:
      numpy.ndarray
    """
    assert len(screens) == 4
    assert screens[0].dtype == np.uint8
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [0, 1]
    raw_values /= 255.0
    return raw_values

# return sampled action here
def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


def prepare_output_dir(args, user_specified_dir=None):
    """Prepare output directory.

    An output directory is created if it does not exist. Then the following
    infomation is saved into the directory:
      args.txt: command-line arguments
      git-status.txt: result of `git status`
      git-log.txt: result of `git log`
      git-diff.txt: result of `git diff`

    Args:
      args: dict that describes command-line arguments
      user_specified_dir: directory path
    """
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        else:
            os.makedirs(user_specified_dir)
        outdir = user_specified_dir
    else:
        outdir = tempfile.mkdtemp()

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        f.write(json.dumps(vars(args)))

    return outdir



### ---------------- Setting value in shareed variables -------------------------
def set_shared_params(model, shared_weights ):
    """
    Args:
    model: model whose params are to be replaced
    shared_weights: list of new weight values arrays
    """
    
    params_dict = model.get_all_weights_params()
    vals_dict = model.get_all_weights()


    # convert both to lists
    params = []
    for k,v in params_dict.items():
        params += v

    vals = []
    for k,v in vals_dict.items():
        vals += v

    for p,v,nv in zip(params, vals, shared_weights):
        p_shape = v.shape
        p_type = v.dtype
        p.set_value(np.frombuffer(nv, dtype=p_type).reshape(p_shape))

### ---------------- Extracting value from shared variables -------------------------
def extract_params_as_shared_arrays(model):
    """
    converts params to shared arrays
    """
    # can get in the form of list -> shared + policy + value
    shared_arrays = []

    weights_dict = model.get_all_weights()
    weight_list = []


    for k,v in weights_dict.items():
        weight_list += v

    for weights in weight_list:
        shared_arrays.append(mp.RawArray('f', weights.ravel()))
    return shared_arrays


def share_params_as_shared_arrays(model):
    shared_arrays = extract_params_as_shared_arrays(model)
    set_shared_params(model, shared_arrays)
    return shared_arrays


### ---------------- Main thread executor -------------------------


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


def copy_param(self):
    """
    copy model params b/w threads?
    """
    pass