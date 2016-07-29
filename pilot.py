"""
Main class 

- loads rom 
- run 
- import net
"""

import sys
sys.setrecursionlimit(10000)

import os
import logging
import cPickle as pickle 
import numpy as np
import random
import time
import statistics
import argparse
import copy

import theano
import multiprocessing as mp


from self_network import Model
from agent import A3C

#from batch_network import Model
#from batch_agent import A3C

import ale_python_interface

import ale

from params import *
import util

import logging
logging.basicConfig(format='%(asctime)s %(message)s')

logger = logging.getLogger()

hdlr = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 

logger.setLevel(LOG_LEVEL)

# set seed
#random.seed(RANDOM_SEED)

def eval_performance(rom, p_func, n_runs):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        env = ale.ALE(rom, treat_life_lost_as_terminal=False)
        test_r = 0
        while not env.is_terminal:
            s = util.dqn_phi(env.state)
            pout = p_func(s)
            a = util.categorical_sample(pout)
            test_r += env.receive_action(a)
        scores.append(test_r)
        print 'test_',i,':',test_r
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def train_loop(process_idx, counter, max_score, args, agent, env, start_time):
    """
    process_idx -> process_id
    counter -> lock 
    max_score -> keeps track of max score so far
    args -> pass the params 
    start_time -> ?
    """
    try:

        total_r = 0
        episode_r = 0
        global_t = 0
        local_t = 0

        while True:

            # Get and increment the global counter

            # aquire lock
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
                # it releases lock here

            local_t += 1

            if global_t > args.steps:
                break

            # learning rate
            agent.model.learning_rate = (( args.steps - global_t - 1.0) / args.steps ) * args.lr

            total_r += env.reward
            episode_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if env.is_terminal:
                if process_idx == 0:
                    print 'global_t:',global_t,' local_t:',local_t,' lr:',float(agent.model.learning_rate),' episode_r:',episode_r
                # reset
                episode_r = 0
                #reset game
                env.initialize()
            else:
                # play that action
                env.receive_action(action)

            if global_t % args.eval_frequency == 0:
                # Evaluation

                # We must use a copy of the model because test runs can change
                # the hidden states of the model
                test_model = copy.deepcopy(agent.model)
                
                # reset_state ?
                #test_model.reset_state()

                def p_func(s):
                    pout, _ = test_model.predict(s)
                    #test_model.unchain_backward()
                    return pout

                mean, median, stdev = eval_performance(
                    args.rom, p_func, args.eval_n_runs)
                
                # enter in the log 
                with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (global_t, elapsed, mean, median, stdev)
                    f.write('\t'.join(str(x) for x in record) + '\n')
                
                with max_score.get_lock():
                    if mean > max_score.value:
                        # Save the best model so far
                        print 'The best score is updated {} -> {}'.format(max_score.value, mean)
                        filename = os.path.join(args.outdir, '{}.h5'.format(global_t))
                        agent.model.save_weights(filename)
                        print 'Saved the current best model to {}'.format(filename)
                        max_score.value = mean

    # Wow, that's how you take care of things !
    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            agent.model.save_weights(os.path.join(
                args.outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
            sys.stderr.write('Saved the current model to {}'.format(
                args.outdir))
        raise

    if global_t == args.steps + 1:
        # Save the final model
        agent.model.save_weights(
            os.path.join(args.outdir, '{}_finish.h5'.format(args.steps)))
        print('Saved the final model to {}'.format(args.outdir))


def main():

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)

    # sdl - for recording/displaying etc
    parser.add_argument('--use-sdl', action='store_true')
    
    # maximum 5 timesteps ?
    parser.add_argument('--t-max', type=int, default=5)
    
    # entropy 
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    #parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--lr', type=float, default=7e-3)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--use-lstm', action='store_true')
    parser.set_defaults(use_sdl=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.outdir = util.prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def model_opt():
        model = Model(n_actions)
        model.learning_rate = args.lr
        #opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)

        return model

    # creates network 
    model = model_opt()

    # shared stuff 
    shared_params = util.share_params_as_shared_arrays(model)

    # define locks here 
    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    # so fuckin awesome code
    with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        f.write('\t'.join(column_names) + '\n')


    def run_func(process_idx):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl)
        # creates local model
        model = model_opt()
        # set shared params
        util.set_shared_params(model, shared_params)

        # create an agent
        agent = A3C(model, args.t_max, beta=args.beta,
                        process_idx=process_idx, phi=util.dqn_phi)

        # train the loop
        train_loop(process_idx, counter, max_score,
               args, agent, env, start_time)

    util.run_async(args.processes, run_func)


if __name__ == '__main__':
    main()
