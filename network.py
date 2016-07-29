"""
Contains the architecture fir net, with train and fit methods
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import numpy as np

import cPickle as pickle

from params import * 
import lasagne

import logging
logger = logging.getLogger(__name__)

class DQN:

    def __init__(self, num_actions):
        
        # remember parameters
        self.num_actions = num_actions
        self.batch_size = BATCH_SIZE
        self.discount_rate = DISCOUNT_RATE
        self.history_length = HISTORY_LENGTH
        self.screen_dim = DIMS
        self.img_height = SCREEN_HEIGHT
        self.img_width = SCREEN_WIDTH
        self.clip_error = CLIP_ERROR
        self.input_color_scale = COLOR_SCALE

        self.target_steps = TARGET_STEPS
        self.train_iterations = TRAIN_STEPS
        self.train_counter = 0
        self.momentum = MOMENTUM
        self.update_rule = UPDATE_RULE
        self.learning_rate = LEARNING_RATE
        self.rms_decay = RMS_DECAY
        self.rms_epsilon = RMS_EPSILON        
        
        self.rng = np.random.RandomState(RANDOM_SEED)

        # set seed
        lasagne.random.set_rng(self.rng)

        # prepare tensors once and reuse them
        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        # terminals are bool for our case
        terminals = T.bcol('terminals')

        # create shared theano variables
        self.states_shared = theano.shared(
            np.zeros((self.batch_size, self.history_length, self.img_height, self.img_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((self.batch_size, self.history_length, self.img_height, self.img_width),
                     dtype=theano.config.floatX))

        # !broadcast ?
        self.rewards_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((self.batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            #np.zeros((self.batch_size, 1), dtype='int32'),
            np.zeros((self.batch_size, 1), dtype='int8'),
            broadcastable=(False, True))

        # can add multiple nets here
        self.l_primary = self.build_network()

        if self.target_steps > 0:
            self.l_secondary = self.build_network()
            self.copy_to_secondary()

        
        """
        # input scale i.e. division can be applied to input directly also to normalize
        """

        # define output symbols
        q_vals = lasagne.layers.get_output(self.l_primary, states / self.input_color_scale)
        
        if self.target_steps > 0:
            q_vals_secondary = lasagne.layers.get_output(self.l_secondary, next_states / self.input_color_scale)
        else:
            # why this ?
            q_vals_secondary = lasagne.layers.get_output(self.l_primary, next_states / self.input_color_scale)
            q_vals_secondary = theano.gradient.disconnected_grad(q_vals_secondary)

        # target = r + max
        target = (rewards + (T.ones_like(terminals) - terminals) * self.discount_rate * T.max(q_vals_secondary, axis=1, keepdims=True))
        
        """
        # check what this does
        """
        diff = target - q_vals[T.arange(self.batch_size),
                               actions.reshape((-1,))].reshape((-1, 1))

        # print shape ? 

        if self.clip_error > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_error)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_error * linear_part
        else:
            loss = 0.5 * diff ** 2

        loss = T.sum(loss)
        
        params = lasagne.layers.helper.get_all_params(self.l_primary)  
        
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }

        g_time = time.time()
        logger.info("graph compiling")


        if self.update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.learning_rate, self.rms_decay,
                                       self.rms_epsilon)
        elif self.update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.learning_rate, self.rms_decay,
                                              self.rms_epsilon)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})

        logger.info("Theano Graph Compiled !! %f", time.time() - g_time)

    def build_network(self):
        # create network
        
        from lasagne.layers import dnn
        #from lasagne.layers import Conv2DLayer

        l_in = lasagne.layers.InputLayer(
            shape=(self.batch_size, self.history_length, self.img_height, self.img_width )
        )

        l_conv1 = dnn.Conv2DDNNLayer(
            incoming=l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1)
            #dimshuffle=True
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            incoming=l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
            #dimshuffle=True
        )

        l_conv3 = dnn.Conv2DDNNLayer(
            incoming=l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
            #dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            incoming=l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=self.num_actions,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def copy_to_secondary(self):
        """
        copy params to secondary 
        """
        all_params = lasagne.layers.helper.get_all_param_values(self.l_primary)
        lasagne.layers.helper.set_all_param_values(self.l_secondary, all_params)


    def train(self, minibatch):
        # expand components of minibatch
        prestates, actions, rewards, poststates, terminals = minibatch
        assert len(prestates.shape) == 4
        assert len(poststates.shape) == 4
        assert len(actions.shape) == 1
        assert len(rewards.shape) == 1
        assert len(terminals.shape) == 1
        assert prestates.shape == poststates.shape
        assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

        # copy values to gpu
        self.states_shared.set_value(prestates)
        self.next_states_shared.set_value(poststates)


        actions = np.vstack(actions)
        terminals = np.vstack(terminals)
        rewards = np.vstack(rewards)
        
               
        self.actions_shared.set_value(actions)
        self.terminals_shared.set_value(terminals)
        self.rewards_shared.set_value(rewards)
        
        # copy to the network to seconday net
        if (self.target_steps > 0 and  self.train_counter % self.target_steps == 0):
            self.copy_to_secondary()
      
        loss, q_vals = self._train()  
       
        # increase number of weight updates (needed for target clone interval)
        self.train_counter += 1

        logger.debug("Loss %f", loss)
        return np.sqrt(loss)


    def predict(self, state):
        # checks here
        states = np.zeros((self.batch_size, self.history_length, self.img_height,
                           self.img_width), dtype=theano.config.floatX)
        states = state
        self.states_shared.set_value(states)
        # check what to return here
        return self._q_vals()[0]

    def load_weights(self, filename):
        """Unpickles and loads parameters into a Lasagne model."""
        weights = pickle.load( open( filename, "rb" ) )
        lasagne.layers.helper.set_all_param_values(self.l_primary, weights)

    def save_weights(self, filename):
        """Pickels the parameters within a Lasagne model."""
        weights = lasagne.layers.helper.get_all_param_values(self.l_primary)
        pickle.dump(weights, open(filename, "wb"))


def theano_rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

