"""
Contains the architecture fir net, with train and fit methods
"""

import os
import sys
import time
from collections import OrderedDict

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


def rmsprop_updates(grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    """
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        try: 
            updates[param] = lasagne.updates.norm_constraint( param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon)) , MAX_NORM )
        except:
            updates[param] = param - (learning_rate * grad /
                                 T.sqrt(accu_new + epsilon))
        
    return updates

class Model:

    def __init__(self, num_actions):
        
        # remember parameters
        self.num_actions = num_actions
        
        # batch size is T_MAX now
        self.batch_size = 1 #BATCH_SIZE

        self.discount_rate = DISCOUNT_RATE
        
        self.history_length = HISTORY_LENGTH
        self.screen_dim = DIMS
        self.img_height = SCREEN_HEIGHT
        self.img_width = SCREEN_WIDTH
        
        self.beta = BETA

        self.learning_rate = LEARNING_RATE
        self.rms_decay = RMS_DECAY
        self.rms_epsilon = RMS_EPSILON        
        
        # prepare tensors once and reuse them
        state = T.tensor3('state')
        reward = T.fscalar('reward')
        advantage = T.fscalar('advantage')
        action = T.iscalar('action')
        
        #beta = T.fscalar('regularization_rate')
        # set learning rate 
        #self.shared_beta = theano.shared(np.zeros((1)), dtype=theano.config.floatX ,
        #        broadcastable=(True))
        #self.shared_beta.set_value([BETA])
        
        # create shared theano variables
        self.state_shared = theano.shared(
            np.zeros((self.history_length, self.img_height, self.img_width),
                     dtype=theano.config.floatX))

        self.reward_shared = theano.shared(
            np.zeros((1), dtype=theano.config.floatX))

        self.advantage_shared = theano.shared(
            np.zeros((1), dtype=theano.config.floatX))

        self.action_shared = theano.shared(
            np.zeros((1), dtype='int32'))

        # can add multiple nets here
        # Shared network parameters here
        self.shared_net = self.build_shared_network()
        shared_out = lasagne.layers.get_output(self.shared_net, state)

        ####### OPTIMIZATION here --------------
        # Policy network parameters here
        self.policy_network = self.build_policy_network(self.shared_net)
        policy_out = lasagne.layers.get_output(self.policy_network, state)

        # Value network parameters here
        self.value_network = self.build_value_network(self.shared_net)
        value_out = lasagne.layers.get_output(self.value_network, state)

        ## ----------------------- LOSS FUNCTION SHIT STARTS HERE ----------------------------------------

        # take log policy loss 
        policy_loss = -T.log(policy_out[0][self.action_shared]).dot(self.advantage_shared)

        # take entropy and add with the regularizer
        entropy = -T.tensordot(policy_out,T.log(policy_out)).dot(-1)
        
        # add regullazrization
        policy_loss += self.beta * entropy

        policy_loss = T.sum(policy_loss)
        
        # get the value loss
        value_loss = ((self.reward_shared - value_out)**2)/2
        value_loss = T.sum(value_loss)


        total_loss = T.sum(policy_loss + (0.5 * value_loss))

        ## ----------------------- LOSS FUNCTION SHIT ENDS HERE ----------------------------------------

        shared_params = lasagne.layers.helper.get_all_params(self.shared_net) 
        only_policy_params = self.policy_network.get_params()
        only_value_params = self.value_network.get_params()

        policy_params = shared_params + only_policy_params
        value_params = shared_params + only_value_params

        g_time = time.time()
        logger.info("graph compiling")

        # get grads here 
        policy_grad = T.grad(total_loss, policy_params)
        value_grad = T.grad(total_loss, value_params)

        # there'll be two kind of updates 
        policy_updates = rmsprop_updates(policy_grad, policy_params, self.learning_rate, self.rms_decay,
                                              self.rms_epsilon)
        
        value_updates = rmsprop_updates(value_grad, value_params, self.learning_rate, self.rms_decay,
                                              self.rms_epsilon)

        givens = {
            state: self.state_shared,
            reward: self.reward_shared,
            action: self.action_shared,
            advantage: self.advantage_shared,
        }

        # theano functions for accumulating the grads
        self._policy_grad = theano.function([], policy_grad, givens=givens)
        self._value_grad = theano.function([], value_grad, givens=givens)

        # train will take input the grads and just apply them

        # NEEDS work here ------------
        self._train_policy = theano.function(policy_grad, [], updates=policy_updates)
        self._train_value = theano.function(value_grad, [], updates=value_updates)

        # get output for a state
        self._policy = theano.function([], policy_out, givens={state: self.state_shared})
        self._value = theano.function([], value_out, givens={state: self.state_shared})

        # need more theano functions for getting policy and value
        logger.info("Theano Graph Compiled !! %f", time.time() - g_time)


    def build_shared_network(self):
        """
        This part contains trhe sharred params (conv layer) for both policy and value networks

        Returns the shared output
        """
        #from lasagne.layers import Conv2DLayer

        l_in = lasagne.layers.InputLayer(
            shape=(self.history_length, self.img_height, self.img_width)
        )

        l_in = lasagne.layers.ReshapeLayer(l_in, (1, self.history_length, self.img_height, self.img_width))

        #l_conv1 = dnn.Conv2DDNNLayer(
        l_conv1 = lasagne.layers.Conv2DLayer(
            incoming=l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1)
            #dimshuffle=True
        )

        #l1_out=l_conv1.get_output_shape_for((self.history_length, self.img_height , self.img_width))
        #print "L1:", l1_out

        l_conv2 = lasagne.layers.Conv2DLayer(
            incoming=l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
            #dimshuffle=True
        )

        #l2_out=l_conv2.get_output_shape_for(l1_out)
        #print "L2:", l2_out
        
        l_hidden1 = lasagne.layers.DenseLayer(
            incoming=l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_hidden1

    def build_policy_network(self, shared_out):
        """
        add a softmax layer to output coming from build_head

        another softmax layer 
        """
        l_policy = lasagne.layers.DenseLayer(
            incoming=shared_out,
            num_units=self.num_actions,
            nonlinearity= lasagne.nonlinearities.softmax,
            #W=lasagne.init.HeUniform(),
            #b=lasagne.init.Constant(.1)
        )
        
        return l_policy 


    def build_value_network(self, shared_out):
        """
        add a linear layer to output coming from build_head
        """
        l_value = lasagne.layers.DenseLayer(
            incoming=shared_out,
            num_units=1,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            #b=lasagne.init.Constant(.1)
        ) 

        return l_value
    
    def train(self, acc_policy_grad, acc_value_grad):
        """
        not working 
        """
        # asser chaecks
        self._train_policy(*acc_policy_grad)
        self._train_value(*acc_value_grad)  
        
        # success return true ?
        return True 

    def predict(self, state):
        # add assert here 
        self.state_shared.set_value(state)
        
        # get output here
        return self._policy(), self._value()

    def get_policy_and_value_grad(self, state, action, advantage, reward):

        # assert checks 
        #print "action", action 
        #print "advantage", advantage
        #print "reward", reward
        #print "state", state.shape

        reward = np.array([reward],dtype=np.float32)
        advantage = np.array([advantage],dtype=np.float32)
        action = np.array([action],dtype=np.int32)

        self.state_shared.set_value(state)
        self.reward_shared.set_value(reward)
        self.advantage_shared.set_value(advantage)
        self.action_shared.set_value(action)
        # same for action and adv
        return self._policy_grad() , self._value_grad()


    def get_value_grad(self, state, action, advantage):
        """
        """
        pass

    def get_all_weights(self):
        weights = OrderedDict()
        weights['shared'] =  lasagne.layers.helper.get_all_param_values(self.shared_net)
        weights['policy'] = [ param.get_value() for param in  self.policy_network.get_params() ]
        weights['value'] = [ param.get_value() for param in self.value_network.get_params() ]
        return weights

    def get_all_weights_params(self):
        weights_params = OrderedDict()
        weights_params['shared'] =  lasagne.layers.helper.get_all_params(self.shared_net)
        weights_params['policy'] = self.policy_network.get_params()
        weights_params['value'] = self.value_network.get_params()
        return weights_params

    def set_all_weights(self, weights):
        #lasagne.layers.helper.get_all_params(l_hidden1) + l_value.get_params() == lasagne.layers.helper.get_all_params(l_value)
        
        # dumb way
        #lasagne.layers.helper.set_all_param_values(self.value_network, weights['shared'] + weights['value'])
        #lasagne.layers.helper.set_all_param_values(self.policy_network, weights['shared'] + weights['policy'])
        
        lasagne.layers.helper.set_all_param_values(self.shared_net, weights['shared'])
        for p,w in zip(self.policy_network.get_params(), weights['policy']):
            p.set_value(w)
        for p,w in zip(self.value_network.get_params(), weights['value']):
            p.set_value(w)


    def load_weights(self, filename):
        """Unpickles and loads parameters into a Lasagne model."""
        weights = pickle.load( open( filename, "rb" ) )
        self.set_all_weights(weights)
        
        
    def save_weights(self, filename):
        """Pickels the parameters within a Lasagne model."""
        weights = self.get_all_weights()
        pickle.dump(weights, open(filename, "wb"))
