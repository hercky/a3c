import random
import time
import sys
import logging
import copy
import numpy as np
import os



logger = logging.getLogger(__name__)

import util
from params import *

class A3C(object):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, model,
                t_max , 
                beta = 1e-2, 
                process_idx=0, 
                phi=lambda x: x):

        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

        self.t_max = t_max
        self.gamma = DISCOUNT_RATE
        self.beta = beta
        
        self.process_idx = process_idx
        self.phi = phi

        self.t = 0 
        self.t_start = 0
        
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_actions = {}

        self.state_list = []
        self.reward_list = []
        self.adv_list = []
        self.action_list = []

    def sync_parameters(self):
        # copy params from shared_model to self_model
        shared_model_params = self.shared_model.get_all_weights()

        # set these new weights
        self.model.set_all_weights(shared_model_params)


    def act(self, state, reward, is_state_terminal):

        if MIN_REWARD is not None:
            reward = np.clip(reward, MIN_REWARD, MAX_REWARD)

        if not is_state_terminal:
            #assert shape
            state = self.phi(state)

        self.past_rewards[self.t - 1] = reward

        if (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max:

            assert self.t_start < self.t

            if is_state_terminal:
                R = 0
            else:
                _, vout = self.model.predict(state)
                R = float(vout)

            pi_loss = 0
            v_loss = 0

            #if self.process_idx == 0:
            #    logger.info('state:%s',state)

            # unroll for 
            for i in reversed(range(self.t_start, self.t)):

                # here we'll do unrolling 
                # and create obs, adv, reward pairs and send to model 
                # for learning as a batch


                R *= self.gamma
                R += self.past_rewards[i]
                v = self.past_values[i]
                
                                  
                advantage = R - v
                
                if self.process_idx == 0:
                    logger.debug('t:%s ',i)
                
                # create mini-batch
                self.state_list.append(self.past_states[i])
                self.reward_list.append(R)
                self.adv_list.append(advantage)
                self.action_list.append(self.past_actions[i])

            # call theano function for applying updates for rmpsProp (train)
            # pass these grads to main model and use it to update it 
            # CHECK THIS WANKER !!
            self.shared_model.train(self.state_list, self.action_list, self.adv_list, self.reward_list)

            # Copy the gradients to the globally shared model
            
            # Update the globally shared model
            if self.process_idx == 0:
                logger.debug('update')
            
            self.sync_parameters()
            # copy model params from main model here
            
            # reset
            self.past_states = {}
            self.past_rewards = {}
            self.past_values = {}
            self.past_actions = {}

            self.state_list = []
            self.adv_list = []
            self.reward_list = []
            self.action_list = []

            self.t_start = self.t

        if not is_state_terminal:

            # keep on playing per policy and store the info 

            self.past_states[self.t] = state
            pout, vout = self.model.predict(state)
            
            # these are symbolic outputs/ only then they make sense 
            # can be moved to the final calculation
            
            # we need this for advantage function
            self.past_values[self.t] = float(vout)

            
            action_sampled = util.categorical_sample(pout)

            self.past_actions[self.t] = action_sampled

            self.t += 1
            if self.process_idx == 0:
                logger.debug('t:%s, probs:%s',
                             self.t, pout)

            return action_sampled
        
        else:
            logger.debug('This is SPARTA!!!')
            return None

