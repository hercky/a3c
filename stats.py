import sys
import csv
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Stats:
    def __init__(self, agent, net, mem, env):
        self.agent = agent
        self.net = net
        self.mem = mem
        self.env = env

        self.agent.callback = self

        self.start_time = time.clock()
        self.validation_states = None

    def reset(self):
        self.epoch_start_time = time.clock()
        self.num_steps = 0
        self.num_games = 0
        self.game_rewards = 0
        self.average_reward = 0
        self.min_game_reward = sys.maxint
        self.max_game_reward = -sys.maxint - 1
        self.last_exploration_rate = 1
        self.average_cost = 0

    # callback for agent
    def on_step(self, action, reward, terminal, screen, exploration_rate):
        self.game_rewards += reward
        self.num_steps += 1
        self.last_exploration_rate = exploration_rate

        if terminal:
            self.num_games += 1
            self.average_reward += float(self.game_rewards - self.average_reward) / self.num_games
            self.min_game_reward = min(self.min_game_reward, self.game_rewards)
            self.max_game_reward = max(self.max_game_reward, self.game_rewards)
            self.game_rewards = 0

    def on_train(self, cost):
        pass
        #self.average_cost += (cost - self.average_cost) / self.net.train_iterations

    def write(self, epoch, phase):
        current_time = time.clock()
        total_time = current_time - self.start_time
        epoch_time = current_time - self.epoch_start_time
        steps_per_second = self.num_steps / epoch_time

        if self.num_games == 0:
            self.num_games = 1
            self.average_reward = self.game_rewards

        if self.validation_states is None and self.mem.count > self.mem.batch_size:
            # sample states for measuring Q-value dynamics
            prestates, actions, rewards, poststates, terminals = self.mem.getMinibatch()
            self.validation_states = prestates
    
        logger.info("  num_games: %d, average_reward: %f, min_game_reward: %d, max_game_reward: %d" % 
            (self.num_games, self.average_reward, self.min_game_reward, self.max_game_reward))
        logger.info("  last_exploration_rate: %f, epoch_time: %ds, steps_per_second: %d" %
            (self.last_exploration_rate, epoch_time, steps_per_second))

    def close(self):
        pass
        