"""

Class for emulator, interface 

"""

import sys
import os
from ale_python_interface import ALEInterface
import cv2
import logging

#LOGGER
logger = logging.getLogger(__name__)


#Constants
from params import *

class Emulator:
    def __init__(self):
    
        self.ale = ALEInterface()
        
        # turn off the sound
        self.ale.setBool('sound', False)
        
        self.ale.setBool('display_screen', EMULATOR_DISPLAY)

        self.ale.setInt('frame_skip', FRAME_SKIP)
        self.ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
        self.ale.setBool('color_averaging', COLOR_AVERAGING)

        self.ale.setInt('random_seed', RANDOM_SEED)

        if RECORD_SCENE_PATH:
            self.ale.setString('record_screen_dir', RECORD_SCENE_PATH)


        self.ale.loadROM(ROM_PATH)

        self.actions = self.ale.getMinimalActionSet()
        logger.info("Actions: " + str(self.actions))

        self.dims = DIMS
        #self.start_lives = self.ale.lives()

    def getActions(self):
        return self.actions

    def numActions(self):
        return len(self.actions)

    def restart(self):
        self.ale.reset_game()
        # can be omitted

    def act(self, action):
        reward = self.ale.act(self.actions[action])
        return reward

    def getScreen(self):
        # why grayscale ?
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, self.dims)
        # normalize
        #resized /= COLOR_SCALE

        return resized

    def isTerminal(self):
        # while training deepmind only ends when agent dies
        #terminate = DEATH_END and TRAIN and (self.ale.lives() < self.start_lives)

        return self.ale.game_over()
