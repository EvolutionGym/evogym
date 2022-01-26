import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from evogym import *
from evogym.envs import BenchmarkBase

import random
import math
import numpy as np
import os

class Balance(BenchmarkBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Balancer-v0.json'))
        self.world.add_from_array('robot', body, 15, 3, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(1 + num_robot_points,), dtype=np.float)

    def get_obs(self, pos_final):
        com_final = np.mean(pos_final, 1)

        return np.array([
            17*self.VOXEL_SIZE - com_final[0],
            5.5*self.VOXEL_SIZE - com_final[1],
        ])

    def get_reward(self, pos_init, pos_final):
        com_init = np.mean(pos_init, 1)
        com_final = np.mean(pos_final, 1)
        
        reward = abs(17*self.VOXEL_SIZE - com_init[0]) - abs(17*self.VOXEL_SIZE - com_final[0])
        reward += (abs(5.0*self.VOXEL_SIZE - com_init[1]) - abs(5.0*self.VOXEL_SIZE - com_final[1]))
        return reward

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        
        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        ort = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            np.array([ort]),
            self.get_relative_pos_obs("robot"),
            ))

        # compute reward
        reward = self.get_reward(pos_1, pos_2)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        
        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs

        
class BalanceJump(BenchmarkBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Balancer-v1.json'))
        self.world.add_from_array('robot', body, 10, 1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(1 + num_robot_points,), dtype=np.float)

    def get_obs(self, pos_final):
        com_final = np.mean(pos_final, 1)

        return np.array([
            17.5*self.VOXEL_SIZE - com_final[0],
            6*self.VOXEL_SIZE - com_final[1],
        ])

    def get_reward(self, pos_init, pos_final):
        com_init = np.mean(pos_init, 1)
        com_final = np.mean(pos_final, 1)
        
        reward = abs(17.5*self.VOXEL_SIZE - com_init[0]) - abs(17.5*self.VOXEL_SIZE - com_final[0])
        reward += (abs(6*self.VOXEL_SIZE - com_init[1]) - abs(6*self.VOXEL_SIZE - com_final[1]))

        return reward

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        
        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        ort = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            np.array([ort]),
            self.get_relative_pos_obs("robot"),
            ))

        # compute reward
        reward = self.get_reward(pos_1, pos_2)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        
        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs