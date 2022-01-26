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

class ClimbBase(BenchmarkBase):
    
    def __init__(self, world):
        
        super().__init__(world)

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs


class Climb0(ClimbBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Climber-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        ClimbBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points,), dtype=np.float)

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[1] - com_1[1])

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check termination condition
        if com_2[1] > (86)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

class Climb1(ClimbBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Climber-v1.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        ClimbBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points,), dtype=np.float)

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[1] - com_1[1])
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check termination condition
        if com_2[1] > (65)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

class Climb2(ClimbBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Climber-v2.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        ClimbBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 3

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_ceil_obs("robot", ["pipe"], self.sight_dist),
            ))

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[1] - com_1[1]) + (com_2[0] - com_1[0])*0.2

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
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_ceil_obs("robot", ["pipe"], self.sight_dist),
            ))

        return obs