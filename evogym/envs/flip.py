import gymnasium as gym
from gymnasium import error, spaces
from gymnasium import utils
from gymnasium.utils import seeding

from evogym import *
from evogym.envs import BenchmarkBase

import random
import math
import numpy as np
import os
from typing import Dict, Any, Optional


class Flipping(BenchmarkBase):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Flipper-v0.json'))
        self.world.add_from_array('robot', body, 60, 1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(1 + num_robot_points,), dtype=float)

        # reward
        self.num_flips = 0


    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        ort_1 = self.object_orientation_at_time(self.get_time(), "robot")
        
        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        ort_2 = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            np.array([ort_2]),
            self.get_relative_pos_obs("robot"),
            ))

        # update flips
        flattened_ort_1 = self.num_flips * 2 * math.pi + ort_1

        if ort_1 < math.pi/3 and ort_2 > 5*math.pi/3:
            self.num_flips -= 1
        if ort_1 > 5*math.pi/3 and ort_2 <  math.pi/3:
            self.num_flips += 1

        flattened_ort_2 = self.num_flips * 2 * math.pi + ort_2

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (flattened_ort_2 - flattened_ort_1) 

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        
        # check goal met
        if com_2[0] < (1)*self.VOXEL_SIZE:
            done = True


        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed, options=options)

        self.num_flips = 0

        # observation
        obs = np.concatenate((
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))

        return obs, {}