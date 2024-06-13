import gymnasium as gym
from gymnasium import error, spaces
from gymnasium import utils
from gymnasium.utils import seeding

from evogym import *
from evogym.envs import BenchmarkBase

import random
from math import *
import numpy as np
import os
from typing import Dict, Any, Optional

class ShapeBase(BenchmarkBase):
    
    def __init__(
        self,
        world: EvoWorld,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(world=world, render_mode=render_mode, render_options=render_options)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed, options=options)
        
        # observation
        obs = np.concatenate((
            self.get_relative_pos_obs("robot"),
            ))

        return obs, {}

    ### ----------------------------------------------------------------------

    # This section of code is modified from the following author 
    # from https://github.com/RodolfoFerro/ConvexHull

    # Author: Rodolfo Ferro 
    # Mail: ferro@cimat.mx
    # Script: Compute the Convex Hull of a set of points using the Graham Scan
    # Function to know if we have a CCW turn
    def CCW(self, p1, p2, p3):
        if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
            return True
        return False

    # Main function:
    def jarvis_march(self, S):
        n = len(S)
        P = [None] * n
        l = np.where(S[:,0] == np.min(S[:,0]))
        pointOnHull = S[l[0][0]]
        i = 0
        while True:
            P[i] = pointOnHull
            endpoint = S[0]
            for j in range(1,n):
                if (endpoint[0] == pointOnHull[0] and endpoint[1] == pointOnHull[1]) or not self.CCW(S[j],P[i],endpoint):
                    endpoint = S[j]
            i = i + 1
            pointOnHull = endpoint
            if endpoint[0] == P[0][0] and endpoint[1] == P[0][1]:
                break
        for i in range(n):
            if P[-1] is None:
                del P[-1]
        return np.array(P)

    ### ----------------------------------------------------------------------

    def convex_poly_area(self, pts_cw):
        area = 0
        for i in range(len(pts_cw)):
            i_1 = i + 1
            if i_1 >= len(pts_cw):
                i_1 = 0
            area += (pts_cw[i,0] * pts_cw[i_1,1] - pts_cw[i_1,0] * pts_cw[i,1])
        return 0.5 * area


class MaximizeShape(ShapeBase):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ShapeChange.json'))
        self.world.add_from_array('robot', body, 7, 1, connections=connections)

        # init sim
        ShapeBase.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(num_robot_points,), dtype=float)

    def step(self, action):

        # collect pre step information
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_relative_pos_obs("robot"),
            ))
       
        # compute reward
        reward = self.get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}
    
    def get_reward(self, robot_pos_init, robot_pos_final):

        # find convex hull of initial state 
        convex_hull_init = self.jarvis_march(np.transpose(robot_pos_init))
        area_init = self.convex_poly_area(convex_hull_init)

        # find convex of final state
        convex_hull_final = self.jarvis_march(np.transpose(robot_pos_final))
        area_final = self.convex_poly_area(convex_hull_final)
        
        reward = (area_final - area_init) * 10
    
        return reward

class MinimizeShape(ShapeBase):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ShapeChange.json'))
        self.world.add_from_array('robot', body, 7, 1, connections=connections)

        # init sim
        ShapeBase.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(num_robot_points,), dtype=float)

    def step(self, action):

        # collect pre step information
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_relative_pos_obs("robot"),
            ))
       
        # compute reward
        reward = self.get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}

    def get_reward(self, robot_pos_init, robot_pos_final):

        # find convex hull of initial state 
        convex_hull_init = self.jarvis_march(np.transpose(robot_pos_init))
        area_init = self.convex_poly_area(convex_hull_init)

        # find convex of final state
        convex_hull_final = self.jarvis_march(np.transpose(robot_pos_final))
        area_final = self.convex_poly_area(convex_hull_final)
        
        reward = (area_init - area_final) * 10
    
        return reward

class MaximizeXShape(ShapeBase):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ShapeChange.json'))
        self.world.add_from_array('robot', body, 7, 1, connections=connections)

        # init sim
        ShapeBase.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(num_robot_points,), dtype=float)

    def step(self, action):

        # collect pre step information
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_relative_pos_obs("robot"),
            ))
       
        # compute reward
        reward = self.get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}
    
    def get_reward(self, robot_pos_init, robot_pos_final):
        
        robot_min_pos_init = np.min(robot_pos_init, axis=1)
        robot_max_pos_init = np.max(robot_pos_init, axis=1)
        
        robot_min_pos_final = np.min(robot_pos_final, axis=1)
        robot_max_pos_final = np.max(robot_pos_final, axis=1)

        span_final = (robot_max_pos_final[0] - robot_min_pos_final[0]) 
        span_initial = (robot_max_pos_init[0] - robot_min_pos_init[0]) 
        
        reward = (span_final - span_initial)
    
        return reward

class MaximizeYShape(ShapeBase):

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'ShapeChange.json'))
        self.world.add_from_array('robot', body, 7, 1, connections=connections)

        # init sim
        ShapeBase.__init__(self, world=self.world, render_mode=render_mode, render_options=render_options)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(num_robot_points,), dtype=float)

    def step(self, action):

        # collect pre step information
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_relative_pos_obs("robot"),
            ))
       
        # compute reward
        reward = self.get_reward(robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, truncated, debugging info
        return obs, reward, done, False, {}
    
    def get_reward(self, robot_pos_init, robot_pos_final):
        
        robot_min_pos_init = np.min(robot_pos_init, axis=1)
        robot_max_pos_init = np.max(robot_pos_init, axis=1)
        
        robot_min_pos_final = np.min(robot_pos_final, axis=1)
        robot_max_pos_final = np.max(robot_pos_final, axis=1)

        span_final = (robot_max_pos_final[1] - robot_min_pos_final[1]) 
        span_initial = (robot_max_pos_init[1] - robot_min_pos_init[1]) 
        
        reward = (span_final - span_initial)
    
        return reward

