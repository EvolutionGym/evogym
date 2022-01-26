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

class Goal():
    def __init__(self, name, requirements = None):
        self.name = name
        self.requirements = requirements if requirements is not None else []
    def evaluate_reward(self, args):
        raise NotImplementedError("Your goal must implement an evaluate function which returns (reward, has_terminated).")

class GoalBase(BenchmarkBase):
    
    def __init__(self, world):
        super().__init__(world)

    def init_reward_goals(self, goals):

        if goals is None or len(goals) == 0:
            raise ValueError("Cannot create env with no goals") 

        self.goals = goals
        self.requirements = {}
        self.current_goal = 0

        for goal in goals:
            for req in goal.requirements:
                if req not in self.requirements:
                    self.requirements[req] = []
                self.requirements[req].append(goal.name)

    def get_reward(self, args):
        
        for req, dependents in self.requirements.items():
            if args[req] is None:
                raise ValueError(f'Args is missing requirement \'{req}\' for {dependents}')

        has_terminated = True
        reward = 0

        while (has_terminated and self.current_goal != len(self.goals)):
            reward, has_terminated = self.goals[self.current_goal].evaluate(args)
            if has_terminated:
                self.current_goal += 1

        done = False
        if self.current_goal == len(self.goals):
            done = True

        return reward, done

    def get_obs(self, args):
        return self.goals[self.current_goal].get_obs(args)

class WalkToX(Goal):

    def __init__(self, x_goal):
        super().__init__(f'Walk to x = {x_goal}', requirements = [
            'robot_com_pos_initial',
            'robot_com_pos_final',
        ])
        self.x_goal = x_goal

    def evaluate(self, args):
        com_init = np.mean(args['robot_com_pos_initial'], axis=1)
        com_final = np.mean(args['robot_com_pos_final'], axis=1)

        dist_init = abs(self.x_goal*0.1 - com_init[0])
        dist_final = abs(self.x_goal*0.1 - com_final[0])

        reward = dist_init - dist_final
        has_terminated = True if dist_final < 2*0.1 else False #2 blocks away

        #print(self.x_goal, com_final[0])

        return reward, has_terminated

    def get_obs(self, args):
        com_final = np.mean(args['robot_com_pos_final'], axis=1)
        return np.array([self.x_goal*0.1, self.x_goal*0.1 - com_final[0]])
    
class BiWalk(GoalBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'BidirectionalWalker-v0.json'))
        self.world.add_from_array('robot', body, 33, 1, connections=connections)

        # init sim
        GoalBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(5 + num_robot_points,), dtype=np.float)

        self.set_random_goals(20, 50, 100)

        # super().init_reward_goals([
        #     WalkToX(40),
        #     WalkToX(10),
        #     WalkToX(40),
        #     WalkToX(10)
        # ])

    def set_random_goals(self, lower_bound, upper_bound, goal_dist):
        
        curr_pos = 35
        dist = 0
        goals = []

        while dist < goal_dist:
            next_pos = random.randrange(lower_bound, upper_bound)
            dist += abs(curr_pos-next_pos)
            curr_pos = next_pos
            goals.append(WalkToX(next_pos))
            #print(f'Goal: {next_pos}')

        super().init_reward_goals(goals)

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        vel_2 = self.object_vel_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            np.array([self.current_goal]),
            ))

        obs = np.concatenate((obs, 
                super().get_obs(args = {
                'robot_com_pos_initial': pos_1,
                'robot_com_pos_final': pos_2
            })
        ))

        # compute reward
        reward, goals_done = super().get_reward(args = {
            'robot_com_pos_initial': pos_1,
            'robot_com_pos_final': pos_2
        })

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        if goals_done:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        self.current_goal = 0
        self.set_random_goals(20, 50, 100)

        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            np.array([self.current_goal]),
            ))

        obs = np.concatenate((obs, 
                super().get_obs(args = {
                'robot_com_pos_final': pos_2
            })
        ))

        return obs
