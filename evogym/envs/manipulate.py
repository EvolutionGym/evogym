from evogym.envs.base import EvoGymBase
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


class PackageBase(BenchmarkBase):
    
    def __init__(self, world):
        super().__init__(world)
        self.default_viewer.track_objects('robot', 'package')

    def get_obs(self, robot_pos_final, robot_vel_final, package_pos_final, package_vel_final):
        
        robot_com_pos = np.mean(robot_pos_final, axis=1)
        robot_com_vel = np.mean(robot_vel_final, axis=1)
        box_com_pos = np.mean(package_pos_final, axis=1)
        box_com_vel = np.mean(package_vel_final, axis=1)

        obs = np.array([
            robot_com_vel[0], robot_com_vel[1],
            box_com_pos[0]-robot_com_pos[0], box_com_pos[1]-robot_com_pos[1],
            box_com_vel[0], box_com_vel[1]
        ])

        return obs

    def get_reward(self, package_pos_init, package_pos_final, robot_pos_init, robot_pos_final):
        
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        package_com_pos_final = np.mean(package_pos_final, axis=1)
        
        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        # positive reward for moving forward
        reward = (package_com_pos_final[0] - package_com_pos_init[0])*0.75
        reward += (robot_com_pos_final[0] - robot_com_pos_init[0])*0.5

        # negative reward for robot/block separating
        reward += abs(robot_com_pos_init[0] - package_com_pos_init[0]) - abs(robot_com_pos_final[0] - package_com_pos_final[0])

        # negative reward for block going below thresh height
        if package_com_pos_final[1] < self.thresh_height:
            reward += 10 * (package_com_pos_final[1] - package_com_pos_init[1])
    
        return reward

    def reset(self):
        
        super().reset()

        # observation
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        obs = self.get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_relative_pos_obs("robot"),
        ))

        return obs


class CarrySmallRect(PackageBase):

    def __init__(self, body, connections=None):
        
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Carrier-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 3.0*self.VOXEL_SIZE

    def get_reward_carry(self, package_pos_init, package_pos_final, robot_pos_init, robot_pos_final):
        
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        package_com_pos_final = np.mean(package_pos_final, axis=1)
        
        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        # positive reward for moving forward
        reward = (package_com_pos_final[0] - package_com_pos_init[0])*0.5
        reward += (robot_com_pos_final[0] - robot_com_pos_init[0])*0.5

        # negative reward for block going below thresh height
        if package_com_pos_final[1] < self.thresh_height:
            reward += 10 * (package_com_pos_final[1] - package_com_pos_init[1])
    
        return reward

    def step(self, action):

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        # observation
        obs = super().get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_relative_pos_obs("robot"),
        ))
       
        # compute reward
        reward = self.get_reward_carry(package_pos_init, package_pos_final, robot_pos_init, robot_pos_final)

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check goal met
        com_2 = np.mean(robot_pos_final, 1)
        if com_2[0] > (99)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

class CarrySmallRectToTable(PackageBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Carrier-v1.json'))
        self.world.add_from_array('robot', body, 1, 4, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 6.0*self.VOXEL_SIZE

    def get_reward_carry(self, package_pos_init, package_pos_final, robot_pos_init, robot_pos_final):
        
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        package_com_pos_final = np.mean(package_pos_final, axis=1)
        
        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        # positive reward for moving block/robot to goal
        reward = 2*(abs(48.5*self.VOXEL_SIZE - package_com_pos_init[0]) - abs(48.5*self.VOXEL_SIZE - package_com_pos_final[0]))
        reward = (abs(40*self.VOXEL_SIZE - robot_com_pos_init[0]) - abs(40*self.VOXEL_SIZE - robot_com_pos_final[0]))

        self.thresh_height = 6.0*self.VOXEL_SIZE
        if package_com_pos_final[0] > 20 * self.VOXEL_SIZE:
            self.thresh_height = 4.0*self.VOXEL_SIZE

        # negative reward for block going below thresh height
        if package_com_pos_final[1] < self.thresh_height:
            reward += 10 * (package_com_pos_final[1] - package_com_pos_init[1])
    
        return reward

    def step(self, action):

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        # observation
        obs = super().get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_relative_pos_obs("robot"),
        ))
       
        # compute reward
        reward = self.get_reward_carry(package_pos_init, package_pos_final, robot_pos_init, robot_pos_final)
       
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

class PushSmallRect(PackageBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Pusher-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 0.0*self.VOXEL_SIZE

    def step(self, action):

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        # observation
        obs = super().get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_relative_pos_obs("robot"),
        ))

        # compute reward
        reward = super().get_reward(package_pos_init, package_pos_final, robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check goal met
        com_2 = np.mean(robot_pos_final, 1)
        if com_2[0] > (99)*0.1:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

class PushSmallRectOnOppositeSide(PackageBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Pusher-v1.json'))
        self.world.add_from_array('robot', body, 13, 1, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 0.0*self.VOXEL_SIZE

    def step(self, action):

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        # observation
        obs = super().get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_relative_pos_obs("robot"),
        ))

        # compute reward
        reward = super().get_reward(package_pos_init, package_pos_final, robot_pos_init, robot_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check goal met
        com_2 = np.mean(robot_pos_final, 1)
        if com_2[0] > (69)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

class ThrowSmallRect(PackageBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Thrower-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 0.0*self.VOXEL_SIZE

    def get_reward_throw(self, robot_pos_init, robot_pos_final, package_pos_init, package_pos_final):
        
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        package_com_pos_final = np.mean(package_pos_final, axis=1)

        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        motion_penalty = (robot_com_pos_init[0] - robot_com_pos_final[0])*0.25
        if robot_com_pos_final[0] < 0:
            motion_penalty *= -1

        reward = package_com_pos_final[0] - package_com_pos_init[0] + motion_penalty

        return reward

    def step(self, action):

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        # observation
        obs = super().get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_relative_pos_obs("robot"),
        ))
       
        # compute reward
        reward = self.get_reward_throw(robot_pos_init, robot_pos_final, package_pos_init, package_pos_final)
        
        #error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

class CatchSmallRect(PackageBase):

    def __init__(self, body, connections=None):

        self.robot_body = body
        self.robot_connections = connections

        self.random_init()
 
    def random_init(self):
        
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', self.robot_body, 22, 1, connections=self.robot_connections)

        self.offsetx = random.randint(-6, 4)
        self.offsety = random.randint(0, 5)

        package = WorldObject.from_json(os.path.join(self.DATA_PATH, 'package.json'))
        package.set_pos(21+self.offsetx, 41+self.offsety)
        package.rename('package')
        self.world.add_object(package)
        
        peg1 = WorldObject.from_json(os.path.join(self.DATA_PATH, 'peg.json'))
        peg1.set_pos(17+self.offsetx, 39+self.offsety)
        peg1.rename('peg1')
        self.world.add_object(peg1)

        peg2 = WorldObject.from_json(os.path.join(self.DATA_PATH, 'peg.json'))
        peg2.set_pos(19+self.offsetx, 25+self.offsety)
        peg2.rename('peg2')
        self.world.add_object(peg2)

        # init sim
        PackageBase.__init__(self, self.world)
        
        self.default_viewer.track_objects('robot', 'package')

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(7 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 5.0*self.VOXEL_SIZE
    
    def get_obs_catch(self, robot_pos_final, package_pos_final):
        
        robot_com_pos = np.mean(robot_pos_final, axis=1)
        package_com_pos = np.mean(package_pos_final, axis=1)

        obs = np.array([
            package_com_pos[0]-robot_com_pos[0], package_com_pos[1]-robot_com_pos[1],
        ])

        return obs

    def get_reward_catch(self, robot_pos_init, robot_pos_final, package_pos_init, package_pos_final):

        package_com_pos_init = np.mean(package_pos_init, axis=1)
        package_com_pos_final = np.mean(package_pos_final, axis=1)

        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        # negative reward for robot/block separating in X
        reward = abs(robot_com_pos_init[0] - package_com_pos_init[0]) - abs(robot_com_pos_final[0] - package_com_pos_final[0])

        # negative reward for block going below thresh height
        if package_com_pos_final[1] < self.thresh_height:
            reward += 10 * (package_com_pos_final[1] - package_com_pos_init[1])
    
        return reward

    def step(self, action):

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")

        # observation
        obs = self.get_obs_catch(robot_pos_final, package_pos_final)
        obs = np.concatenate((
            obs,
            self.get_vel_com_obs("robot"),
            self.get_vel_com_obs("package"),
            self.get_ort_obs("package"),
            self.get_relative_pos_obs("robot"),
        ))
       
        # compute reward
        reward = self.get_reward_catch(robot_pos_init, robot_pos_final, package_pos_init, package_pos_final)

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        EvoGymBase.reset(self)
        self.default_viewer.hide_debug_window()
        self.random_init()

        # self.translate_object(-self.offsetx*self.VOXEL_SIZE, -self.offsety*self.VOXEL_SIZE, "package")
        # self.translate_object(-self.offsetx*self.VOXEL_SIZE, -self.offsety*self.VOXEL_SIZE, "peg1")
        # self.translate_object(-self.offsetx*self.VOXEL_SIZE, -self.offsety*self.VOXEL_SIZE, "peg2")

        # self.offsetx = random.randint(-6, 4)
        # self.offsety = random.randint(0, 5)

        # self.translate_object(self.offsetx*self.VOXEL_SIZE, self.offsety*self.VOXEL_SIZE, "package")
        # self.translate_object(self.offsetx*self.VOXEL_SIZE, self.offsety*self.VOXEL_SIZE, "peg1")
        # self.translate_object(self.offsetx*self.VOXEL_SIZE, self.offsety*self.VOXEL_SIZE, "peg2")

        # observation
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")

        obs = self.get_obs_catch(robot_pos_final, package_pos_final)
        obs = np.concatenate((
            obs,
            self.get_vel_com_obs("robot"),
            self.get_vel_com_obs("package"),
            self.get_ort_obs("package"),
            self.get_relative_pos_obs("robot"),
        ))

        return obs

class ToppleBeam(PackageBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'BeamToppler-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(7 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 0.0*self.VOXEL_SIZE

        #tracking
        self.default_viewer.track_objects('robot', 'beam')
    
    def get_obs_topple(self, robot_pos_final, beam_pos_final):

        beam_com_pos_final = np.mean(beam_pos_final, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        diff = beam_com_pos_final - robot_com_pos_final
        return np.array([diff[0], diff[1]])

    def get_reward_topple(self, robot_pos_init, robot_pos_final, beam_pos_init, beam_pos_final):
        
        beam_com_pos_init = np.mean(beam_pos_init, axis=1)
        beam_com_pos_final = np.mean(beam_pos_final, axis=1)

        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        # rewarded for moving to beam
        reward = abs(beam_com_pos_init[0] - robot_com_pos_init[0]) - abs(beam_com_pos_final[0] - robot_com_pos_final[0])

        # reward for moving beam
        reward += abs(beam_com_pos_final[0] - beam_com_pos_init[0])*1.0
        reward += abs(beam_com_pos_final[1] - beam_com_pos_init[1])*3.0

        # reward for making beam fall
        reward += (beam_com_pos_init[1] - beam_com_pos_final[1])*10

        task_complete = False
        if beam_com_pos_final[0] < 2 * self.VOXEL_SIZE:
            task_complete = True

        return reward, task_complete

    def step(self, action):

        # collect pre step information
        beam_pos_init = self.object_pos_at_time(self.get_time(), "beam")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        beam_pos_final = self.object_pos_at_time(self.get_time(), "beam")

        # observation
        obs = self.get_obs_topple(robot_pos_final, beam_pos_final)
        obs = np.concatenate((
            obs,
            self.get_vel_com_obs("robot"),
            self.get_vel_com_obs("beam"),
            self.get_ort_obs("beam"),
            self.get_relative_pos_obs("robot"),
        ))
       
        # compute reward
        reward, task_complete = self.get_reward_topple(robot_pos_init, robot_pos_final, beam_pos_init, beam_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        if task_complete:
            reward += 1.0
            done = True

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        EvoGymBase.reset(self)

        # observation
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        beam_pos_final = self.object_pos_at_time(self.get_time(), "beam")

        obs = self.get_obs_topple(robot_pos_final, beam_pos_final)
        obs = np.concatenate((
            obs,
            self.get_vel_com_obs("robot"),
            self.get_vel_com_obs("beam"),
            self.get_ort_obs("beam"),
            self.get_relative_pos_obs("robot"),
        ))

        return obs

class SlideBeam(PackageBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'BeamSlider-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(7 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 0.0*self.VOXEL_SIZE

        # tracking
        self.default_viewer.track_objects('robot', 'beam')

    def get_obs_topple(self, robot_pos_final, beam_pos_final):

        beam_com_pos_final = np.mean(beam_pos_final, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        diff = beam_com_pos_final - robot_com_pos_final
        return np.array([diff[0], diff[1]])

    def get_reward_topple(self, robot_pos_init, robot_pos_final, beam_pos_init, beam_pos_final):
        
        beam_com_pos_init = np.mean(beam_pos_init, axis=1)
        beam_com_pos_final = np.mean(beam_pos_final, axis=1)

        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        # rewarded for moving to beam
        reward = abs(beam_com_pos_init[0] - robot_com_pos_init[0]) - abs(beam_com_pos_final[0] - robot_com_pos_final[0])

        # reward for moving beam (in positive x)
        reward += (beam_com_pos_final[0] - beam_com_pos_init[0])*1.0

        task_complete = False
        if beam_com_pos_final[0] < 2 * self.VOXEL_SIZE:
            task_complete = True

        return reward, task_complete

    def step(self, action):

        # collect pre step information
        beam_pos_init = self.object_pos_at_time(self.get_time(), "beam")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        beam_pos_final = self.object_pos_at_time(self.get_time(), "beam")

        # observation
        obs = self.get_obs_topple(robot_pos_final, beam_pos_final)
        obs = np.concatenate((
            obs,
            self.get_vel_com_obs("robot"),
            self.get_vel_com_obs("beam"),
            self.get_ort_obs("beam"),
            self.get_relative_pos_obs("robot"),
        ))
       
        # compute reward
        reward, task_complete = self.get_reward_topple(robot_pos_init, robot_pos_final, beam_pos_init, beam_pos_final)
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        if task_complete:
            reward += 1.0
            done = True

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        EvoGymBase.reset(self)

        # observation
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        beam_pos_final = self.object_pos_at_time(self.get_time(), "beam")

        obs = self.get_obs_topple(robot_pos_final, beam_pos_final)
        obs = np.concatenate((
            obs,
            self.get_vel_com_obs("robot"),
            self.get_vel_com_obs("beam"),
            self.get_ort_obs("beam"),
            self.get_relative_pos_obs("robot"),
        ))

        return obs

class LiftSmallRect(PackageBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Lifter-v0.json'))
        self.world.add_from_array('robot', body, 2, 3, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(7 + num_robot_points,), dtype=np.float)

    def get_reward_lift(self, robot_pos_init, robot_pos_final, package_pos_init, package_pos_final):
        
        package_com_pos_init = np.mean(package_pos_init, axis=1)
        package_com_pos_final = np.mean(package_pos_final, axis=1)

        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)

        reward = (package_com_pos_final[1] - package_com_pos_init[1])*10

        # penalize x movement
        goal = 5.5 * self.VOXEL_SIZE
        reward += (abs(goal-package_com_pos_init[0]) - abs(goal-package_com_pos_final[0]))*10

        # penalize robot falling below certain y com
        thresh = 3
        if robot_com_pos_final[1] < thresh*self.VOXEL_SIZE:
            reward += 20 * (robot_com_pos_final[1] - robot_com_pos_init[1])

        return reward

    def step(self, action):

        # collect pre step information
        package_pos_init = self.object_pos_at_time(self.get_time(), "package")
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        # observation
        obs = super().get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_ort_obs("package"),
            self.get_relative_pos_obs("robot"),
        ))
       
        # compute reward
        reward = self.get_reward_lift(robot_pos_init, robot_pos_final, package_pos_init, package_pos_final)

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}
    
    def reset(self):
        
        EvoGymBase.reset(self)

        # observation
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        package_pos_final = self.object_pos_at_time(self.get_time(), "package")
        package_vel_final = self.object_vel_at_time(self.get_time(), "package")

        obs = self.get_obs(robot_pos_final, robot_vel_final, package_pos_final, package_vel_final)
        obs = np.concatenate((
            obs,
            self.get_ort_obs("package"),
            self.get_relative_pos_obs("robot"),
        ))

        return obs
        