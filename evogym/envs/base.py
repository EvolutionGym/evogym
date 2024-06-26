
import gymnasium as gym
from gymnasium import error, spaces
from gymnasium import utils
from gymnasium.utils import seeding

from typing import Dict, Optional, List, Any
from evogym import *

import random
import math
import pkg_resources
import numpy as np
import os

class EvoGymBase(gym.Env):
    """
    Base class for all Evolution Gym environments.

    Args:
        world (EvoWorld): object specifying the voxel layout of the environment.
        render_mode (Optional[str]): values of `screen` and `human` will automatically render to a debug window every `step()`. If set to `img` or `rgb_array`, `render()` will return an image array. No rendering by default (default = None)
        render_options (Optional[Dict[str, Any]]): dictionary of rendering options. See EvoGymBase.render() for details (default = None)
    """
    
    metadata = {'render_modes': ['screen', 'human', 'img', 'rgb_array']}
    
    def __init__(
        self,
        world: EvoWorld,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ) -> None:

        # sim
        self._sim = EvoSim(world)
        self._default_viewer = EvoViewer(self._sim)
        
        # render
        self._render_mode = render_mode
        self._render_options = render_options

    def step(self, action: Dict[str, np.ndarray]) -> bool:
        """
        Step the environment by running physics computations.

        Args:
            action (Dict[str, np.ndarray]): dictionary mapping robot names to actions. Actions are `(n,)` arrays, where `n` is the number of actuators in the target robot.
            
        Returns:
            bool: whether or not the simulation has reached an unstable state and cannot be recovered (`True` = unstable).
        """
        #step
        for robot_name, a in action.items():
            a = np.clip(a, 0.6, 1.6)
            a[abs(a) < 1e-8] = 0
            self._sim.set_action(robot_name, a)
        done = self._sim.step()
        
        if self._render_mode == 'human' or self._render_mode == 'screen':
            self.render()

        return done

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Reset the simulation to the initial state.
        """
        self._sim.reset()

    @property
    def sim(self,) -> EvoSim:
        """
        Returns the environment's simulation.

        Returns:
            EvoSim: simulation object to return.
        """
        return self._sim

    @property
    def default_viewer(self,) -> EvoViewer:
        """
        Returns the environment's default viewer.

        Returns:
            EvoSim: viewer object to return.
        """
        return self._default_viewer
    
    def render(
        self,
    ) -> Optional[np.ndarray]:
        """
        Render the simulation according to the `render_mode` and `render_options` specified at initialization.
        The following rendering options are available as key-value pairs in the `render_options` dictionary:
        - `verbose` (bool): whether or not to print the rendering speed (rps) every second. (default = False)
        - `hide_background` (bool): whether or not to render the cream-colored background. If shut off background will be white. (default = False)
        - `hide_grid` (bool): whether or not to render the grid. (default = False)
        - `hide_edges` (bool): whether or not to render edges around all objects. (default = False)
        - `hide_voxels` (bool): whether or not to render voxels. (default = False)
        
        Returns:
            Optional[np.ndarray]: if `mode` is set to `img` or `rgb_array`, will return an image array. Otherwise, will return `None`.
        """
        mode, render_options = self._render_mode, {} if self._render_options is None else self._render_options
        if mode is None:
            return None
        
        verbose = render_options.get('verbose', False)
        hide_background = render_options.get('hide_background', False)
        hide_grid = render_options.get('hide_grid', False)
        hide_edges = render_options.get('hide_edges', False)
        hide_voxels = render_options.get('hide_voxels', False)
        
        return self.default_viewer.render(mode, verbose, hide_background, hide_grid, hide_edges, hide_voxels)

    def close(self) -> None:
        """
        Close the simulation.
        """
        self.default_viewer.close()
        del self._default_viewer
        del self._sim

    def get_actuator_indices(self, robot_name: str) -> np.ndarray:
        """
        Returns the voxel indices a target robot's actuators in the environment's simulation.

        Args:
            robot_name (str): name of robot.
        
        Returns:
            np.ndarray: `(n,)` array of actuator indices, where `n` is the number of actuators.
        """
        return self._sim.get_actuator_indices(robot_name)

    def get_dim_action_space(self, robot_name: str) -> int:
        """
        Returns the number of actuators for a target robot in the environment's simulation.

        Args:
            robot_name (str): name of robot.
        
        Returns:
            int: number of actuators.
        """
        return self._sim.get_dim_action_space(robot_name)

    def get_time(self, ) -> int:
        """
        Returns the current time as defined in the environment's simulator. Time starts at `0` and is incremented each time the environment steps. Time resets to `0` when the environment is reset.

        Returns:
            int: the current time.
        """
        return self._sim.get_time()

    def pos_at_time(self, time: int) -> np.ndarray:
        """
        Returns positions of all point-masses in the environment's simulation at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of points in the environment's simulation.
        """
        return self._sim.pos_at_time(time)

    def vel_at_time(self, time: int) -> np.ndarray:
        """
        Returns velocities of all point-masses in the environment's simulation at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of points in the environment's simulation.
        """
        return self._sim.vel_at_time(time)
        
    def object_pos_at_time(self, time: int, object_name: str) -> np.ndarray:
        """
        Returns positions of all point-masses in a target object at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of point-masses in the target object.
        """
        return self._sim.object_pos_at_time(time, object_name)

    def object_vel_at_time(self, time: int, object_name: str) -> np.ndarray:
        """
        Returns velocities of all point-masses in a target object at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of point-masses in the target object.
        """
        return self._sim.object_vel_at_time(time, object_name)

    def object_orientation_at_time(self, time: int, object_name: str) -> float:
        """
        Returns an estimate of the orientation of an object at time `time`. Use `EvoGymBase.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurement.
            object_name (str): name of object
        
        Returns:
            float: orientation with respect to x-axis in radians (increasing counter-clockwise) from the range [0, 2π].
        """
        return self._sim.object_orientation_at_time(time, object_name)  

    def get_pos_com_obs(self, object_name: str) -> np.ndarray:
        """
        Observation helper-function. Computes the position of the center of mass of a target object by averaging the positions of the object's point masses.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2,)` array of the position of the center of mass.
        """
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        return np.array([object_pos_com[0], object_pos_com[1]])

    def get_vel_com_obs(self, object_name: str) -> np.ndarray:
        """
        Observation helper-function. Computes the velocity of the center of mass of a target object by averaging the velocities of the object's point masses.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2,)` array of the velocity of the center of mass.
        """
        object_points_vel = self._sim.object_vel_at_time(self.get_time(), object_name)
        object_vel_com = np.mean(object_points_vel, axis=1)
        return np.array([object_vel_com[0], object_vel_com[1]])

    def get_relative_pos_obs(self, object_name: str):
        """
        Observation helper-function. Computes the positions of a target object's point masses relative to their center of mass.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2n,)` array of positions, where `n` is the number of point masses.
        """
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        return (object_points_pos-np.array([object_pos_com]).T).flatten()

    def get_ort_obs(self, object_name: str):
        """
        Observation helper-function. Returns the orientation of a target object.

        Args:
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(1,)` array of the object's orientation.
        """
        return np.array([self.object_orientation_at_time(self.get_time(), object_name)])

    def get_floor_obs(
        self, 
        object_name: str, 
        terrain_list: List[str], 
        sight_dist: int, 
        sight_range: float = 5) -> np.ndarray:
        """
        Observation helper-function. Computes an observation describing the shape of the terrain below the target object. Specifically, for each voxel to the left and right of the target object's center of mass (along with the voxel containing the center of mass), the following observation is computed: min(y-distance in voxels to the nearest terrain object below the target object's center of mass, `sight_range`). Results are returned in a 1D numpy array.

        Args:
            object_name (str): name of target object.
            terrain_list (List[str]): names of objects to be considered terrain in the computation.
            sight_dist (int): number of voxels to the left and right of the target object's center of mass for which an observation should be returned.
            sight_range (float): the max number of voxels below the object that can be seen. (default = 5)
        
        Returns:
            np.ndarray: `(2 * sight_range + 1, )` array of distance observations.
        """
        
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        
        if len(terrain_list) == 0:
            return None

        terrain_pos = self._sim.object_pos_at_time(self.get_time(), terrain_list[0])
        for i in range(1, len(terrain_list)):
            terrain_pos = np.concatenate((terrain_pos, self._sim.object_pos_at_time(self.get_time(), terrain_list[i])), axis = 1)

        right_mask = terrain_pos[0, :] > (object_pos_com[0] - (sight_dist+0.5))
        terrain_pos = terrain_pos[:, right_mask]

        left_mask = terrain_pos[0, :] < (object_pos_com[0] + (sight_dist+0.5))
        terrain_pos = terrain_pos[:, left_mask]

        bot_mask = terrain_pos[1, :] < (object_pos_com[1])
        terrain_pos = terrain_pos[:, bot_mask]

        elevations = np.zeros((sight_dist*2+1)) - sight_range + object_pos_com[1]
        for i in range(-sight_dist, sight_dist+1):
            less_than_mask = terrain_pos[0, :] > (object_pos_com[0] + (i-0.5))
            greater_than_mask = terrain_pos[0, :] < (object_pos_com[0] + (i+0.5))
            try:
                max_elevation = np.max(terrain_pos[1, (less_than_mask & greater_than_mask)])
                elevations[i+sight_dist] = max_elevation
            except:
                pass

        elevations = object_pos_com[1] - elevations
        elevations = np.clip(elevations, 0, sight_range)

        return elevations

    def get_ceil_obs(
        self, 
        object_name: str, 
        terrain_list: List[str], 
        sight_dist: int, 
        sight_range: float = 5) -> np.ndarray:
        """
        Observation helper-function. Computes an observation describing the shape of the terrain above the target object. Specifically, for each voxel to the left and right of the target object's center of mass (along with the voxel containing the center of mass), the following observation is computed: min(y-distance in voxels to the nearest terrain object above the target object's center of mass, `sight_range`). Results are returned in a 1D numpy array.

        Args:
            object_name (str): name of target object.
            terrain_list (List[str]): names of objects to be considered terrain in the computation.
            sight_dist (int): number of voxels to the left and right of the target object's center of mass for which an observation should be returned.
            sight_range (float): the max number of voxels above the object that can be seen. (default = 5)
        
        Returns:
            np.ndarray: `(2 * sight_range + 1, )` array of distance observations.
        """
        object_points_pos = self._sim.object_pos_at_time(self.get_time(), object_name)
        object_pos_com = np.mean(object_points_pos, axis=1)
        
        if len(terrain_list) == 0:
            return None

        terrain_pos = self._sim.object_pos_at_time(self.get_time(), terrain_list[0])
        for i in range(1, len(terrain_list)):
            terrain_pos = np.concatenate((terrain_pos, self._sim.object_pos_at_time(self.get_time(), terrain_list[i])), axis = 1)

        right_mask = terrain_pos[0, :] > (object_pos_com[0] - (sight_dist+0.5))
        terrain_pos = terrain_pos[:, right_mask]

        left_mask = terrain_pos[0, :] < (object_pos_com[0] + (sight_dist+0.5))
        terrain_pos = terrain_pos[:, left_mask]

        bot_mask = terrain_pos[1, :] > (object_pos_com[1])
        terrain_pos = terrain_pos[:, bot_mask]

        elevations = np.zeros((sight_dist*2+1)) + sight_range + object_pos_com[1]
        for i in range(-sight_dist, sight_dist+1):
            less_than_mask = terrain_pos[0, :] > (object_pos_com[0] + (i-0.5))
            greater_than_mask = terrain_pos[0, :] < (object_pos_com[0] + (i+0.5))
            try:
                max_elevation = np.min(terrain_pos[1, (less_than_mask & greater_than_mask)])
                elevations[i+sight_dist] = max_elevation
            except:
                pass

        elevations =  elevations - object_pos_com[1]
        elevations = np.clip(elevations, 0, sight_range)

        return elevations

class BenchmarkBase(EvoGymBase):

    DATA_PATH = pkg_resources.resource_filename('evogym.envs', os.path.join('sim_files'))
    VOXEL_SIZE = 0.1

    def __init__(
        self,
        world: EvoWorld,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):

        EvoGymBase.__init__(self, world=world, render_mode=render_mode, render_options=render_options)
        self.default_viewer.track_objects('robot')
    
    def pos_at_time(self, time):
        return super().pos_at_time(time)*self.VOXEL_SIZE

    def vel_at_time(self, time):
        return super().vel_at_time(time)*self.VOXEL_SIZE
        
    def object_pos_at_time(self, time, object_name):
        return super().object_pos_at_time(time, object_name)*self.VOXEL_SIZE

    def object_vel_at_time(self, time, object_name):
        return super().object_vel_at_time(time, object_name)*self.VOXEL_SIZE

    def get_pos_com_obs(self, object_name):
        return super().get_pos_com_obs(object_name)*self.VOXEL_SIZE

    def get_vel_com_obs(self, object_name):
        temp = super().get_vel_com_obs(object_name)*self.VOXEL_SIZE
        # print(f'child says super vel obs: {super().get_vel_com_obs(object_name)}\n')
        # print(f'vel obs: {temp}\n\n')
        return temp

    def get_relative_pos_obs(self, object_name):
        return super().get_relative_pos_obs(object_name)*self.VOXEL_SIZE

    def get_floor_obs(self, object_name, terrain_list, sight_dist, sight_range = 5):
        return super().get_floor_obs(object_name, terrain_list, sight_dist, sight_range)*self.VOXEL_SIZE

    def get_ceil_obs(self, object_name, terrain_list, sight_dist, sight_range = 5):
        return super().get_ceil_obs(object_name, terrain_list, sight_dist, sight_range)*self.VOXEL_SIZE