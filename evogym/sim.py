#!/usr/bin/env python

"""
This module defines the EvoSim class which provides a clean interface to Evolution Gym's simulator.
"""

from typing import Dict, Set
import numpy as np
from evogym.world import EvoWorld
from evogym.utils import *
from evogym.simulator_cpp import Sim

### TODO: Add documentation for current_time() and step()

class EvoSim(Sim):
    """
    Create, step, reset, and get data from an Evolution Gym simulation.

    Args:
        world (EvoWorld): object containing world voxel specification.
    """

    _has_displayed_version = False
    VOXEL_SIZE = 0.1

    def __init__(self, world: EvoWorld) -> None:

        if not EvoSim._has_displayed_version:
            Sim.get_version()
            EvoSim._has_displayed_version = True

        super().__init__()

        self.init(world.grid_size.x, world.grid_size.y)

        self._object_names: Set[str] = set()
        self._robot_name_to_actuator_indices: Dict[str, np.ndarray] = {}

        self._init_world_items(world)

    def _init_world_items(self, world: EvoWorld) -> None:
        """
        Initializes simulation from world voxel data.

        Args:
            world (EvoWorld): object containing world voxel specification.
        """

        # sort objects for consistency
        names_ordered_tup = []
        for name, obj in world.objects.items():
            #is robot
            if np.sum(obj.grid == VOXEL_TYPES['V_ACT']) + np.sum(
                    obj.grid == VOXEL_TYPES['H_ACT']) > 0:
                    names_ordered_tup.append((0, name))
            else:
                    names_ordered_tup.append((1, name))
        names_ordered_tup = sorted(names_ordered_tup)
        names_ordered = [name for (is_robot_value, name) in names_ordered_tup]

        # loads objects into sim
        for name in names_ordered:
            obj = world.objects[name]
            structure = np.flipud(obj.grid)

            is_robot = False
            if np.sum(obj.grid == VOXEL_TYPES['V_ACT']) + np.sum(
                    obj.grid == VOXEL_TYPES['H_ACT']) > 0:
                is_robot = True

            connections = []
            for y in range(obj.grid_size.y):
                for x in range(obj.grid_size.x):
                    if obj.grid[y][x] != VOXEL_TYPES['EMPTY'] and Pair(x, y) in obj.neighbors:
                        for nei in obj.neighbors[Pair(x, y)]:
                            oy = (obj.grid_size.y - 1) - y
                            nei_oy = (obj.grid_size.y - 1) - nei.y
                            connections.append([
                                oy * obj.grid_size.x + x,
                                nei_oy * obj.grid_size.x + nei.x
                            ])

            connections_numpy = np.array(connections).T
            if len(connections_numpy) == 0:
                connections_numpy = np.empty((2, 1))

            if is_robot:
                self.read_robot_from_array(structure, connections_numpy, obj.name,
                                           obj.pos.x, obj.pos.y)
                self._robot_name_to_actuator_indices[
                    obj.name] = self.get_indices_of_actuators(obj.name).flatten()
            else:
                self.read_object_from_array(structure, connections_numpy, obj.name,
                                            obj.pos.x, obj.pos.y)
            self._object_names.add(obj.name)

    def get_actuator_indices(self, robot_name: str) -> np.ndarray:
        """
        Returns the voxel indices a target robot's actuators in the simulation.

        Args:
            robot_name (str): name of robot.
        
        Returns:
            np.ndarray: `(n,)` array of actuator indices, where `n` is the number of actuators.
        """
        self._check_valid_robot_name(robot_name)
        return self._robot_name_to_actuator_indices[robot_name].copy()

    def get_dim_action_space(self, robot_name: str) -> int:
        """
        Returns the number of actuators for a target robot in the simulation.

        Args:
            robot_name (str): name of robot.
        
        Returns:
            int: number of actuators.
        """
        self._check_valid_robot_name(robot_name)
        return len(self._robot_name_to_actuator_indices[robot_name])

    def set_action(self, robot_name: str, action: np.ndarray) -> None:
        """
        Set an action for a target robot. This function updates the robot's actuator targets, but will not step the simulation.

        Args:
            robot_name (str): name of robot
            action (np.ndarray): `(n,)` array of actions, where `n` is the number of actuators of the target robot.
        """
        self._check_valid_robot_name(robot_name)
        indices = self._robot_name_to_actuator_indices[robot_name].copy()

        if indices.shape != action.flatten().shape:
            raise ValueError(
                f'expected action with {len(indices)} values but got action with {len(action.flatten())} values for {robot_name}'
            )
        informative_action = np.stack((indices, action.flatten()), axis=1)
        
        super().set_action(robot_name, informative_action)

    def _check_valid_time(self, time: int) -> None:
        """
        Throws an error if `time` is invalid.

        Args:
            time (int): time to check.
        """
        if not isinstance(time, int):
            raise TypeError(f'time {time} is not of type int')
        if time < 0 or time > super().get_time():
            raise ValueError(f'time {time} is outside valid range [{0}, {super().get_time()}]')
    
    def _check_valid_robot_name(self, robot_name: str) -> None:
        """
        Throws an error if no robot by the name 'robot_name' has been loaded into the simulation.

        Args:
            robot_name (str): name of robot to check.
        """
        if robot_name not in self._robot_name_to_actuator_indices:
            raise ValueError(f'{robot_name} is not a valid robot name')

    def _check_valid_object_name(self, object_name: str) -> None:
        """
        Throws an error if no object by the name 'object_name' has been loaded into the simulation.

        Args:
            object_name (str): object to check.
        """
        if object_name not in self._object_names:
            raise ValueError(f'{object_name} is not a valid object name')

    def pos_at_time(self, time: int) -> np.ndarray:
        """
        Returns positions of all point-masses in the simulation at time `time`. Use `EvoSim.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of points in the simulation.
        """
        self._check_valid_time(time)
        return super().pos_at_time(time)/self.VOXEL_SIZE

    def vel_at_time(self, time: int) -> np.ndarray:
        """
        Returns velocities of all point-masses in the simulation at time `time`. Use `EvoSim.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of points in the simulation.
        """
        self._check_valid_time(time)
        return super().vel_at_time(time)/self.VOXEL_SIZE
        
    def object_pos_at_time(self, time: int, object_name: str) -> np.ndarray:
        """
        Returns positions of all point-masses in a target object at time `time`. Use `EvoSim.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of point-masses in the target object.
        """
        self._check_valid_time(time)
        self._check_valid_object_name(object_name)
        return super().object_pos_at_time(time, object_name)/self.VOXEL_SIZE

    def object_vel_at_time(self, time: int, object_name: str) -> np.ndarray:
        """
        Returns velocities of all point-masses in a target object at time `time`. Use `EvoSim.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurements.
            object_name (str): name of object
        
        Returns:
            np.ndarray: `(2, n)` array of measurements, where `n` is the number of point-masses in the target object.
        """
        self._check_valid_time(time)
        self._check_valid_object_name(object_name)
        return super().object_vel_at_time(time, object_name)/self.VOXEL_SIZE

    def object_orientation_at_time(self, time: int, object_name: str) -> float:
        """
        Returns an estimate of the orientation of an object at time `time`. Use `EvoSim.get_time()` to get current measurements.

        Args:
            time (int): time at which to return measurement.
            object_name (str): name of object
        
        Returns:
            float: orientation with respect to x-axis in radians (increasing counter-clockwise) from the range [0, 2π].
        """
        self._check_valid_time(time)
        self._check_valid_object_name(object_name)
        return super().object_orientation_at_time(time, object_name)

    def reset(self,) -> None:
        """
        Reset the simulation to time `0`.
        """
        self.revert(0)