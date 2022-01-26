#!/usr/bin/env python

"""
This module defines the EvoWorld and WorldObject classes which provide a clean interface to store and manipulate objects in a Evolution Gym environment.
"""

from __future__ import annotations
from typing import Dict, Optional, Any, List
import json
import numpy as np
from evogym.utils import *


class EvoWorld():
    """
    Specify the layout of an Evolution Gym environment.
    """

    def __init__(self,) -> None:
        self.grid = np.array([[0]])
        self.grid_size = Pair(1, 1)
        self.objects: Dict[str, WorldObject] = {}

    @classmethod
    def from_json(cls, file_path: str) -> EvoWorld:
        """
        Create an `EvoWorld` object from a `.json` environment specification file (these can be created using the EvoGym Design Tool) and return it.

        Args:
            file_path (str): path to file. Ex: `my_env.json`.
        
        Returns:
            EvoWorld: resulting `EvoWorld` object.
        """
        out = cls()
        out.add_from_json(file_path)
        return out

    def add_from_json(self, file_path: str) -> None:
        """
        Add objects to an existing `EvoWorld` object from a `.json` environment specification file. These can be created using the EvoGym Design Tool.

        Args:
            file_path (str): path to file. Ex: `my_env.json`.
        """

        with open(file_path, 'r') as infile:
            state = json.load(infile)

        file_grid_size = Pair(state['grid_width'], state['grid_height'])

        # read in objects
        for name, obj_data in state['objects'].items():

            # assert lists of same length
            if not len(obj_data['indices']) == len(obj_data['types']):
                raise ValueError(
                    f'cannot read in file {file_path} with corrupted object {name}'
                )
            if not len(obj_data['indices']) == len(obj_data['neighbors']):
                raise ValueError(
                    f'cannot read in file {file_path} with corrupted object {name}'
                )

            if name in self.objects:
                raise ValueError(
                    f'cannot read in file {file_path} with duplicate object name: {name}'
                )

            obj = WorldObject()
            obj.load_from_parsed_json(name, obj_data, file_grid_size)
            self.add_object(obj)

    def add_from_array(
        self, 
        name: str, 
        structure: np.ndarray, 
        x: int, 
        y: int, 
        connections: Optional[np.ndarray] = None) -> None:
        """
        Add a single object to the world from array.

        Args:
            name (str): object name.
            structure (np.ndarray): `(n, m)` array specifing the voxel structure of the object. See `evogym.VOXEL_TYPES`. 
            x (int): x-position of the bottom & leftmost voxel of the object. Starts at `0`.
            y (int): y-position of the bottom & leftmost voxel of the object. Starts at `0`.
            connections (Optional[np.ndarray]): `(2, k)` array specifying `k` pairwise voxel connections. Voxels are specified by their index into the 1D array `np.flatten(structure)`. The default behavior assumes all adjacent voxels are connected. (default = None)
        """
        new_obj = WorldObject.from_array(name, structure, connections)
        new_obj.set_pos(x, y)
        self.add_object(new_obj)

    def add_object(self, obj: WorldObject) -> None:
        """
         Add a single object to the world.

        Args:
            obj (WorldObject): object to add.
        """
        if obj.name in self.objects:
            raise ValueError(
                f'cannot add object with duplicate name: {obj.name}')

        x_start = obj.pos.x
        x_end = obj.pos.x + obj.grid_size.x

        y_start = obj.pos.y
        y_end = obj.pos.y + obj.grid_size.y

        # extend grid if needed
        if x_end > self.grid_size.x or y_end > self.grid_size.y:
            temp_grid = self.grid.tolist()

            # extend y
            while y_end > self.grid_size.y:
                temp_grid.append([])
                for x in range(self.grid_size.x):
                    temp_grid[-1].append(0)
                self.grid_size.y += 1

            # extend x
            while x_end > self.grid_size.x:
                for y in range(self.grid_size.y):
                    temp_grid[y].append(0)
                self.grid_size.x += 1

            self.grid = np.array(temp_grid)

        if np.sum(self.grid[y_start:y_end, x_start:x_end]) != 0:
            raise ValueError(
                f'cannot add object {obj.name} to world since it overlaps with an existing object'
            )

        self.grid[y_start:y_end, x_start:x_end] = obj.grid.copy()
        self.objects[obj.name] = obj

    def remove_object(self, obj_name: str) -> WorldObject:
        """
        Remove an object from world by name and return it.

        Args:
            obj_name (str): object name.
        
        Returns:
            WorldObject: removed, returned object.
        """
        if obj_name not in self.objects:
            raise ValueError(
                f'world does not contain object named {obj_name}')
        obj = self.objects[obj_name].copy()

        x_start = obj.pos.x
        x_end = obj.pos.x + obj.grid_size.x

        y_start = obj.pos.y
        y_end = obj.pos.y + obj.grid_size.y

        self.grid[y_start:y_end, x_start:x_end] = 0
        del self.objects[obj_name]

        return obj

    def translate_object(self, obj_name: str, dx: int, dy: int) -> None:
        """
        Translate an object in world by name.

        Args:
            obj_name (str): object name.
            dx (int): change in x-position.
            dy (int): change in y-position.
        """
        if obj_name not in self.objects:
            raise ValueError(
                f'world does not contain object named {obj_name}')

        obj = self.remove_object(obj_name)
        obj.translate(dx, dy)
        try:
            self.add_object(obj)
        except ValueError:
            raise ValueError(
                f'cannot translate object {obj.name} in world since the translated object overlaps with an existing object'
            )

    def move_object(self, obj_name: str, x: int, y: int) -> None:
        """
        Move an object in world by name.

        Args:
            obj_name (str): object name.
            x (int): x-position of the bottom & leftmost voxel of the object. Starts at `0`.
            y (int): y-position of the bottom & leftmost voxel of the object. Starts at `0`.
        """
        if obj_name not in self.objects:
            raise ValueError(
                f'world does not contain object named {obj_name}')

        obj = self.remove_object(obj_name)
        obj.set_pos(x, y)
        try:
            self.add_object(obj)
        except ValueError:
            raise ValueError(
                f'cannot move object {obj.name} to desired location since it will overlap with an existing object'
            )

    def pretty_print(self, voxels_per_line: int = 50) -> None:
        """
        Print world to console for debugging. Voxels are specified by (R)igid, (S)oft, (H)orizontal Actuator, (V)ertical Actuator, (F)ixed.

        Args:
            voxels_per_line (int): Number of voxels to print per line -- reduce for smaller screens. (default = 50)
        """
        print_values = {
            0: ". ",
            1: "R ",
            2: "S ",
            3: "H ",
            4: "V ",
            5: "F ",
        }
        for start_x in range(0, self.grid_size.x, voxels_per_line):
            for y in reversed(range(self.grid_size.y)):
                print(f'\n{y%10} | ', end='')
                for x in range(start_x,
                               min(self.grid_size.x,
                                   start_x + voxels_per_line)):
                    print(print_values[self.grid[y][x]], end='')

            print(f'\n   -', end='')
            for x in range(start_x,
                           min(self.grid_size.x, start_x + voxels_per_line)):
                print('--', end='')
            print(f'\n    ', end='')
            for x in range(start_x,
                           min(self.grid_size.x, start_x + voxels_per_line)):
                print(f'{x%10} ', end='')
            print()


class WorldObject():
    """
    Store and manipulate objects in a Evolution Gym environment.
    """

    def __init__(self) -> None:
        self.name = ''
        self.pos = Pair(0, 0)
        self.grid_size = Pair(0, 0)

        self.voxels: List[Pair] = []
        self.grid: np.ndarray = np.zeros((0,0))
        self.neighbors: Dict[Pair, List[Pair]] = {}

    @classmethod
    def from_json(cls, file_path: str) -> WorldObject:
        """
        Load object from a `.json` environment specification file (these can be created using the EvoGym Design Tool) and return it. Throws a `ValueError` if the file contains more than one object.

        Args:
            file_path (str): path to file. Ex: `my_env.json`.
        
        Returns:
            WorldObject: resulting `WorldObject` object.
        """
        world = EvoWorld.from_json(file_path)
        if len(world.objects) != 1:
            raise ValueError(
                f'loaded file, {file_path}, contains more than one object')
        names = [name for name in world.objects]
        return world.objects[names[0]]

    def load_from_json(self, file_path: str) -> None:
        """
        Load object from a `.json` environment specification file (these can be created using the EvoGym Design Tool). Throws a `ValueError` if the file contains more than one object.

        Args:
            file_path (str): path to file. Ex: `my_env.json`.
        """
        temp = WorldObject.from_json(file_path)
        self.name, self.pos, self.grid_size = temp.name, temp.pos, temp.grid_size
        self.voxels, self.grid, self.neighbors = temp.voxels, temp.grid, temp.neighbors

    @classmethod
    def from_array(
        cls, 
        name: str, 
        structure: np.ndarray, 
        connections: Optional[np.ndarray] = None) -> WorldObject:
        """
        Load an object from array and return it.

        Args:
            name (str): object name.
            structure (np.ndarray): `(n, m)` array specifing the voxel structure of the object. See `evogym.VOXEL_TYPES`.
            connections (Optional[np.ndarray]): `(2, k)` array specifying `k` pairwise voxel connections. Voxels are specified by their index into the 1D array `np.flatten(structure)`. The default behavior assumes all adjacent voxels are connected. (default = None)
        """
        out = WorldObject()
        out.load_from_array(name, structure, connections)
        return out

    def load_from_array(
        self, 
        name: str, 
        structure: np.ndarray, 
        connections: Optional[np.ndarray] = None) -> None:
        """
        Load an object from array.

        Args:
            name (str): object name.
            structure (np.ndarray): `(n, m)` array specifing the voxel structure of the object. See `evogym.VOXEL_TYPES`.
            connections (Optional[np.ndarray]): `(2, k)` array specifying `k` pairwise voxel connections. Voxels are specified by their index into the 1D array `np.flatten(structure)`. The default behavior assumes all adjacent voxels are connected. (default = None)
        """
        self.name = name

        # assume fully connected by default
        corrected_connections = None
        if connections is None:
            corrected_connections = get_full_connectivity(structure).T
        else:
            corrected_connections = connections.copy().T

        self.grid = np.flipud(structure)
        self.grid_size = Pair(self.grid.shape[1], self.grid.shape[0])
        self.pos = Pair(0, 0)

        # apply flip
        for i in range(len(corrected_connections)):
            for j in range(len(corrected_connections[i])):
                x, y = corrected_connections[i][
                    j] % self.grid_size.x, corrected_connections[i][
                        j] // self.grid_size.x
                y = (self.grid_size.y - 1) - y
                corrected_connections[i][j] = y * self.grid_size.x + x

        # set voxels
        self.voxels = []
        self.neighbors = {}
        idx_to_voxel = {}
        for y in range(self.grid_size.y):
            for x in range(self.grid_size.x):
                idx = y * self.grid_size.x + x
                if self.grid[y][x] != VOXEL_TYPES['EMPTY']:
                    self.voxels.append(Pair(x, y))
                    idx_to_voxel[idx] = Pair(x, y)

        # set connections
        for connection in corrected_connections:
            a, b = tuple(connection)
            if a not in idx_to_voxel or b not in idx_to_voxel:
                raise ValueError(
                    f'could not load object {self.name} from array -- connections matrix contains invalid elements'
                )
            if idx_to_voxel[a] not in self.neighbors:
                self.neighbors[idx_to_voxel[a]] = []
            if idx_to_voxel[b] not in self.neighbors:
                self.neighbors[idx_to_voxel[b]] = []

            ## POSSIBLE ERROR
            # self.neighbors[idx_to_voxel[b]].append(idx_to_voxel[b])
            self.neighbors[idx_to_voxel[a]].append(idx_to_voxel[b])
            self.neighbors[idx_to_voxel[b]].append(idx_to_voxel[a])

    def load_from_parsed_json(self, name: str, json_data: Any, grid_size: Pair) -> None:
        """
        Load object from parsed `json` data. It is recommended to use `WorldObject.load_from_json()` instead.

        Args:
            name (str): object name.
            json_data (Any): parsed json data.
            grid_size (Pair): grid size of world object is loaded from.
        """
        self.name = name

        # read in indices
        voxels = []
        index_to_voxel = {}
        num_voxels = len(json_data['indices'])
        for i in range(num_voxels):
            index_curr = json_data['indices'][i]
            voxels.append(
                Pair(index_curr % grid_size.x, index_curr // grid_size.x))
            index_to_voxel[index_curr] = voxels[-1].copy()

        if len(voxels) == 0:
            raise ValueError(f'object {self.name} has no voxels')

        #compute bounding box
        max_voxel = voxels[0].copy()
        min_voxel = voxels[0].copy()

        for voxel in voxels:
            max_voxel = max_voxel.each_max(voxel)
            min_voxel = min_voxel.each_min(voxel)

        # translate voxels according to bounding box
        self.pos = min_voxel.copy()
        self.grid_size = max_voxel - min_voxel + Pair(1, 1)

        self.voxels = []
        for voxel in voxels:
            self.voxels.append(voxel - self.pos)

        for index in index_to_voxel.keys():
            index_to_voxel[index] = index_to_voxel[index] - self.pos

        # set grid and neighbors
        grid: List[List[int]] = []
        for y in range(self.grid_size.y):
            grid.append([])
            for x in range(self.grid_size.x):
                grid[-1].append(0)

        self.neighbors = {}
        for voxel in self.voxels:
            self.neighbors[voxel] = []

        for i in range(num_voxels):
            index_curr = json_data['indices'][i]
            voxel_curr = index_to_voxel[index_curr]
            grid[voxel_curr.y][voxel_curr.x] = json_data['types'][i]
            for nei in json_data['neighbors'][f'{index_curr}']:
                if not nei in index_to_voxel:
                    raise ValueError(
                        f'object {self.name} has voxels with invalid neighbors'
                    )
                nei_voxel = index_to_voxel[nei]
                self.neighbors[voxel_curr].append(nei_voxel)

        self.grid = np.array(grid)

    def translate(self, dx: int, dy: int) -> None:
        """
        Translate an object. Objects retain their position when added to an instance of `EvoWorld`.

        Args:
            dx (int): change in x-position.
            dy (int): change in y-position.
        """
        new_pos = self.pos + Pair(dx, dy)
        if new_pos.x < 0 or new_pos.y < 0:
            raise ValueError(
                f'new pos for object {self.name}, {new_pos}, is invalid. Pos must be strictly non-negative'
            )
        self.pos = new_pos

    def set_pos(self, x: int, y: int) -> None:
        """
        Move an object. Objects retain their position when added to an instance of `EvoWorld`.

        Args:
            x (int): x-position of the bottom & leftmost voxel of the object. Starts at `0`.
            y (int): y-position of the bottom & leftmost voxel of the object. Starts at `0`.
        """
        new_pos = Pair(x, y)
        if new_pos.x < 0 or new_pos.y < 0:
            raise ValueError(
                f'new pos for object {self.name}, {new_pos}, is invalid. Pos must be strictly non-negative'
            )
        self.pos = new_pos

    def get_structure(self) -> np.ndarray:
        """
        Return an object's structure matrix.

        Return:
            np.ndarray: `(n, m)` array specifing the voxel structure of the object. See `evogym.VOXEL_TYPES`.
        """
        return np.flipud(self.grid)

    def get_connections(self) -> np.ndarray:
        """
        Return an object's connections matrix.

        Return:
            np.ndarray: `(2, k)` array specifying `k` pairwise voxel connections. Voxels are specified by their index into the 1D array `np.flatten(structure)`.
        """
        out = []
        for pos, neighs in self.neighbors.items():
            for neigh in neighs:
                out.append([
                    (self.grid_size.y-pos.y-1)*self.grid_size.x + pos.x,
                    (self.grid_size.y-neigh.y-1)*self.grid_size.x + neigh.x
                ])
        return np.array(out).T

    def get_pos(self) -> Tuple[int, int]:
        """
        Return an object's position.

        Return:
            Tuple[int, int]: position of the object `(x, y)`.
        """
        return (self.pos.x, self.pos.y)

    def rename(self, name: str) -> None:
        """
        Rename an object.

        Args:
            name (str): new name.
        """
        self.name = name

    def get_name(self) -> str:
        """
        Return an object's name.

        Return:
            str: name of object.
        """
        return self.name

    def copy(self,) -> WorldObject:
        """
        Returns a copy of object.

        Returns:
            WorldObject: copy of object.
        """
        return self.__copy__()

    def __copy__(self,) -> WorldObject:
        """
        Returns a copy of object.

        Returns:
            WorldObject: copy of object.
        """
        out = WorldObject()
        out.name = self.name
        out.pos = self.pos.copy()
        out.grid_size = self.grid_size.copy()

        out.voxels = []
        for voxel in self.voxels:
            out.voxels.append(voxel.copy())
        out.grid = self.grid.copy()

        out.neighbors = {}
        for voxel, neis in self.neighbors.items():
            out.neighbors[voxel.copy()] = []
            for nei in neis:
                out.neighbors[voxel.copy()].append(nei.copy())
        return out

    def __str__(self) -> str:
        """
        Returns nice string for debugging and printing.
        
        Returns:
            str: nice string for debugging and printing.
        """
        return f'Size {self.grid_size} object named {self.name} at {self.pos}.'

    def __repr__(self) -> str:
        """
        Returns nice string for debugging and printing.
        
        Returns:
            str: nice string for debugging and printing.
        """
        return f'WorldObject({self.__str__()})'