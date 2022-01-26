import random
import numpy as np
import time
from typing import Tuple, Optional

### CONSTANTS ###

VOXEL_TYPES = {
    'EMPTY': 0,
    'RIGID': 1,
    'SOFT': 2,
    'H_ACT': 3,
    'V_ACT': 4,
    'FIXED': 5,
}

BASELINE_ENV_NAMES = [
    'Walker-v0',
    'BridgeWalker-v0',
    'BidirectionalWalker-v0',
    'Carrier-v0',
    'Carrier-v1',
    'Pusher-v0',
    'Pusher-v1',
    'Thrower-v0',
    'Catcher-v0',
    'BeamToppler-v0',
    'BeamSlider-v0',
    'Lifter-v0',
    'Climber-v0',
    'Climber-v1',
    'Climber-v2',
    'UpStepper-v0',
    'DownStepper-v0',
    'ObstacleTraverser-v0',
    'ObstacleTraverser-v1',
    'Hurdler-v0',
    'PlatformJumper-v0',
    'GapJumper-v0',
    'Traverser-v0',
    'CaveCrawler-v0',
    'AreaMaximizer-v0',
    'AreaMinimizer-v0',
    'WingspanMazimizer-v0',
    'HeightMaximizer-v0',
    'Flipper-v0',
    'Jumper-v0',
    'Balancer-v0',
    'Balancer-v1'
    ]


### PROBABILITY AND RANDOM ROBOT SAMPLING ###

def get_uniform(x: int) -> np.ndarray:
    """
    Return a uniform distribution of a given size.

    Args:
        x (int): size of distribution.
    
    Returns:
        np.ndarray: array representing the probability distribution.
    """
    return np.ones((x)) / x

def draw(pd: np.ndarray) -> int:
    """
    Sample from a probability distribution.

    Args:
        pd (np.ndarray): array representing the probability of sampling each element.
    
    Returns:
        int: sampled index.
    """
    pd_copy = pd.copy()
    if (type(pd_copy) != type(np.array([]))):
        pd_copy = np.array(pd_copy)
    pd_copy = pd_copy / pd_copy.sum()

    rand = random.uniform(0, 1)
    sum = 0
    for i in range(pd_copy.size):
        sum += pd_copy[i]
        if rand <= sum:
            return i

def sample_robot(
    robot_shape: Tuple[int, int], 
    pd: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a randomly sampled robot of a particular size.

    Args:
        robot_shape (Tuple(int, int)): robot shape to sample `(h, w)`.
        pd (np.ndarray): `(5,)` array representing the probability of sampling each robot voxel (empty, rigid, soft, h_act, v_act). Defaults to a custom distribution. (default = None)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: randomly sampled (valid) robot voxel array and its associated connections array.
    """
    done = False

    while (not done):

        if pd is None:
            pd = get_uniform(5)
            pd[0] = 0.6

        robot = np.zeros(robot_shape)
        for i in range(robot.shape[0]):
            for j in range(robot.shape[1]):
                robot[i][j] = draw(pd)

        if is_connected(robot) and has_actuator(robot):
            done = True

    return robot, get_full_connectivity(robot)

# def prob(x):
#     return x <= random.uniform(0, 1)

### ROBOT PROPERTY CHECKING AND COMPUTATIONS ###

def _is_in_bounds(x: int, y: int, width: int, height: int) -> bool:
    """
    Returns whether or not a certain index is within bounds.

    Args:
        x (int): x pos.
        y (int): y pos.
        width (int): max x.
        height (int): max y.
    """
    if x < 0:
        return False
    if y < 0:
        return False
    if x >= width:
        return False
    if y >= height:
        return False
    return True

def _recursive_search(x: int, y: int, connectivity: np.ndarray, robot: np.ndarray) -> None:
    """
    Performs a floodfill search.
    
    Args:
        x (int): x pos.
        y (int): y pos.
        connectivity (np.ndarray): array to be filled in during floodfill.
        robot (np.ndarray): array specifing the voxel structure of the robot.
    """
    if robot[x][y] == 0:
        return
    if connectivity[x][y] != 0:
        return

    connectivity[x][y] = 1

    for x_offset in [-1, 1]:
        if _is_in_bounds(x + x_offset, y, robot.shape[0], robot.shape[1]):
            _recursive_search(x + x_offset, y, connectivity, robot)

    for y_offset in [-1, 1]:
        if _is_in_bounds(x, y + y_offset, robot.shape[0], robot.shape[1]):
            _recursive_search(x, y + y_offset, connectivity, robot)

def is_connected(robot: np.ndarray) -> bool:
    """
    Returns whether or not a certain robot is connected by running floodfill.

    Args:
        robot (np.ndarray): array specifing the voxel structure of the robot.
    
    Returns:
        bool: whether or not the robot is connected.
    """
    is_found = np.zeros(robot.shape)

    start = None
    for i in range(robot.shape[0]):
        if start:
            break
        for j in range(robot.shape[1]):
            if robot[i][j] != 0:
                start = (i, j)
                break

    if start == None:
        return False

    connectivity = np.zeros(robot.shape)
    _recursive_search(start[0], start[1], connectivity, robot)

    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] != 0 and connectivity[i][j] != 1:
                return False

    return True

def has_actuator(robot: np.ndarray) -> bool:
    """
    Returns whether or not a certain robot has an actuator.
    Args:
        robot (np.ndarray): array specifing the voxel structure of the robot.
    
    Returns:
        bool: whether or not the robot has an actuator.
    """
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] == 3 or robot[i][j] == 4:
                return True

    return False

def get_full_connectivity(robot: np.ndarray) -> np.ndarray:
    """
    Returns a connections array given a connected robot structure. Assumes all adjacent voxels are connected.

    Args:
        robot (np.ndarray): array specifing the voxel structure of the robot.
    
    Returns:
        np.ndarray: `(2, k)` array specifying `k` pairwise voxel connections. Voxels are specified by their index into the 1D array `np.flatten(robot)`.
    """
    out = []

    for i in range(robot.size):
        x = i % robot.shape[1]
        y = i // robot.shape[1]

        if robot[y][x] == 0:
            continue

        nx = x + 1
        ny = y

        if _is_in_bounds(nx, ny, robot.shape[1],
                        robot.shape[0]) and robot[ny][nx] != 0:
            out.append([x + robot.shape[1] * y, nx + robot.shape[1] * ny])

        nx = x
        ny = y + 1

        if _is_in_bounds(nx, ny, robot.shape[1],
                        robot.shape[0]) and robot[ny][nx] != 0:
            out.append([x + robot.shape[1] * y, nx + robot.shape[1] * ny])

    if len(out) == 0:
        return np.empty((0, 2)).T
    return np.array(out).T

### HELPER FUNCTIONS AND CLASSES ###

def hashable(robot: np.ndarray) -> str:
    """
    Returns a hashable representation of a robot.

    Args:
        robot (np.ndarray): array specifing the voxel structure of the robot.
    
    Returns:
        str: string representation of the robot.
    """
    out = ""
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            out += str(int(robot[i][j]))
    return out

class Timer():
    """
    Helpful timer class to set a target step/render frequency. Used for visualizations.

    Args:
        target_step_frequency (int): target frequency (in steps per second). If `None`, always steps. (default = None)
    """
    def __init__(self, target_step_frequency: Optional[int] = None) -> None:
        self._target_rps = target_step_frequency if target_step_frequency is not None else float('inf')
        self._total_steps = 0
        self.total_count = 0
        self._steps = 0
        self._old_time = self._current_time()

    def _current_time(self) -> int:
        """
        Get the current time in ms.

        Returns:
            int: current time.
        """
        return int(round(time.time() * 1000))

    def should_step(self) -> bool:
        """
        Returns whether or not to step at the current time to maintin the target step frequency.

        Returns:
            bool: whether or not to step (True = should step).
        """
        if self._steps >= (self._target_rps-1)*(self._current_time() - self._old_time)/1000:
            return False
        return True

    def step(self, verbose=False):
        """
        Step the timer.

        Args:
            verbose (bool): whether or not to print out the current step frequency and the average step frequency since starting the timer. (default = False)
        """
        self._steps += 1
        if self._current_time() - self._old_time > 1000:
            self._total_steps += self._steps
            self.total_count += 1
            self._old_time += 1000
            if verbose:
                print(f'rps: {self._steps} | avg rps: {round(self._total_steps/self.total_count*100.0)/100.0}')
            self._steps = 0

class Pair():
    """
    Helpful tuple of two ints that supports many useful operations such as addition, subtraction, multiplication, and division.
    """
    def __init__(self, x, y):
        if not isinstance(x, int):
            raise TypeError(f'{x} is not instance of int')
        if not isinstance(y, int):
            raise TypeError(f'{y} is not instance of int')
        self.x = x
        self.y = y

    def __getitem__(self, key):
        if key != 0 and key != 1:
            raise IndexError('Pair can only be indexed with values 0 or 1')
        if key == 0:
            return self.x
        if key == 1:
            return self.y

    def __add__(self, b):
        if isinstance(b, int):
            return Pair(self.x + b, self.y + b)
        if isinstance(b, Pair):
            return Pair(self.x + b.x, self.y + b.y)
        raise TypeError(f'cannot add Pair and {type(b)}')

    def __sub__(self, b):
        if isinstance(b, int):
            return Pair(self.x - b, self.y - b)
        if isinstance(b, Pair):
            return Pair(self.x - b.x, self.y - b.y)
        raise TypeError(f'cannot subtract Pair and {type(b)}')

    def __mul__(self, b):
        if isinstance(b, int):
            return Pair(self.x * b, self.y * b)
        if isinstance(b, Pair):
            return Pair(self.x * b.x, self.y * b.y)
        raise TypeError(f'cannot multiply Pair and {type(b)}')

    def __truediv__(self, b):
        if isinstance(b, int):
            return Pair(self.x // b, self.y // b)
        if isinstance(b, Pair):
            return Pair(self.x // b.x, self.y // b.y)
        raise TypeError(f'cannot divide Pair and {type(b)}')

    def __floordiv__(self, b):
        if isinstance(b, int):
            return Pair(self.x // b, self.y // b)
        if isinstance(b, Pair):
            return Pair(self.x // b.x, self.y // b.y)
        raise TypeError(f'cannot divide Pair and {type(b)}')

    def __mod__(self, b):
        if isinstance(b, int):
            return Pair(self.x % b, self.y % b)
        if isinstance(b, Pair):
            return Pair(self.x % b.x, self.y % b.y)
        raise TypeError(f'cannot compute mod of Pair and {type(b)}')

    def __eq__(self, b):
        if isinstance(b, int):
            return self.x == b and self.y == b
        if isinstance(b, Pair):
            return self.x == b.x and self.y == b.y
        return False

    def abs(self,):
        return Pair(abs(self.x), abs(self.y))

    def each_max(self, b):
        if isinstance(b, Pair):
            return Pair(max(self.x, b.x), max(self.y, b.y))
        raise TypeError(
            f'cannot compute element-wise max of Pair and {type(b)}')

    def each_min(self, b):
        if isinstance(b, Pair):
            return Pair(min(self.x, b.x), min(self.y, b.y))
        raise TypeError(
            f'cannot compute element-wise min of Pair and {type(b)}')

    def __copy__(self,):
        return Pair(self.x, self.y)

    def copy(self,):
        return self.__copy__()

    def __str__(self,):
        return f'({self.x}, {self.y})'

    def __repr__(self,):
        return f'Pair{self.__str__()}'

    def __hash__(self,):
        return hash(self.__repr__())