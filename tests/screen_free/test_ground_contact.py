import gymnasium as gym
import numpy as np
import pytest

import evogym.envs
from evogym import VOXEL_TYPES

# (action, number of steps): a horizontal muscle on a rigid voxel, squeezed so that its bottom
# corners slip into the neighboring ground voxels while its sides cross the top of the one between
ACTIONS = [
    (0.7800052091400137, 2),
    (1.160452179980335, 14),
    (1.2479329293224728, 30),
    (0.942821476132681, 14),
]

@pytest.mark.lite
def test_robot_does_not_sink_into_ground():
    """
    - A robot point inside a ground voxel is pushed out, even when neither of its own edges crosses that voxel's top
    """

    body = np.zeros((5, 5))
    body[1, 2] = VOXEL_TYPES['H_ACT']
    body[2, 2] = VOXEL_TYPES['RIGID']
    env = gym.make("Jumper-v0", body=body)
    env.reset(seed=None, options=None)
    sim = env.unwrapped

    robot_pos = sim.object_pos_at_time(sim.get_time(), "robot")
    ground_pos = sim.object_pos_at_time(sim.get_time(), "ground")
    under_robot = (ground_pos[0] >= robot_pos[0].min()) & (ground_pos[0] <= robot_pos[0].max())
    ground_top = ground_pos[1][under_robot].max()

    max_depth = 0.0
    for action, n_steps in ACTIONS:
        for _ in range(n_steps):
            env.step(np.array([action]))
            robot_pos = sim.object_pos_at_time(sim.get_time(), "robot")
            max_depth = max(max_depth, ground_top - robot_pos[1].min())

    env.close()

    # a voxel is 0.1 deep; ordinary contact lets a point in by a few millimeters
    assert max_depth < 0.05, f"Robot sank {max_depth:.4f} into the ground, expected less than 0.05"
