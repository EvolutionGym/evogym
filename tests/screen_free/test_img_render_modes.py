import gymnasium as gym
import pytest
import warnings
import numpy as np
from itertools import product

import evogym.envs
from evogym import sample_robot

# @pytest.mark.parametrize(
#     "env_name, render_mode",
#     list(product(
#         evogym.BASELINE_ENV_NAMES,
#         [None, "img", "rgb_array"],
#     ))
# )
@pytest.mark.parametrize(
    "env_name, render_mode",
    list(product(
        ["ObstacleTraverser-v1", "Traverser-v0"],
        [None, "img", "rgb_array"],
    ))
)
def test_render(env_name, render_mode):
    """
    - Env can render to none and to image
    """
    
    body, _ = sample_robot((5, 5))
    env = gym.make(env_name, body=body, render_mode=render_mode)
    
    # Reset
    obs, info = env.reset(seed=None, options=None)
    
    for i in range(10):
    
        # Render
        result = env.render()
        
        if render_mode is None:
            # Result should be None
            assert result is None, f"Env returned {type(result)} instead of None"
        else:
            # Check img
            assert isinstance(result, np.ndarray), f"Env returned {type(result)} instead of np.ndarray"
            x, y, c = result.shape
            assert c == 3, f"Env returned image with {c} channels, expected 3"
        
        # Step
        action = env.action_space.sample() - 1
        ob, reward, terminated, truncated, info = env.step(action)
    
    env.close()