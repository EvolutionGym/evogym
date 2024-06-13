import gymnasium as gym
import pytest
import warnings
import numpy as np
from itertools import product

import evogym.envs
from evogym import sample_robot

@pytest.mark.parametrize(
    "render_mode, add_options",
    list(product(
        ["human", "screen"],
        [True, False],
    ))
)
def test_render_modes(render_mode, add_options):
    """
    - Env can render to screen
    """
    
    body, _ = sample_robot((5, 5))
    if add_options:
        env = gym.make("Walker-v0", body=body, render_mode=render_mode, render_options={
            "verbose": False,
            "hide_background": False,
            "hide_grid": False,
            "hide_edges": False,
            "hide_voxels": False
        })
    else:
        env = gym.make("Walker-v0", body=body, render_mode=render_mode)

    # Reset
    obs, info = env.reset(seed=None, options=None)
    
    for i in range(10):
        
        # Step -- we don't need to render explicitly
        action = env.action_space.sample() - 1
        ob, reward, terminated, truncated, info = env.step(action)
        
    env.close()
    
@pytest.mark.parametrize("env_name", evogym.BASELINE_ENV_NAMES)
def test_all_env_render(env_name):
    """
    - Env can render to screen
    """
    
    body, _ = sample_robot((5, 5))
    env = gym.make(env_name, body=body, render_mode="human")

    # Reset
    obs, info = env.reset(seed=None, options=None)
    
    for i in range(10):
        
        # Step -- we don't need to render explicitly
        action = env.action_space.sample() - 1
        ob, reward, terminated, truncated, info = env.step(action)
        
    env.close()
    