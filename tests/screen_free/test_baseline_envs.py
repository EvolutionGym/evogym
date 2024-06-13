import gymnasium as gym
import pytest
import warnings

import evogym.envs
from evogym import sample_robot

LITE_TEST_ENV_NAMES = [
    "Pusher-v0",
    "Walker-v0",
    "Traverser-v0",
]

def get_params():
    return [
        env_name if env_name not in LITE_TEST_ENV_NAMES 
        else pytest.param(env_name, marks=pytest.mark.lite) 
        for env_name in evogym.BASELINE_ENV_NAMES 
    ]

@pytest.mark.parametrize("env_name", get_params())
def test_env_creatable_and_has_correct_api(env_name):
    """
    - Env is creatable
    - Env steps for the correct number of steps
    - Env follows the gym API
    """
    
    body, _ = sample_robot((5, 5))
    env = gym.make(env_name, body=body)
    
    target_steps = env.spec.max_episode_steps
    assert isinstance(target_steps, int), f"Env {env_name} does not have a max_episode_steps attribute"
    
    # Reset
    obs, info = env.reset(seed=None, options=None)
    
    # Rollout with random actions
    n_steps = 0
    while True:
        action = env.action_space.sample() - 1
        ob, reward, terminated, truncated, info = env.step(action)
        
        n_steps += 1
        
        if terminated or truncated:
            env.reset(seed=None, options=None)
            break
        
        if n_steps > target_steps:
            break

    # Make sure we can still step after resetting
    env.step(env.action_space.sample() - 1)
    
    # Check that the env terminated after the correct number of steps
    assert n_steps <= target_steps, f"Env {env_name} terminated after {n_steps} steps, expected at most {target_steps}"
    if n_steps < target_steps:
        warnings.warn(f"Env {env_name} terminated early after {n_steps} steps, expected {target_steps}")
        
    env.close()
    