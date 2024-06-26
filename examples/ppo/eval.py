from typing import List, Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def eval_policy(
    model: PPO,
    body: np.ndarray,
    env_name: str,
    n_evals: int = 1,
    n_envs: int = 1,
    connections: Optional[np.ndarray] = None,
    render_mode: Optional[str] = None,
    deterministic_policy: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> List[float]:
    """
    Evaluate the controller for the robot in the environment.
    Returns the result of `n_evals` evaluations.
    """
    
    def run_evals(n: int) -> List[float]:
        """
        Run `n` evaluations in parallel.
        """
        
        # Parallel environments
        vec_env = make_vec_env(env_name, n_envs=n, seed=seed, env_kwargs={
            'body': body,
            'connections': connections,
            "render_mode": render_mode,
        })
        
        # Evaluate
        rewards = []
        obs = vec_env.reset()
        cum_done = np.array([False]*n)
        while not np.all(cum_done):
            action, _states = model.predict(obs, deterministic=deterministic_policy)
            obs, reward, done, info = vec_env.step(action)
            
            # Track when environments terminate
            if verbose:
                for i, (d, cd) in enumerate(zip(done, cum_done)):
                    if d and not cd:
                        print(f"Environment {i} terminated after {len(rewards)} steps")
            
            # Keep track of done environments
            cum_done = np.logical_or(cum_done, done)
            
            # Update rewards -- done environments will not be updated
            reward[cum_done] = 0
            rewards.append(reward)    
        vec_env.close()
        
        # Sum rewards over time
        rewards = np.asarray(rewards)
        return np.sum(rewards, axis=0)
    
    # Run evaluations n_envs at a time
    rewards = []
    for i in range(np.ceil(n_evals/n_envs).astype(int)):
        rewards.extend(run_evals(min(n_envs, n_evals - i*n_envs)))
    
    return rewards
