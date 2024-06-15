import os
from typing import List, Optional
import numpy as np
from ppo.eval import eval_policy
from stable_baselines3.common.callbacks import BaseCallback

class EvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(
        self,
        body: np.ndarray,
        env_name: str,
        eval_every: int,
        n_evals: int,
        n_envs: int,
        model_save_dir: str,
        model_save_name: str,
        connections: Optional[np.ndarray] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        
        self.body = body
        self.connections = connections
        self.env_name = env_name
        self.eval_every = eval_every
        self.n_evals = n_evals
        self.n_envs = n_envs
        self.model_save_dir = model_save_dir
        self.model_save_name = model_save_name
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            
        self.best_reward = -float('inf')
        
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        
        if self.num_timesteps % self.eval_every == 0:
            self._validate_and_save()
        return True
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self._validate_and_save()
    
    def _validate_and_save(self) -> None:
        rewards = eval_policy(
            model=self.model,
            body=self.body,
            connections=self.connections,
            env_name=self.env_name,
            n_evals=self.n_evals,
            n_envs=self.n_envs,
        )
        out = f"[{self.model_save_name}] Mean: {np.mean(rewards):.3}, Std: {np.std(rewards):.3}, Min: {np.min(rewards):.3}, Max: {np.max(rewards):.3}"
        mean_reward = np.mean(rewards).item()
        if mean_reward > self.best_reward:
            out += f" NEW BEST ({mean_reward:.3} > {self.best_reward:.3})"
            self.best_reward = mean_reward
            self.model.save(os.path.join(self.model_save_dir, self.model_save_name))
        if self.verbose > 0:
            print(out)
        