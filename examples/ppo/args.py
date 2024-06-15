import argparse

def add_ppo_args(parser: argparse.ArgumentParser) -> None:
    """
    Add PPO arguments to the parser
    """
    
    ppo_parser: argparse.ArgumentParser = parser.add_argument_group('ppo arguments')
    
    ppo_parser.add_argument(
        '--verbose-ppo', default=1, type=int, help='Verbosity level for PPO: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages (default: 1)'
    )
    ppo_parser.add_argument(
        '--learning-rate', default=2.5e-4, type=float, help='Learning rate for PPO (default: 2.5e-4)'
    )
    ppo_parser.add_argument(
        '--n-steps', default=128, type=int, help='The number of steps to run for each environment per update for PPO (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) (default: 128)'
    )
    ppo_parser.add_argument(
        '--batch-size', default=4, type=int, help='Mini-batch size for PPO (default: 4)'
    )
    ppo_parser.add_argument(
        '--n-epochs', default=4, type=int, help='Number of epochs when optimizing the surrogate objective for PPO (default: 4)'
    )
    ppo_parser.add_argument(
        '--gamma', default=0.99, type=float, help='Discount factor for PPO (default: 0.99)'
    )
    ppo_parser.add_argument(
        '--gae-lambda', default=0.95, type=float, help='Lambda parameter for Generalized Advantage Estimation for PPO (default: 0.95)'
    )
    ppo_parser.add_argument(
        '--vf-coef', default=0.5, type=float, help='Value function coefficient for PPO loss calculation (default: 0.5)'
    )
    ppo_parser.add_argument(
        '--max-grad-norm', default=0.5, type=float, help='The maximum value of the gradient clipping for PPO (default: 0.5)'
    )
    ppo_parser.add_argument(
        '--ent-coef', default=0.01, type=float, help='Entropy coefficient for PPO loss calculation (default: 0.01)'
    )
    ppo_parser.add_argument(
        '--clip-range', default=0.1, type=float, help='Clipping parameter for PPO (default: 0.1)'
    )
    ppo_parser.add_argument(
        '--total-timesteps', default=1e6, type=int, help='Total number of timesteps for PPO (default: 1e6)'
    )
    ppo_parser.add_argument(
        '--log-interval', default=50, type=int, help='Episodes before logging PPO (default: 50)'
    )
    ppo_parser.add_argument(
        '--n-envs', default=1, type=int, help='Number of parallel environments for PPO (default: 1)'
    )
    ppo_parser.add_argument(
        '--n-eval-envs', default=1, type=int, help='Number of parallel environments for PPO evaluation (default: 1)'
    )
    ppo_parser.add_argument(
        '--n-evals', default=1, type=int, help='Number of times to run the environment during each eval (default: 1)'
    )
    ppo_parser.add_argument(
        '--eval-interval', default=1e5, type=int, help='Number of steps before evaluating PPO model (default: 1e5)'
    )