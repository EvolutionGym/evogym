import random
import numpy as np
import argparse

from cppn_neat.run import run_cppn_neat
from ppo.args import add_ppo_args

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    parser = argparse.ArgumentParser(description='Arguments for ga script')
    parser.add_argument('--exp_name', type=str, default='test_cppn', help='Name of the experiment (default: test_cppn)')
    parser.add_argument('--env_name', type=str, default='Walker-v0', help='Name of the environment (default: Walker-v0)')
    parser.add_argument('--pop_size', type=int, default=3, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5,5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max_evaluations', type=int, default=6, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num_cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    add_ppo_args(parser)
    args = parser.parse_args()

    best_robot, best_fitness = run_cppn_neat(args)
    
    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)