import random
import numpy as np

from cppn_neat.run import run_cppn_neat

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    best_robot, best_fitness = run_cppn_neat(
        experiment_name='test_cppn',
        structure_shape=(5, 5),
        pop_size=3,
        max_evaluations=6,
        train_iters=50,
        num_cores=3,
    )
    
    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)