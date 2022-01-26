import random
import numpy as np

from bo.run import run_bo

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # best_robot, best_fitness = run_bo(
    #     experiment_name='test_bo',
    #     structure_shape=(5, 5),
    #     pop_size=50,
    #     max_evaluations=500,
    #     train_iters=1000,
    #     num_cores=25,
    # )

    best_robot, best_fitness = run_bo(
        experiment_name='test_bo',
        structure_shape=(5, 5),
        pop_size=3,
        max_evaluations=6,
        train_iters=50,
        num_cores=3,
    )

    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)