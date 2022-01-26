# Examples

This readme describes how to run several control optimization and co-design experiments and visualize the results.

## PPO (control optimization)

To set the parameters of group ppo, you can edit `run_group_ppo.py` and change the following:

* `experiment_name` = all experiment files are saved to `saved_data/experiment_name`

Create `SimJob`s to specify which robots to train and in which environments to train them. `SimJob`s are parameterized by the following:

* `name` = all job files are saved to `saved_data/experiment_name/name`
* `robots` = array of robot names specifying which robots to train. Robot files must be of type `.json` (created using the EvoGym Design Tool) or `.npz` (saved from another experiment) and must be located in `examples/world_data` 
* `envs` = array of environment names in which to train robots
* `train_iters` = number of iterations of ppo to train each robot's controller

Each robot in `robots` will be trained in each environment in `envs` for `train_iters` iterations of ppo.

From within `example`, you can run group ppo with the following command:

```shell
python run_group_ppo.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50
```

All ppo hyperparameters are specified through command line arguments. For more details please see [this repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## Genetic Algorithm (co-design)

To set the parameters of the genetic algorithm, you can edit `run_ga.py` and change the following:

* `seed` = seed to control randomness
* `pop_size` = the algorithm evolves robots in populations of this size
* `structure_shape` = each robot is represented by `(m,n)` matrix of voxels 
* `experiment_name` = all experiment files are saved to `saved_data/experiment_name`
* `max_evaluations` = maximum number of unique robots to evaluate
* `train_iters` = number of iterations of ppo to train each robot's controller
* `num_cores` = number of robots to train in parallel. Note: the total number of processes created will be `num_cores * num_processes` (as specified below in the command line)

From within `example`, you can run the genetic algorithm with the following command:

```shell
python run_ga.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50
```

The environment name as well as all ppo hyperparameters are specified through command line arguments. For more details please see [this repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## Bayesian Optimization (co-design)

To set the parameters of bayesian optimization, you can edit `run_bo.py` and change the following:

* `seed` = seed to control randomness
* `pop_size` = the algorithm evolves robots in populations of this size
* `structure_shape` = each robot is represented by `(m,n)` matrix of voxels 
* `experiment_name` = all experiment files are saved to `saved_data/experiment_name`
* `max_evaluations` = maximum number of unique robots to evaluate. Should be a multiple of `pop_size`
* `train_iters` = number of iterations of ppo to train each robot's controller
* `num_cores` = number of robots to train in parallel. Note: the total number of processes created will be `num_cores * num_processes` (as specified below in the command line)

From within `example`, you can run bayesian optimization with the following command:

```shell
python run_bo.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50
```

The environment name as well as all ppo hyperparameters are specified through command line arguments. For more details please see [this repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## CPPN-NEAT (co-design)

To set the parameters of cppn-neat, you can edit `run_cppn_neat.py` and change the following:

* `seed` = seed to control randomness
* `pop_size` = the algorithm evolves robots in populations of this size
* `structure_shape` = each robot is represented by `(m,n)` matrix of voxels 
* `experiment_name` = all experiment files are saved to `saved_data/experiment_name`
* `max_evaluations` = maximum number of unique robots to evaluate. Should be a multiple of `pop_size`
* `train_iters` = number of iterations of ppo to train each robot's controller
* `num_cores` = number of robots to train in parallel. Note: the total number of processes created will be `num_cores * num_processes` (as specified below in the command line)

From within `example`, you can run cppn-neat with the following command:

```shell
python run_cppn_neat.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50
```

The environment name as well as all ppo hyperparameters are specified through command line arguments. For more details please see [this repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## Visualize

From within `example`, you can visualize the results from any co-design experiment with the following command:

```shell
python visualize.py --env-name "Walker-v0"
```

Use the appropriate environment name and follow the on-screen instructions.

To visualize the results of a group ppo experiment you can use the same command -- the environment name is no longer necessary.

## Make Gifs

To set the parameters of the gif generating script, you can edit `make_gifs.py` and change the following:

* `GIF_RESOLUTION` = resolution of produced gifs
* `NUM_PROC` = number of gifs to produce in parallel
* `name` = all files are saved to `saved_data/all_media/name`
* `experiment_names`, `env_names` = arrays of experiments their corresponding environments to generate gifs for
* `load_dir` = directory where experiments are stored
* `generations` = array of generation numbers to use in the gif generation process. The default behavior is to use all of them
* `ranks` = which robots to use in the gif generation process. Robots are given a rank from `1` to `n` based on their reward during training. The default behavior is to use all robots.
* `organize_by_experiment` = flag specifying whether or not to organize gifs into a separate folders for each experiment if multiple experiments are specified in `experiment_names`
* `organize_by_generation` = flag specifying whether or not to organize gifs into separate folders for each generation

The script runs without any command line arguments.

### Example

```python
my_job = Job(
    name = 'ga_walking_experiment_gifs',
    experiment_names= ['ga_walking_experiment'],
    env_names = ['Walker-v0'],
    load_dir = exp_root,
    ranks = [i for i in range(5)]
    organize_by_generation=True,
)
```

In this example, gifs are generated for the experiment `ga_walking_experiment` which contains robots trained in the `Walker-v0` environment. Gifs for the top `5` robots in each generation `i` will be generated and saved to `saved_data/all_media/ga_walking_experiment_gifs/generation_i`.

