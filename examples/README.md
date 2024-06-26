# Examples

This readme describes how to run several control optimization and co-design experiments and visualize the results. All scripts should be run from within the `examples` directory. Ensure that you have installed requirements: `pip install -r requirements.txt` and cloned the repo with submodules before running any of these scripts.

> [!WARNING]
> Many of these scripts have been modified in favor of improved usability. If you wish to exactly recreate the results from the original EvoGym paper, please see the [original release](https://github.com/EvolutionGym/evogym/releases/tag/1.0.0).

## PPO (control optimization)

The script is set up to train a robot's policy in the `Walker-v0` environment:

```shell
python run_ppo.py --n-envs 4 --n-eval-envs 4 --n-evals 4 --eval-interval 10000 --total-timesteps 100000
```

Results saved to `saved_data/test_ppo`. See `python run_ppo.py --help` to see how training can be customized. We recommend setting `total-timesteps > 1e6` for a serious training run.

## Co-Design

All three co-design algorithms: the genetic algorithm, bayesian optimization, and CPPN-NEAT have the same core parameters:

* `exp-name` = all experiment files are saved to `saved_data/exp_name`
* `env-name` = environment on which to run co-design
* `pop-size` = the algorithm evolves robots in populations of this size
* `structure-shape` = each robot is represented by `(m,n)` matrix of voxels
* `max-evaluations` = maximum number of unique robots to evaluate
* `num-cores` = number of robots to train in parallel. Note: the total number of processes created will be `num-cores * n-envs` (as specified below in the command line)

See all options via 
`python run_ga.py --help`
`python run_bo.py --help`
`python run_cppn_neat.py --help`

From within `example`, you can run the co-design algorithms as follows:

```shell
python run_ga.py --eval-interval 10000 --total-timesteps 100000
python run_bo.py --eval-interval 10000 --total-timesteps 100000
python run_cppn_neat.py --eval-interval 10000 --total-timesteps 100000
```

Note that the default parameters are set for testing purposes, and will not produce task-performant robots. Feel free to increase the co-design/PPO parameters based on your compute availability. You may also reference evaluation parameters from [Appendix D. of our paper](https://arxiv.org/pdf/2201.09863).

## Visualize

From within `example`, you can visualize the results from any co-design experiment with the following command:

```shell
python visualize.py --env-name "Walker-v0"
```

Use the appropriate environment name and follow the on-screen instructions.

## Make Gifs

This script generates gifs from co-design experiments. To set the parameters of the gif generating script, you can edit `make_gifs.py` and change the following:

* `GIF_RESOLUTION` = resolution of produced gifs
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

