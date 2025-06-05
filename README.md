# Evolution Gym

[![Build](https://github.com/EvolutionGym/evogym/actions/workflows/wheels.yml/badge.svg?branch=main)](https://github.com/EvolutionGym/evogym/actions/workflows/wheels.yml)
[![Test](https://github.com/EvolutionGym/evogym/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/EvolutionGym/evogym/actions/workflows/test.yml)

Evolution Gym is a large-scale benchmark for co-optimizing the design and control of soft robots. It provides a lightweight soft-body simulator wrapped with a gym-like interface for developing learning algorithms. EvoGym also includes a suite of 32 locomotion and manipulation tasks, detailed on our [website](https://evolutiongym.github.io/all-tasks). Task suite evaluations are described in our [NeurIPS 2021 paper](https://arxiv.org/pdf/2201.09863).


<br>
<p align="center">
  <a href="https://forms.gle/Rn1TwzYGuVSAPQKfA" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Get%20feature%20notifications-orange?style=for-the-badge&logo=tacobell&logoColor=black&color=fda158" alt="Get notified on releases">
  </a>
  <!-- &nbsp;&nbsp; -->
  <a href="https://forms.gle/vH5Ta7HtVVQb6GpR9" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Submit%20a%20feature%20request-blue?style=for-the-badge&logo=googleforms&logoColor=black&color=a3d7ff" alt="Submit a feature request">
  </a>
</p>



<!-- <p align="center">
  <a href="https://forms.gle/Rn1TwzYGuVSAPQKfA" target="_blank">
    <img src="https://img.shields.io/badge/🔔%20Get%20feature%20notifications-orange?style=for-the-badge&logoColor=black&color=ff8119" alt="Get notified on releases">
  </a>
  <a href="https://forms.gle/vH5Ta7HtVVQb6GpR9" target="_blank">
    <img src="https://img.shields.io/badge/Submit%20a%20feature%20request-blue?style=for-the-badge&logo=googleforms&logoColor=white&color=199cff" alt="Submit a feature request">
  </a>
</p> -->



> [!NOTE]
> **[06/25]** 90k+ robot structures and 2.5k+ robot policies from the original EvoGym paper are now [available for download with instructions](https://github.com/EvolutionGym/evogym-datasets)!

> [!NOTE]
> **[07/24]** EvoGym has been recently updated! TLDR: requirements have been modernized (gym/gymnasium, numpy, etc.), and the library is now pip-installable.

[//]: # (<img src="https://github.com/EvolutionGym/evogym/raw/main/images/teaser-low-res.gif" alt="teaser" width="800"/>)
![teaser](https://github.com/EvolutionGym/evogym/raw/main/images/teaser-low-res.gif)

# Installation

EvoGym supports python `3.7` to `3.10` on most operating systems:

```shell
pip install evogym --upgrade
```

<!-- > [!CAUTION]
> This doesn't work yet -- coming soon! For now, you can install from test pypi:
> ```shell
> pip install "numpy<2.0.0" gymnasium
> pip install -i https://test.pypi.org/simple/ evogym
> ``` -->

On **Linux** install the following packages (or equivalent):

```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
```

## From Source

If your platform is not supported, you may alternatively build from source:

### Requirements

* Python 3
* Linux, macOS, or Windows with [Visual Studios 2017](https://visualstudio.microsoft.com/vs/older-downloads/) build tools.
* [CMake](https://cmake.org/download/)

Clone the repo and submodules:

```shell
git clone --recurse-submodules https://github.com/EvolutionGym/evogym.git
```

On **Linux only**:

```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
```

Finally, to install `evogym`, run the following in the environment of your choice:

```shell
pip install -e .
```

## Test Installation

If you have the repo cloned, `cd` to the `examples` folder and run the following script:

```shell
python gym_test.py
```

Alternatively, you can run the following snippet:

```python
import gymnasium as gym
import evogym.envs
from evogym import sample_robot


if __name__ == '__main__':

    body, connections = sample_robot((5,5))
    env = gym.make('Walker-v0', body=body, render_mode='human')
    env.reset()

    while True:
        action = env.action_space.sample()
        ob, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    env.close()
```

This script creates a random `5x5` robot in the `Walking-v0` environment. The robot is taking random actions. A window should open with a visualization of the environment -- kill the process from the terminal to close it.

## Known Issues

### Linux and Conda

Error message: `libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so`

Fix: `conda install -c conda-forge libstdcxx-ng`

# Usage

In addition to the resources below, you can find API documentation on our [website](https://evolutiongym.github.io/documentation).

## Tutorials

You can find tutorials for getting started with the codebase on our [website](https://evolutiongym.github.io/tutorials). Completed code from all tutorials is also available in the `tutorials` folder, along with a `README`. Tutorials are included for:
- Using the [evogym API](https://evolutiongym.github.io/tutorials/basic-api.html)
- Making a [custom evogym environment](https://evolutiongym.github.io/tutorials/new-env.html)
- Supported [rendering options](https://github.com/EvolutionGym/evogym/blob/main/tutorials/rendering_options.py)

## Examples

To run co-design and control optimization experiments in EvoGym, please see the `examples` folder and its `README`. Included are scripts for:
- Running PPO
- Running a Genetic Algorithm
- Running Bayesian Optimization
- Running CPPN-NEAT
- Visualizing results
- Saving results as gifs

Make sure you clone the repo with submodules:

```shell
git clone --recurse-submodules https://github.com/EvolutionGym/evogym.git
```

Install the necessary python requirements:
```shell
pip install -r requirements.txt
```

## Design Tool

The Design Tool provides a gui for creating Evolution Gym environments. Please see [this repo](https://github.com/EvolutionGym/evogym-design-tool).

[//]: # (<img src="https://github.com/EvolutionGym/evogym/raw/main/images/design-tool.gif" alt="design-tool" width="800"/>)
![design-tool](https://github.com/EvolutionGym/evogym/raw/main/images/design-tool.gif)

## Headless Mode

EvoGym runs in headless mode by default, and avoids initializing rendering libraries until necessary. If using a server without rendering capabilities, ensure that:

```python
# Envs are created with render_mode=None (None by default)
env = gym.make('Walker-v0', body=body, render_mode=None)
```

```python
# If using the low-level api, do not call EvoViewer.render()
world = EvoWorld.from_json(os.path.join('world_data', 'simple_environment.json'))
sim = EvoSim(world)
viewer = EvoViewer(sim)
viewer.render('img') # <-- Rendering libraries are initialized; do not call this
```

# Datasets
We've released two datasets of robot structures and policies from the original EvoGym paper. Instructions for downloading and using these datasets are available in the [evogym-datasets](https://github.com/EvolutionGym/evogym-datasets) repo. All datasets are hosted on [huggingface](https://huggingface.co/EvoGym).

- [EvoGym/robots](https://huggingface.co/datasets/EvoGym/robots): 90k+ annotated robot structures
- [EvoGym/robot-with-policies](https://huggingface.co/datasets/EvoGym/robots-with-policies): 2.5k+ annotated robot structures and policies

# Dev

Install the repo with submodules:

```shell
git clone --recurse-submodules https://github.com/EvolutionGym/evogym.git
```

Install the necessary python requirements. You will additionally need to install the dev requirements:
```shell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Run Tests

From within the `tests` directory run the full test suite:

```shell
cd tests
pytest -s -v -n auto
```

Or the lite test suite:


```shell
cd tests
pytest -s -v -n auto -m lite
```

# Citation

If you find our repository helpful to your research, please cite our paper:

```
@article{bhatia2021evolution,
  title={Evolution gym: A large-scale benchmark for evolving soft robots},
  author={Bhatia, Jagdeep and Jackson, Holly and Tian, Yunsheng and Xu, Jie and Matusik, Wojciech},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
