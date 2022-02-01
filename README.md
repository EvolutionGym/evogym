# Evolution Gym

A large-scale benchmark for co-optimizing the design and control of soft robots. As seen in [Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots](https://evolutiongym.github.io/) (**NeurIPS 2021**).

[//]: # (<img src="images/teaser.gif" alt="teaser" width="800"/>)
![teaser](images/teaser.gif)

# Installation

Clone the repo and submodules:

```shell
git clone --recurse-submodules https://github.com/EvolutionGym/evogym.git
```

### Requirements

* Python 3.7/3.8
* Linux, macOS, or Windows with [Visual Studios 2017](https://visualstudio.microsoft.com/vs/older-downloads/)
* [OpenGL](https://www.opengl.org//)
* [CMake](https://cmake.org/download/)
* [PyTorch](http://pytorch.org/)

<!--- (See [installation instructions](#opengl-installation-on-unix-based-systems) on Unix based systems) --->

On **Linux only**:

```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
```

Either install Python dependencies with conda:

```shell
conda env create -f environment.yml
conda activate evogym
```

or with pip:

```shell
pip install -r requirements.txt
```

### Build and Install Package

To build the C++ simulation, build all the submodules, and install `evogym` run the following command:

```shell
python setup.py install
``` 

### Test Installation

cd to the `examples` folder and run the following script:

```shell
python gym_test.py
```

This script creates a random `5x5` robot in the `Walking-v0` environment. The robot is taking random actions. A window should open with a visualization of the environment -- kill the process from the terminal to close it.

<!--### OpenGL installation on Unix-based systems

To install OpenGL via [homebrew](https://brew.sh/), run the following commands:

```shell
brew install glfw
```
--->

# Usage

## Examples

To see example usage as well as to run co-design and control optimization experiments in EvoGym, please see the `examples` folder and its `README`.

## Tutorials

You can find tutorials for getting started with the codebase on our [website](https://evolutiongym.github.io/tutorials). Completed code from all tutorials is also available in the `tutorials` folder.

## Docs

You can find documentation on our [website](https://evolutiongym.github.io/documentation).

## Design Tool

For instructions on how to use the Evolution Gym Design Tool, please see [this repo](https://github.com/EvolutionGym/evogym-design-tool).

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
