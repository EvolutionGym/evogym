# Tutorials

This folder contains completed code for all tutorials on our [website](https://evolutiongym.github.io/tutorials).

## Custom Environment

To see an example of custom environment creation, see `envs/simple_env.py`. This environment is registered in `envs/__init__.py`, and can be visualized by running `python .\visualize_simple_env.py` from this directory.

## EvoGym Simulator API

See `basic_api.py` for a simple example of how to create, step, and render an EvoGym simulator with objects of your choice. EvoGym can be used to simulate any number of objects and robots (although simulation speed may slow with many objects).

To see understand the different rendering options available in EvoGym, see `rendering_options.py`.
You can run:

```bash
python .\rendering_options.py --render-option to-debug-screen
```

| Option             | Description                                                            |
|--------------------|------------------------------------------------------------------------|
| to-debug-screen    | Render to EvoGym's default viewer                                      |
| to-numpy-array     | Render to a numpy array (visualized with open cv)                      |
| special-options    | Render with special flags (for pretty visualization)                   |
| very-fast          | Render without fps limit                                               |
