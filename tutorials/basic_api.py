from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot
import os
import numpy as np

# create a EvoWorld object by loading a simple environment created with the Evolution Gym Design Tool.

world = EvoWorld.from_json(os.path.join('world_data', 'simple_environment.json'))

# the best way to visualize environments is within the Evolution Gym Design Tool
# however we can also visualize them with this handy function

world.pretty_print()

# would like to add a randomly sampled 5x5 robot to this environment

robot_structure, robot_connections = sample_robot((5, 5))
world.add_from_array(
    name='robot', 
    structure=robot_structure, 
    x=3, 
    y=1, 
    connections=robot_connections)

world.pretty_print()

# we create a simulation using our world object to specify the locations of all objects/voxels

sim = EvoSim(world)
sim.reset()

# a viewer object will allow us to visualize our simulation
# the timer object allows us to control the frequency at which we step the simulation, which is useful for visualizations

viewer = EvoViewer(sim)
viewer.track_objects('robot', 'box')

# we put it all together in this loop in which we sample a random action for our simulation, step the simulation, and render it

while True:

    sim.set_action(
        'robot', 
        np.random.uniform(
            low = 0.6,
            high = 1.6,
            size=(sim.get_dim_action_space('robot'),))
        )
    sim.step()
    viewer.render('screen')

