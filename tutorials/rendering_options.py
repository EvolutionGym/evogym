from tabnanny import verbose
from evogym import EvoWorld, WorldObject, EvoSim, EvoViewer, sample_robot
import os
import numpy as np
import cv2

# create a simple Evolution Gym Environment as done in basic_api.py

# create world
world = EvoWorld.from_json(os.path.join('world_data', 'simple_environment.json'))

# add robot
robot_structure, robot_connections = sample_robot((5, 5))
world.add_from_array(
    name='robot', 
    structure=robot_structure, 
    x=3, 
    y=1, 
    connections=robot_connections)

# create simulation 
sim = EvoSim(world)
sim.reset()

# set up viewer
viewer = EvoViewer(sim)
viewer.track_objects('robot', 'box')

# select a rendering option
options = ['to-debug-screen', 'to-numpy-array', 'special-options', 'very-fast']
option = options[0]

print(f'\nUsing rendering option {option}...\n')

# if the 'very-fast' option is chosen, set the rendering speed to be unlimited
if option == 'very-fast':
    viewer.set_target_rps(None)

while True:

    sim.set_action(
        'robot', 
        np.random.uniform(
            low = 0.6,
            high = 1.6,
            size=(sim.get_dim_action_space('robot'),))
        )
    sim.step()

    # step and render to a debug screen
    if option == 'to-debug-screen':
        viewer.render('screen')

    # step and render to a numpy array
    # use open cv to visualize output
    if option == 'to-numpy-array':
        img = viewer.render('img')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.waitKey(1)
        cv2.imshow("Open CV Window", img)

    # rendering with more options
    if option == 'special-options':   
        img = viewer.render(
            'screen', 
            verbose = True,
            hide_background = False,
            hide_grid = True,
            hide_edges = False,
            hide_voxels = False)

    # rendering as fast as possible
    if option == 'very-fast':
        viewer.render('screen', verbose=True)

cv2.destroyAllWindows()
