import time
from typing import Optional, Tuple, List

import numpy as np
from evogym.sim import EvoSim
from evogym.simulator_cpp import Viewer, Camera
from evogym.utils import Timer


class EvoViewer():
    """
    Visualize an Evolution Gym simulation.

    Args:
        sim_to_view (EvoSim): simulation to view. 
        target_rps (Optional[int]): target rendering speed in renders per second. If `None`, renders as fast as possible. (default = 50)
        pos (Tuple[float, float]): position of camera measured in voxels `(x,y)`. (default = (12, 4))
        view_size (Tuple[float, float]): size of viewbox -- the number of voxels the camera can see `(w, h)`. (default = (40, 20)) 
        resolution (Tuple[int, int]): resolution of image generated in pixels `(w, h)`. (default = (1200, 600))
    """
    def __init__(
        self, 
        sim_to_view: EvoSim, 
        target_rps: Optional[int] = 50,
        pos: Tuple[float, float] = (12, 4), 
        view_size: Tuple[float, float] = (40, 20), 
        resolution: Tuple[int, int] = (1200, 600)) -> None:

        self._sim = sim_to_view

        self._has_init_viewer = False
        self._has_init_screen_camera = False
        self._has_init_img_camera = False
        self._is_showing_debug_window = False

        self._last_rendered = -1
        self._tracking_objects: List[str] = []
        self._tracking_padding = (5, 3)
        self._tracking_multiplier = (0.2, 0.2)
        self._tracking_lock = {'x': False, 'y': False, 'width': False, 'height': False}
        self._tracking_history = None
        self._tracking_history2 = None
        self._tracking_sum = [0,0,0,0]
        self._old_targets = None
        self._last_tracked_sim_time = 0

        self.set_target_rps(target_rps)
        self.set_pos(pos)
        self.set_view_size(view_size)
        self.set_resolution(resolution)
        self._voxel_size = 0.1

    def set_pos(self, pos: Tuple[float, float]) -> None:
        """
        Set position of camera.

        Args:
            pos (Tuple[float, float]): new position of camera measured in voxels `(x,y)`.
        """
        if not isinstance(pos, tuple):
            raise TypeError(
                f'pos of camera must be of type tuple. Got type {type(pos)}')
        self.pos = pos

        if self._has_init_screen_camera:
            self.screen_camera.set_pos(self.pos[0] * self._voxel_size,
                                       self.pos[1] * self._voxel_size)
        if self._has_init_img_camera:
            self.img_camera.set_pos(self.pos[0] * self._voxel_size,
                                    self.pos[1] * self._voxel_size)

    def set_view_size(self, view_size: Tuple[float, float]) -> None:
        """
        Set size of viewbox -- the number of voxels the camera can see.

        Args:
            view_size (Tuple[float, float]): new size of viewbox `(w, h)`.
        """
        if not isinstance(view_size, tuple):
            raise TypeError(
                f'view size of camera must be of type tuple. Got type {type(view_size)}'
            )
        self.view_size = view_size

        if self._has_init_screen_camera:
            self.screen_camera.set_size(self.view_size[0] * self._voxel_size,
                                        self.view_size[1] * self._voxel_size)
        if self._has_init_img_camera:
            self.img_camera.set_size(self.view_size[0] * self._voxel_size,
                                     self.view_size[1] * self._voxel_size)

    def set_resolution(self, resolution: Tuple[int, int]) -> None:
        """
        Set resolution of image generated in pixels.

        Args:
            view_size (Tuple[float, float]): new resolution `(w, h)`.
        """
        if not isinstance(resolution, tuple):
            raise TypeError(
                f'resolution of camera must be of type tuple. Got type {type(resolution)}'
            )
        self.resolution = resolution

        if self._has_init_screen_camera:
            self.screen_camera.set_size(self.resolution[0], self.resolution[1])
        if self._has_init_img_camera:
            self.img_camera.set_size(self.resolution[0], self.resolution[1])

    def set_target_rps(self, target_rps: Optional[int]) -> None:
        """
        Set the target render frequency, in renders per second.

        Args:
            target_rps (Optional[int]): target rendering speed in renders per second. If `None`, renders as fast as possible. (default = 50)
        """
        self._target_rps = int(target_rps) if target_rps is not None else None
        assert self._target_rps is None or self._target_rps > 1, (
            'The lowest supported target rps is 2'
        )
        self._timer = Timer(self._target_rps)

    def show_debug_window(self,) -> None:
        """
        Make the debug window visible.
        """
        self._init_viewer()
        self._viewer.show_debug_window()

    def hide_debug_window(self,) -> None:
        """
        Make the debug window invisible.
        """
        if self._has_init_viewer:
            self._viewer.hide_debug_window()
    
    def track_objects(self, *objects: Tuple[str]) -> None:
        """
        Set objects for the viewer to automatically track. The viewer tracks objects by adjusting its `pos` and `view_size` automatically every time before rendering.

        Args:
            Comma separated names of objects for the viewer to track.
        """
        self._tracking_objects = list(objects)

    def set_tracking_settings(self, **settings) -> None:
        """
        Adjust viewer object tracking settings.

        Args:
            padding (Tuple[float, float]): padding, measured in voxels, to apply to the adjusted viewbox `(padding-x, padding-y)`.
            scale (Tuple[float, float]): scaling to apply to the adjusted viewbox `(scale-x, scale-y)`.
            lock_x (Union[bool, float]): locks the x-pos of the viewer to the given value, or unlocks if set to `False`. 
            lock_y (Union[bool, float]): locks the y-pos of the viewer to the given value, or unlocks if set to `False`. 
            lock_width (Union[bool, float]): locks the width of the viewer's viewbox to the given value, or unlocks if set to `False`. 
            lock_height (Union[bool, float]): locks the height of the viewer's viewbox to the given value, or unlocks if set to `False`. 
        """
        valid = ['padding', 'scale', 'lock_x', 'lock_y', 'lock_width', 'lock_height']

        for arg, value in settings.items():
            if arg not in valid:
                raise ValueError(
                    f'invalid tracking setting argument {arg}. The valid arguments are {valid}'
                )
            if arg == 'padding' or arg == 'scale':
                if not isinstance(value, tuple):
                    raise TypeError(
                        f'{arg} must be of type tuple. Got type {type(value)}'
                    )
                if not len(value) == 2:
                    raise TypeError(
                        f'{arg} must have length 2. Got length {len(value)}'
                    )
                if arg == 'padding':
                    self._tracking_padding = value
                if arg == 'scale':
                    self._tracking_multiplier = value
            else:
                if not (isinstance(value, bool) or isinstance(value, int) or isinstance(value, float)):
                    raise ValueError(
                        f'{arg} must be of type bool, int, or float. Got type {type(value)}'
                    )
                if isinstance(value, bool) and value == True:
                    raise ValueError(
                        f'{arg} can only have values of *numeric* or *False*. Got {value}'
                    )
                parsed_arg = arg.split('_')[1]
                self._tracking_lock[arg] = parsed_arg 

    def render(self,
               mode: str ='screen',
               verbose: bool = False,
               hide_background: bool = False,
               hide_grid: bool = False,
               hide_edges: bool = False,
               hide_voxels: bool = False) -> Optional[np.ndarray]:
        """
        Render the simulation.

        Args:
            mode (str): values of 'screen' and 'human' will render to a debug window. If set to 'img' will return an image array.
            verbose (bool): whether or not to print the rendering speed (rps) every second.
            hide_background (bool): whether or not to render the cream-colored background. If shut off background will be white.
            hide_grid (bool): whether or not to render the grid.
            hide_edges (bool): whether or not to render edges around all objects.
            hide_voxels (bool): whether or not to render voxels.

        Returns:
            Optional[np.ndarray]: if `mode` is set to `img`, will return an image array.
        """

        accepted_modes = ['screen', 'human', 'img']
        if not mode in accepted_modes:
            raise ValueError(
                f'mode {mode} is not a valid mode. The valid modes are {accepted_modes}'
            )

        self._init_viewer()
        render_settings = (hide_background, hide_grid, hide_edges, hide_voxels)

        current_time = self._sim.get_time()
        if current_time < self._last_rendered:
            self._last_rendered = current_time-50
        while self._last_rendered < current_time:
            self._update_tracking()
            self._last_rendered += 1

        out = None

        if mode == 'screen' or mode == 'human':
            if not self._is_showing_debug_window:
                self.show_debug_window()
                self._is_showing_debug_window = True
            if not self._has_init_screen_camera:
                self._init_screen_camera()
                self._has_init_screen_camera = True
            self._viewer.render(self.screen_camera, *render_settings)

        if mode == 'img':
            if not self._has_init_img_camera:
                self._init_img_camera()
                self._has_init_img_camera = True
            self._viewer.render(self.img_camera, *render_settings)

            img_out = self.img_camera.get_image()
            img_out = np.array(img_out)
            img_out.resize(self.img_camera.get_resolution_height(),
                           self.img_camera.get_resolution_width(), 3)

            out = img_out

        self._timer.step(verbose=verbose)
        while not self._timer.should_step():
            time.sleep(0.001)
        
        return out

    def _update_tracking(self,) -> None:
        """
        Updates tracking of objects.
        """
        if len(self._tracking_objects) == 0:
            return

        # calcluate new values
        x_min = float('inf')
        x_max = -float('inf')
        y_min = float('inf')
        y_max = -float('inf')
        
        for obj_name in self._tracking_objects:
            pos_array = self._sim.object_pos_at_time(self._sim.get_time(), obj_name)
            x_min = min(np.min(pos_array, 1)[0], x_min)
            x_max = max(np.max(pos_array, 1)[0], x_max)
            y_min = min(np.min(pos_array, 1)[1], y_min)
            y_max = max(np.max(pos_array, 1)[1], y_max)

        x_min -= self._tracking_padding[0]
        y_min -= self._tracking_padding[1]
        x_max += self._tracking_padding[0]
        y_max += self._tracking_padding[1]

        x_min -= (self._tracking_multiplier[0]*(x_max-x_min))/2
        y_min -= (self._tracking_multiplier[1]*(y_max-y_min))/2
        x_max += (self._tracking_multiplier[0]*(x_max-x_min))/2
        y_max += (self._tracking_multiplier[1]*(y_max-y_min))/2

        new_x = (x_min + x_max)/2.0
        new_y = (y_min + y_max)/2.0

        x_range = (x_max - x_min)
        y_range = (y_max - y_min)

        #prevent jitters
        if self._old_targets is None:
            self.set_pos((new_x, new_y))
            self.set_view_size((x_range, y_range))
            self._old_targets = (new_x, new_y, x_range, y_range)

        if abs(self._old_targets[0] - new_x) < 1.0:
            new_x = self._old_targets[0]
        if abs(self._old_targets[1] - new_y) < 1.0:
            new_y = self._old_targets[1]
        if abs(1-self._old_targets[2]/x_range) < 0.3:
            x_range = self._old_targets[2]
        if abs(1-self._old_targets[3]/y_range) < 0.3:
            y_range = self._old_targets[3]

        # new_x_v = new_x - self._old_targets[0]
        # new_y_v = new_y - self._old_targets[1]
        # x_range_v = x_range - self._old_targets[2]
        # y_range_v = y_range - self._old_targets[3]

        self._old_targets = (new_x, new_y, x_range, y_range)

        # update smoothing
        if self._tracking_history is None:
            self._tracking_history = (new_x, new_y, x_range, y_range)
        if self._tracking_history2 is None:
            self._tracking_history2 = (new_x, new_y, x_range, y_range)

        new_x_v = self._tracking_history[0] - self._tracking_history2[0]
        new_y_v = self._tracking_history[1] - self._tracking_history2[1]
        x_range_v = self._tracking_history[2] - self._tracking_history2[2]
        y_range_v = self._tracking_history[3] - self._tracking_history2[3]

        self._tracking_sum[0] += new_x - self._tracking_history[0]
        self._tracking_sum[1] += new_y - self._tracking_history[1]
        self._tracking_sum[2] += x_range - self._tracking_history[2]
        self._tracking_sum[3] += y_range - self._tracking_history[3]

        for i in range(4):
            self._tracking_sum[i] *= 0.9

        kp = 0.04
        ki = 0.005
        kd = 0.08

        new_x = self.pos[0] + kp*(new_x - self._tracking_history[0]) + kd*new_x_v + ki*self._tracking_sum[0]
        new_y = self.pos[1] + kp*(new_y - self._tracking_history[1]) + kd*new_y_v + ki*self._tracking_sum[1]
        x_range = self.view_size[0] + kp*(x_range - self._tracking_history[2]) + kd*x_range_v + ki*self._tracking_sum[2]
        y_range = self.view_size[1] + kp*(y_range - self._tracking_history[3]) + kd*y_range_v + ki*self._tracking_sum[3]

        # convergence1 = 0.05
        # convergence2 = 0.005

        # new_x_v = new_x - (self._tracking_history[0]*(1-convergence1) + new_x*(convergence1))
        # new_y_v = new_y - (self._tracking_history[1]*(1-convergence1) + new_y*(convergence1))
        # x_range_v = x_range - (self._tracking_history[2]*(1-convergence1) + x_range*(convergence1))
        # y_range_v = y_range - (self._tracking_history[3]*(1-convergence1) + y_range*(convergence1))
        
        # new_x = self._tracking_history[0]*(1-convergence1) + new_x*(convergence1) - new_x_v*convergence2
        # new_y = self._tracking_history[1]*(1-convergence1) + new_y*(convergence1) - new_y_v*convergence2
        # x_range = self._tracking_history[2]*(1-convergence1) + x_range*(convergence1) - x_range_v*convergence2
        # y_range = self._tracking_history[3]*(1-convergence1) + y_range*(convergence1) - y_range_v*convergence2

        if not isinstance(self._tracking_lock['x'], bool):
            new_x = self._tracking_lock['x']
        if not isinstance(self._tracking_lock['y'], bool):
            new_y = self._tracking_lock['y']
        if not isinstance(self._tracking_lock['width'], bool):
            x_range = self._tracking_lock['width']
        if not isinstance(self._tracking_lock['height'], bool):
            y_range = self._tracking_lock['height']

        self._tracking_history2 = tuple([data for data in self._tracking_history])
        self._tracking_history = (new_x, new_y, x_range, y_range)

        # fix aspect ratio
        screen_width_over_height = self.resolution[0] / self.resolution[1]

        if x_range/y_range < screen_width_over_height:
            x_range *= (y_range/x_range) * screen_width_over_height

        elif x_range/y_range > screen_width_over_height:
            y_range /= (y_range/x_range) * screen_width_over_height

        self.set_pos((new_x, new_y))
        self.set_view_size((x_range, y_range))

    def _init_screen_camera(self,) -> None:
        """
        Initializes camera for rendering to debug window.
        """
        self.screen_camera = Camera(False)
        self.screen_camera.set_pos(self.pos[0] * self._voxel_size,
                                   self.pos[1] * self._voxel_size)
        self.screen_camera.set_size(self.view_size[0] * self._voxel_size,
                                    self.view_size[1] * self._voxel_size)
        self.screen_camera.set_resolution(self.resolution[0],
                                          self.resolution[1])

    def _init_img_camera(self,) -> None:
        """
        Initializes camera for rendering to image arrays.
        """
        self.img_camera = Camera(True)
        self.img_camera.set_pos(self.pos[0] * self._voxel_size,
                                self.pos[1] * self._voxel_size)
        self.img_camera.set_size(self.view_size[0] * self._voxel_size,
                                 self.view_size[1] * self._voxel_size)
        self.img_camera.set_resolution(self.resolution[0], self.resolution[1])

    def _init_viewer(self,):
        """
        Initializes viewer. Done automatically before rendering.
        """
        if not self._has_init_viewer:
            self._viewer = Viewer(self._sim)
            self._has_init_viewer = True
