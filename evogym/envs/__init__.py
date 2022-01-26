
from evogym.envs.base import *
from evogym.envs.balance import *
from evogym.envs.manipulate import *
from evogym.envs.climb import *
from evogym.envs.flip import *
from evogym.envs.jump import *
from evogym.envs.multi_goal import *
from evogym.envs.change_shape import *
from evogym.envs.traverse import *
from evogym.envs.walk import *

from gym.envs.registration import register

## SIMPLE ##
register(
    id = 'Walker-v0',
    entry_point = 'evogym.envs.walk:WalkingFlat',
    max_episode_steps=500
)

register(
    id = 'BridgeWalker-v0',
    entry_point = 'evogym.envs.walk:SoftBridge',
    max_episode_steps=500
)

register(
    id = 'CaveCrawler-v0',
    entry_point = 'evogym.envs.walk:Duck',
    max_episode_steps=1000
)

register(
    id = 'Jumper-v0',
    entry_point = 'evogym.envs.jump:StationaryJump',
    max_episode_steps=500
)

register(
    id = 'Flipper-v0',
    entry_point = 'evogym.envs.flip:Flipping',
    max_episode_steps=600
)

register(
    id = 'Balancer-v0',
    entry_point = 'evogym.envs.balance:Balance',
    max_episode_steps=600
)

register(
    id = 'Balancer-v1',
    entry_point = 'evogym.envs.balance:BalanceJump',
    max_episode_steps=600
)

register(
    id = 'UpStepper-v0',
    entry_point = 'evogym.envs.traverse:StepsUp',
    max_episode_steps=600
)

register(
    id = 'DownStepper-v0',
    entry_point = 'evogym.envs.traverse:StepsDown',
    max_episode_steps=500
)

register(
    id = 'ObstacleTraverser-v0',
    entry_point = 'evogym.envs.traverse:WalkingBumpy',
    max_episode_steps=1000
)

register(
    id = 'ObstacleTraverser-v1',
    entry_point = 'evogym.envs.traverse:WalkingBumpy2',
    max_episode_steps=1000
)

register(
    id = 'Hurdler-v0',
    entry_point = 'evogym.envs.traverse:VerticalBarrier',
    max_episode_steps=1000
)

register(
    id = 'GapJumper-v0',
    entry_point = 'evogym.envs.traverse:Gaps',
    max_episode_steps=1000
)

register(
    id = 'PlatformJumper-v0',
    entry_point = 'evogym.envs.traverse:FloatingPlatform',
    max_episode_steps=1000
)

register(
    id = 'Traverser-v0',
    entry_point = 'evogym.envs.traverse:BlockSoup',
    max_episode_steps=600
)

## PACKAGE ##
register(
    id = 'Lifter-v0',
    entry_point = 'evogym.envs.manipulate:LiftSmallRect',
    max_episode_steps=300
)

register(
    id = 'Carrier-v0',
    entry_point = 'evogym.envs.manipulate:CarrySmallRect',
    max_episode_steps=500
)

register(
    id = 'Carrier-v1',
    entry_point = 'evogym.envs.manipulate:CarrySmallRectToTable',
    max_episode_steps=1000
)

register(
    id = 'Pusher-v0',
    entry_point = 'evogym.envs.manipulate:PushSmallRect',
    max_episode_steps=500
)

register(
    id = 'Pusher-v1',
    entry_point = 'evogym.envs.manipulate:PushSmallRectOnOppositeSide',
    max_episode_steps=600
)

register(
    id = 'BeamToppler-v0',
    entry_point = 'evogym.envs.manipulate:ToppleBeam',
    max_episode_steps=1000
)

register(
    id = 'BeamSlider-v0',
    entry_point = 'evogym.envs.manipulate:SlideBeam',
    max_episode_steps=1000
)

register(
    id = 'Thrower-v0',
    entry_point = 'evogym.envs.manipulate:ThrowSmallRect',
    max_episode_steps=300
)

register(
    id = 'Catcher-v0',
    entry_point = 'evogym.envs.manipulate:CatchSmallRect',
    max_episode_steps=400
)

### SHAPE ###
register(
    id = 'AreaMaximizer-v0',
    entry_point = 'evogym.envs.change_shape:MaximizeShape',
    max_episode_steps=600
)

register(
    id = 'AreaMinimizer-v0',
    entry_point = 'evogym.envs.change_shape:MinimizeShape',
    max_episode_steps=600
)

register(
    id = 'WingspanMazimizer-v0',
    entry_point = 'evogym.envs.change_shape:MaximizeXShape',
    max_episode_steps=600
)

register(
    id = 'HeightMaximizer-v0',
    entry_point = 'evogym.envs.change_shape:MaximizeYShape',
    max_episode_steps=500
)

### CLIMB ###
register(
    id = 'Climber-v0',
    entry_point = 'evogym.envs.climb:Climb0',
    max_episode_steps=400
)

register(
    id = 'Climber-v1',
    entry_point = 'evogym.envs.climb:Climb1',
    max_episode_steps=600
)

register(
    id = 'Climber-v2',
    entry_point = 'evogym.envs.climb:Climb2',
    max_episode_steps=1000
)

### MULTI GOAL ###
register(
    id = 'BidirectionalWalker-v0',
    entry_point = 'evogym.envs.multi_goal:BiWalk',
    max_episode_steps=1000
)