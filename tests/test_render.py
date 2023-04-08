from unittest import TestCase

import evogym.envs
import gym
from evogym import sample_robot


class RenderTest(TestCase):
    def test_it(self):
        body, connections = sample_robot((5, 5))
        env = gym.make("Walker-v0", body=body)
        env.reset()

        for _ in range(100):
            action = env.action_space.sample() - 1
            ob, reward, done, info = env.step(action)
            env.render()

            if done:
                env.reset()

        env.close()
