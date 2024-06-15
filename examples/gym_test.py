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
