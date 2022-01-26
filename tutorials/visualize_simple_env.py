import gym
from evogym import sample_robot

# import envs from the envs folder and register them
import envs

if __name__ == '__main__':

    # create a random robot
    body, connections = sample_robot((5,5))

    # make the SimpleWalkingEnv using gym.make and with the robot information
    env = gym.make('SimpleWalkingEnv-v0', body=body)
    env.reset()

    # step the environment for 500 iterations
    for i in range(500):

        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        env.render(verbose=True)

        if done:
            env.reset()

    env.close()
