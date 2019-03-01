import gym_nao
import gym
import time

from gym import error, spaces, utils
from gym.utils import seeding


env = gym.make('gym_nao_standUp-v0')

episodes = 1000

for i in range(episodes):

    action = env.action_space.sample()

    for i in range(50):
        env.step(action)

        if env.exit == -1:
            print("Exit")
            break

        time.sleep(0.03)

    print("Episode Terminated!")
    obs = env.reset()
    time.sleep(1)
