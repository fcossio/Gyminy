import gym_nao
import gym
import time

from gym import error, spaces, utils
from gym.utils import seeding


env = gym.make('gym_nao_standUp-v0')

obs = env.reset()
action = env.action_space.sample()

while True:
    env.step(action)

    if env.exit == -1:
        print("Exit")
        break

    time.sleep(1)
