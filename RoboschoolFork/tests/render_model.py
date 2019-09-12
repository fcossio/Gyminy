import sys
import gym
import time
import matplotlib.pyplot as plt
from gym import wrappers
import roboschool
import roboschoolfork_nao
from stable_baselines import PPO2
import imageio
from tqdm import tqdm

model_path = sys.argv[1]
model = PPO2.load(model_path)
env = gym.make('NaoLLC-v1')
# env = wrappers.Monitor(env, './recording/' + str(time()) + '/')
# Enjoy trained agent
obs = env.reset()
episodes = 0
for i in range(5000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    #print(rewards)
    if dones:
        obs = env.reset()
        episodes += 1
