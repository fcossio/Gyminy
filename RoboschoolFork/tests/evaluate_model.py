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
length = 0
total_length = 0
max_len = 0
ep_rew = 0
total_rew = 0
max_rew = 0
for i in tqdm(range(1000)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    length += 1
    ep_rew += rewards
    if dones:
        obs = env.reset()
        episodes += 1
        total_length+=length
        total_rew += ep_rew
        max_len = max(max_len,length)
        max_rew = max(max_rew,ep_rew)
        length = 0
        ep_rew = 0
print("av len",total_length/episodes)
print("av ep rew",total_rew/episodes)
print("max ep len",max_len)
print("max ep rew",max_rew)
