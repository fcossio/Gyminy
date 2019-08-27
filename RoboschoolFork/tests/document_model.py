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
env = gym.make('RoboschoolNaoForwardWalk-v1')
# env = wrappers.Monitor(env, './recording/' + str(time()) + '/')
# Enjoy trained agent
obs = env.reset()
episodes = 0
with imageio.get_writer(model_path.split('.')[0]+'.gif', mode='I', fps=30) as writer:
    for i in tqdm(range(300)):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()
            episodes += 1
        writer.append_data(env.render(mode='rgb_array'))
