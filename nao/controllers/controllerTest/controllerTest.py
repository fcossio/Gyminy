import gym_nao
import gym
import time

from gym import error, spaces, utils
from gym.utils import seeding

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2

env = gym.make('gym_nao_standUp-v0')
env = DummyVecEnv([lambda : env])

model = PPO2(MlpPolicy, env, verbose=1)

episodes = 10000

print("Learning...")

model.learn(total_timesteps = episodes)

print("Finished Learning!")

model.save("PPO2_Balance")

for i in range(episodes):

    obs = env.reset()
    done = False
    score = 0
    j = 0

    while done is False:
        j += 1

        action,_states = model.predict(obs)
        obs,reward,done,_ = env.step(action)

        print("Timestep: ", j, "     Reward: ", reward)

        score += reward

        #time.sleep(0.01)

    print("==========================================================")
    print("Episode ", i, " Terminated!")
    print("Total Score = ", score)
    print("Average Reward = ", score/i)
