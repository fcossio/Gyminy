import time
from datetime import timedelta

import gym
import roboschool
from gym.spaces import prng
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
      # _states are only useful when using LSTM policies
      actions, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(actions)
      
      # Stats
      for i in range(env.num_envs):
          episode_rewards[i][-1] += rewards[i]
          if dones[i]:
              episode_rewards[i].append(0.0)

    mean_rewards =  [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])     
        n_episodes += len(episode_rewards[i])   

    # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward

env_id = "RoboschoolHumanoid-v1"
save_path = 'Humanoid.pkl'
new_model = True
load_path = 'Humanoid.pkl.recovered'

tensorboard_path = 'tbHumanoid2'

n_timesteps = 20000000


num_cpu = 4  # Number of processes to use
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])# Create the vectorized environment

print('----------------------------------------')

if new_model:
    print('Creating new model')
    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log = tensorboard_path)
else:
    print('Loading model from: ' + load_path)
    model = PPO2.load(load_path, env, verbose=0, tensorboard_log = tensorboard_path)

print('----------------------------------------')
print('Evaluate initial performance of model...')
mean_reward_before_train = evaluate(model, num_steps=1000)
# Multiprocessed RL Training
start_time = time.time()
print('Evaluating processing time with ' + str(num_cpu) + ' cpu(s)')
model.learn(1000*num_cpu, log_interval=100)
total_time_multi = time.time() - start_time
start_time = time.time()
local_start_time = time.asctime(time.localtime(start_time))
print('----------------------------------------')
print('Start: '  + str(local_start_time))
expected_time = total_time_multi*(n_timesteps/(1000*num_cpu))
print('Expected processing time: ' + str(timedelta(seconds = expected_time)))
ETA = time.asctime( time.localtime(start_time + expected_time))
print('ETA: ' + str(ETA))
print('----------------------------------------')
print('Training for ' + str(n_timesteps) + ' timesteps')

try:
    model.learn(n_timesteps, log_interval=10000)###Train the model
except:                                         ### in case it breakes in the meantime, try to save the model
    print('Trying to save model...')
    print('Saving model to: ' + save_path + '.recovered')
    model.save(save_path + '.recovered')

finish_time = time.time()
delta_time = finish_time - start_time
print('----------------------------------------')
print('ETA was ' + str(ETA))
print('Finished: ' + str(time.asctime(time.localtime(finish_time))))
print('Total time elapsed: ' + str(timedelta(seconds = delta_time)))
print('----------------------------------------')
print('Evaluate final performace...')
mean_reward_after_train = evaluate(model, num_steps=1000)
print('Saving model to: ' + save_path)
model.save(save_path)
print('----------------------------------------')
print('Done')
