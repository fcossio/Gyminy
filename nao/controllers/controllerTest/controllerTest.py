import gym_nao
import gym

from gym import error, spaces, utils

from datetime import timedelta
import time
import csv
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import tensorflow as tf


def evaluate(model, num_steps=512):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        actions, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(actions)
        # Stats

        episode_rewards[-1] += rewards
        if dones:
            env.reset()
            episode_rewards.append(0.0)

    mean_rewards = 0.0
    n_episodes = len(episode_rewards) - 1
    mean_reward = np.sum(episode_rewards)/n_episodes

    print("Mean episode reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward


experiment_name = "LeftStep"
experiment_version = "2"
log_path = "log_"+ experiment_name + ".v" + experiment_version +".csv"
env_id = 'gym_nao_standUp-v1'
save_path = experiment_name + ".v" + experiment_version + ".plk"
tensorboard_path = "../tensorboards/" + experiment_name + ".v" + experiment_version
new_model = False
n_timesteps = 500000
batch_n_timesteps = 2048
n_steps = 128
learning_rate = 0.00025
activation_function = tf.nn.relu
net_arch = [256,64,32]
noptepochs = 15
env = gym.make(env_id)
envVec = DummyVecEnv([lambda : env])

def getTrainedTimesteps():
    with open(log_path,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        a = list(csv.reader(f))[-1]
    return int(a[1])

if new_model:
    with open(log_path, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        row = ["Time","TrainedTimesteps","mean_reward", "net_arch", "learning_rate", "n_steps","noptepochs"]
        writer.writerow(row)
        row = [str(time.ctime()),0,learning_rate,str(net_arch),n_steps,noptepochs]
        writer.writerow(row)
    print('Creating new model' + save_path)
    policy_kwargs = dict(act_fun=activation_function, net_arch = net_arch)
    model = PPO2(MlpPolicy, envVec, policy_kwargs=policy_kwargs,
        learning_rate = learning_rate, cliprange = 0.2, nminibatches = 1, noptepochs = noptepochs, n_steps=n_steps,
        verbose=0, tensorboard_log = tensorboard_path)
else:
    print('Loading Model ' + save_path)
    model = PPO2.load(save_path, envVec, verbose=0, tensorboard_log = tensorboard_path)

trainedT = getTrainedTimesteps()
print("Last trained timestep was : ", trainedT)
if trainedT == 0:
    episode_mean = None
    print('----------------------------------------')
    # RL Training
    start_time = time.time()
    print("Evaluating initial model performance")
    episode_mean = evaluate(model)
    print('Evaluating processing time with 1 cpu')
    model.learn(total_timesteps = 512)
    total_time_multi = time.time() - start_time
    start_time = time.time()
    local_start_time = time.asctime(time.localtime(start_time))
    print('----------------------------------------')
    print('Start: '  + str(local_start_time))
    expected_time = total_time_multi*(n_timesteps/(512))
    print('Expected processing time: ' + str(timedelta(seconds = expected_time)))
    ETA = time.asctime( time.localtime(start_time + expected_time))
    print('ETA: ' + str(ETA))
    print('----------------------------------------')
    trainedT = 1 #just to make it learn
if trainedT < n_timesteps:
    if new_model:
        print("REMEMBER TO TURN NEW MODEL = FALSE OR THE ALGORITHM WONT LEARN")
        trainedT = 0
    else:
        episode_mean = None
    print("Learning for ", batch_n_timesteps, " timesteps before resetting")
    model.learn(total_timesteps = batch_n_timesteps)
    print("Saving to " + save_path)
    model.save(save_path)
    if ((trainedT + batch_n_timesteps) % (10 * batch_n_timesteps)) == 0:
        episode_mean = evaluate(model)
    with open(log_path, 'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        row = [str(time.ctime()),trainedT + batch_n_timesteps, episode_mean]
        writer.writerow(row)
    print("Finished on global timestep ", trainedT + batch_n_timesteps)

    env.reset(master = True)

if trainedT > n_timesteps:
    trainedT = getTrainedTimesteps()
    episode_mean = None
    episode_mean = evaluate(model,1024)
    with open(log_path, 'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        row = [str(time.ctime()),trainedT, episode_mean]
        writer.writerow(row)
    print("DONE")
