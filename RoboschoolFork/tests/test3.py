import sys
n_cpu = int(sys.argv[1])
total_train_timesteps = int(sys.argv[2])
import gym, roboschool, roboschoolfork_nao
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from time import time



# multiprocess environment

env = SubprocVecEnv([lambda: gym.make('NaoLLC-v1') for i in range(n_cpu)])
initial_timestep = 0


activation_function = tf.nn.relu
net_arch = [512,256,128]
# checkpoint_timesteps = 10000
start_time = time()
policy_kwargs = dict(
    #act_fun=activation_function,
    net_arch = net_arch)
model = PPO2(MlpPolicy, env, verbose=1, #tensorboard_log="./fixed_body_leg_cycle",
    learning_rate = 0.00025,
    nminibatches = n_cpu,
    n_steps = 128,
    gamma = 0.95,
    policy_kwargs=policy_kwargs)
# model = PPO2.load("ppo2_NaoForwardWalk11jul.pkl", env = env, tensorboard_log="./ppo2_NaoForwardWalk")
try:
    model.learn(total_timesteps=total_train_timesteps)
except:
    print("training aborted")
model.save(str(time())+".pkl")
print("Saved")
end_time = time()

print("Elapsed time training", end_time - start_time)
# del model # remove to demonstrate saving and loading
#
# model = PPO2.load("ppo2_NaoForwardWalk12jul.pkl")
# env = gym.make('RoboschoolNaoForwardWalk-v1')
# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     if dones:
#         obs = env.reset()
#     env.render()
