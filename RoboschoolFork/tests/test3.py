import gym, roboschool, roboschoolfork_nao
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('RoboschoolNaoForwardWalk-v1') for i in range(n_cpu)])
#
initial_timestep = 0
total_train_timesteps = 100
activation_function = tf.nn.tanh
net_arch = [256,128,64]
# checkpoint_timesteps = 10000

policy_kwargs = dict(act_fun=activation_function, net_arch = net_arch)
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo2_NaoForwardWalk12jul",
   policy_kwargs=policy_kwargs)
# model = PPO2.load("ppo2_NaoForwardWalk11jul.pkl", env = env, tensorboard_log="./ppo2_NaoForwardWalk")
model.learn(total_timesteps=total_train_timesteps)
model.save("ppo2_NaoForwardWalk12jul.pkl")
print("Saved")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_NaoForwardWalk12jul.pkl")
env = gym.make('RoboschoolNaoForwardWalk-v1')
# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
    env.render()
