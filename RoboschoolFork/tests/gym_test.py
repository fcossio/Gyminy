import roboschool, gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import time

env = gym.make('RoboschoolHumanoid-v1')
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=10)

obs = env.reset()
env.render()

for i in range(1000):
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

model.learn(total_timesteps=10)

obs = env.reset()
env.render()

for i in range(1000):
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
