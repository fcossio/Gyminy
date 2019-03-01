import gym
import roboschool
import time

env = gym.make('RoboschoolNaoForwardWalk-v1')
env.reset()
env.render()
while True:
    env.step(env.action_space.sample())
