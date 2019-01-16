#Environments

#Here’s a bare minimum example of getting something running. 
#This will run an instance of the CartPole-v0 environment for 
#1000 timesteps, rendering the environment at each step. You 
#should see a window pop up rendering the classic cart-pole problem:
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
