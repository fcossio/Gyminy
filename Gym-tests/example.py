import gym
env = gym.make('Humanoid-v2')
env.reset()
for _ in range(100000):
    env.render()
    env.step(env.action_space.sample()) # take a random action