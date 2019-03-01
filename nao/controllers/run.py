import gym_nao
import gym
import pybullet as p
import time

p.connect(p.GUI)
p.loadURDF("gym_nao/src/plane.urdf")
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

env = gym.make('gym_nao_forwardWalk-v0')

obs = env.reset()

while True:
        obs,reward,done,info = env.step(env.action_space.sample())

        time.sleep(env.timeStep)
