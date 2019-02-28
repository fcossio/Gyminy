import pybullet as p
import time
import numpy as np
import gym

from gym import error, spaces, utils
from gym.utils import seeding
from controller import *
from nao import Nao

class NaoStandUpEnv(gym.Env):

    def __init__(self):
        self.robot = Nao()

        self.exit = 0
        self.motors = self.robot.motorLimits.keys()

        print(self.motors)

        high = np.ones([len(self.motors)])              #25 Degrees of Freedom
        self.action_space = spaces.Box(-high, high)
        high = np.inf*np.ones([90])                     #Observations
        self.observation_space = spaces.Box(-high, high)

    def step(self, action):
        self.exit = self.robot.step(self.robot.timeStep)
        
        jointPositions = dict()
        for j in range(len(self.motors)):
            jointPositions[self.motors[j]] = action[j]

        self.robot.setJointPositions(jointPositions)
        accel = self.robot.getAcceleration()
        gyro = self.robot.getGyroscope()
        position = self.robot.getPos()

        obs = accel
        return obs, self._get_reward(obs), self._is_done(obs), None

    def reset(self):
        accel = self.robot.getAcceleration()
        gyro = self.robot.getGyroscope()
        position = self.robot.getPos()

        obs = accel
        return obs

    def _get_reward(self, state):
        return 0

    def _is_done(self, obs):
        return False

    def getExit(self):
        return self.exit
