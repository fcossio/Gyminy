import pybullet as p
import time
import numpy as np
import gym
import math

from gym import error, spaces, utils
from gym.utils import seeding
from controller import *
from nao import Nao

class NaoStandUpEnv(gym.Env):

    def __init__(self):
        self.robot = Nao()

        self.exit = 0
        self.motors = self.robot.motorLimits.keys()

        high = np.ones([len(self.motors)])
        self.action_space = spaces.Box(-high, high)
        high = np.inf*np.ones([90])
        self.observation_space = spaces.Box(-high, high)

        self.initialMotors = self.robot.readMotorPosition()
        self.initialPosition = self.robot.getSelf().getPosition()
        self.initialAccel = (0, 0, 0)
        self.initialGyro = (0, 0)

    def step(self, action):
        self.exit = self.robot.step(self.robot.timeStep)

        jointPositions = dict()
        for j in range(len(self.motors)):
            jointPositions[self.motors[j]] = action[j]

        self.robot.setJointPositions(jointPositions)

        accel = self.robot.getAcceleration()
        gyro = self.robot.getGyroscope()
        position = self.robot.getPos()
        motors = self.robot.readMotorPosition()

        obs = [accel[0], accel[1], accel[2], gyro[0], gyro[1]]

        for m in motors:
            obs.append(m)
        #print("Observation: ", len(obs))
        #print(obs)

        return obs, self._get_reward(obs), self._is_done(obs), None

    def reset(self):
        self.robot.simulationReset()

    def _get_reward(self, state):

        inclination = abs(np.arctan(state[0]/state[1]))
        z_position = self.robot.getPos()[2]

        if math.isnan(inclination) and math.isnan(z_position):
            f = 0
        else:
            f = z_position - inclination

        print("-------------------------Rewards-----------------------------------")
        print("Inclination: ", inclination)
        print("Z position: ", z_position)

        print("f = ", f)

        return f

    def _is_done(self, obs):
        return False

    def getExit(self):
        return self.exit
