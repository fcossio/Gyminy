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

        self.timeout = 0
        self.fallen = False

    def step(self, action):
        for i in range(15):
            self.exit = self.robot.step(self.robot.timeStep)
        self.timeout += 1
        jointPositions = dict()
        for j in range(len(self.motors)):
            jointPositions[self.motors[j]] = action[j]

        self.robot.setJointPositions(jointPositions)
        readings = self.robot.getAllReadings()
        ax,ay,az = readings[0:3]
        roll,pitch = readings[5:7]
        gyro = readings[3:5]
        x,y,z = self.robot.getPos()
        motors = readings[21:]

        obs = [x,y,z,ax,ay,az,roll,pitch,0,gyro[0],gyro[1]]

        for m in motors:
            obs.append(m)
        #print("Observation: ", len(obs))
        #print(obs)

        return obs, self._get_reward(obs), self._is_done(obs), None

    def _get_reward(self, state):

        y_position = state[1]*1.4

        roll,pitch,yaw = state[6:9]

        if math.isnan(roll) and math.isnan(y_position) and math.isnan(pitch):
            f = 0
        else:
            f = 2*(y_position - 1.2*(abs(roll) + abs(pitch)))

        self.fallen = self.hasFallen(state)

        if self.fallen:
            f = -10
        elif self.timeout > 199 and not self.fallen:
            f = 10

        #print("-------------------------Rewards--------------------------------")
        #print("Timestep: ", self.timeout)
        #print("")
        #print("Roll: ", roll)
        #print("Pitch: ", pitch)
        #print("Y position: ", y_position)

        #print("f = ", f)
        #print("----------------------------------------------------------------")

        return f

    def _is_done(self, state):
        if self.timeout >= 200:
            return True
        else:
            return self.fallen

    def hasFallen(self, state):
        if state[1] < 0.165 and (abs(state[6])>0.3 or abs(state[7])>0.3):
            print("=====================================================")
            print("OOPS! I FELL")
            print("=====================================================")
            return True
        else:
            return False

    def reset(self):
        self.exit = self.robot.step(self.robot.timeStep)
        time.sleep(0.1)
        self.robot.simulationReset()

    def getExit(self):
        return self.exit
