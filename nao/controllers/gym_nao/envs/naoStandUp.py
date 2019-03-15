import pybullet as p
import time
import numpy as np
import gym
import math

from .nao import Nao
from gym import error, spaces, utils
from gym.utils import seeding
from controller import *

class NaoStandUpEnv(gym.Env):

    def __init__(self):
        self.robot = Nao()

        self.exit = 0
        self.motors = list(self.robot.motorLimits.keys())

        high = np.ones([len(self.motors)])
        self.action_space = spaces.Box(-high, high)
        high = np.inf*np.ones([58])
        self.observation_space = spaces.Box(-high, high)

        self.node = self.robot.getSelf()
        self.translation = self.node.getField("translation")
        self.rotation = self.node.getField("rotation")

        self.init_translation = [0, 0.333, 0]
        self.init_rotation = [1, 0, 0, -1.5708]

        #print("Initial Position: ", self.init_translation)
        #print("Translation Field: ", self.translation.getTypeName())
        #print("Initial Orientation: ", self.init_rotation)
        #print("Rotation Field: ", self.rotation.getTypeName())

        self.timeout = 0
        self.fallen = False

    def step(self, action):

        for i in range(10):
            self.exit = self.robot.step(self.robot.timeStep)

        self.timeout += 1
        jointPositions = dict()

        for j in range(len(self.motors)):
            #print(type(action[j]))
            jointPositions[self.motors[j]] = action[j]

        self.robot.setJointPositions(jointPositions)
        readings = self.robot.getAllReadings()
        ax,ay,az = readings[0:3]
        roll,pitch = readings[5:7]
        gyro = readings[3:5]
        x,y,z = self.robot.getPos()
        motor_values = readings[21:]

        obs = [x,y,z,ax,ay,az,roll,pitch,0,gyro[0],gyro[1]]

        for m in motor_values:
            obs.append(m)
        #print("Observation: ", len(obs))
        #print(obs)

        return obs, self._get_reward(obs), self._is_done(obs), dict()

    def _get_reward(self, state):

        y_position = state[1]*1.4

        roll,pitch,yaw = state[6:9]

        if math.isnan(roll) and math.isnan(y_position) and math.isnan(pitch):
            f = 0
        else:
            f = 2*(1.5*y_position - 1.2*(abs(roll) + abs(pitch)))

        self.fallen = self.hasFallen(state)

        if self.fallen:
            f = -20
        elif self.timeout > 99 and not self.fallen:
            f = 20

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
        if (state[1] < 0.17 and (abs(state[6])>0.2 or abs(state[7])>0.2)) or (abs(state[6])>0.5 or abs(state[7])>0.5):
            return True
        else:
            return False

    def reset(self, master=False):
        if master:
            self.robot.simulationReset()
        else:
            for i in range(100):
                self.exit = self.robot.step(self.robot.timeStep)

            self.node.resetPhysics()
            #self.exit = self.robot.step(self.robot.timeStep)

            self.robot.resetRobotPosition()
            for i in range(15):
                self.exit = self.robot.step(self.robot.timeStep)

            self.timeout = 0

            #self.node.resetPhysics()
            self.robot.simulationResetPhysics()

            readings = self.robot.getAllReadings()
            ax,ay,az = readings[0:3]
            roll,pitch = readings[5:7]
            gyro = readings[3:5]
            x,y,z = self.robot.getPos()
            motor_values = readings[21:]

            obs = [x,y,z,ax,ay,az,roll,pitch,0,gyro[0],gyro[1]]

            for m in motor_values:
                obs.append(m)

        return obs

    def getExit(self):
        return self.exit
