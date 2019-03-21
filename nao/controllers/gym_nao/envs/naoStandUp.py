import pybullet as p
import time
import numpy as np
import gym
import math
from time import time
from .nao import Nao
from gym import error, spaces, utils
from gym.utils import seeding
from controller import *

class NaoStandUpEnv(gym.Env):

    def __init__(self):
        self.robot = Nao()
        self.freeze_upper_body = True
        self.exit = 0
        self.motors = self.robot.motor_names

        if self.freeze_upper_body:
            high = np.ones([12])
        else:
            high = np.ones([24])
        self.action_space = spaces.Box(-high, high)
        high = np.inf*np.ones([68])
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

        self.timeout += 1
        jointPositions = dict()
        defase = 0
        for j in range(len(self.motors)):
            if ( 7 <= j and j <= 14 ) or (20 <= j and j <= 27): #is a Phalanx. No action taken
                jointPositions[self.motors[j]] = -1.0
                defase +=1
            elif self.freeze_upper_body and j < 27: #freeze upper body
                jointPositions[self.motors[j]] = self.robot.INITIAL_MOTOR_POS[self.motors[j]]
                defase +=1
            else:
                jointPositions[self.motors[j]] = action[j-defase]

        for i in range(5):
            self.exit = self.robot.step(self.robot.timeStep)

        self.robot.setJointPositions(jointPositions)
        readings = self.robot.getAllReadings()
        ax,ay,az = readings[0:3]
        roll,pitch = readings[5:7]
        gyro = readings[3:5]
        x,y,z = self.robot.getPos()
        motor_values = readings[21:]

        obs = readings
        # print('action taken' + str(time()))
        return obs, self._get_reward(obs), self._is_done(obs), dict()

    def _get_reward(self, state):
        x,y,z = self.robot.getPos()
        y_position = y
        motors= state[49:]
        joints_at_limit_discount = 0
        for m in motors:
            if abs(m)>0.98:
                joints_at_limit_discount += 0.5
            else:
                joints_at_limit_discount += (m*m*m*m)/5

        joints_at_limit_discount /= 5
        print('joints limit discount' + str(joints_at_limit_discount/5))
        # print('Ankles: R=' + str(RAnklePitch) + ' L=' + str(LAnklePitch))
        roll,pitch = state[5:7]

        if math.isnan(roll) and math.isnan(y_position) and math.isnan(pitch):
            f = 0
        else:
            f = 2*(1.5 * y + x - 1.2*(abs(roll) + abs(pitch)))-joints_at_limit_discount
        self.fallen = self.hasFallen(y, roll, pitch)

        if self.fallen:
            f = -10 + self.timeout
        elif self.timeout > 99 and not self.fallen:
            f = 10 + self.timeout

        # print("-------------------------Rewards--------------------------------")
        # print("Timestep: ", self.timeout)
        # print("")
        # print("Roll: ", roll)
        # print("Pitch: ", pitch)
        # print("Y position: ", y_position)
        #
        # print("f = ", f)
        # print("----------------------------------------------------------------")
        return f

    def _is_done(self, state):
        if self.timeout >= 2000:
            return True
        else:
            return self.fallen

    def hasFallen(self, y, roll, pitch):
        if (y < 0.17 and (abs(roll)>0.2 or abs(pitch)>0.2)) or (abs(roll)>0.5 or abs(pitch)>0.5):
            # print('I fell :(')
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

            obs = readings

        return obs

    def getExit(self):
        return self.exit
