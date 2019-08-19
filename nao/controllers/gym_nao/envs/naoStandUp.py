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
        roll,pitch = state[5:7]
        accel = state[0:3]
        y_accel = math.pow(abs(accel[1]-9.81), 1)
        #print("Discount: ", accel)

        #torque discount
        torque_discount = 0
        for t in self.torques.values():
            torque_discount += abs(t)
        torque_discount /= 60

        #height discount
        height_discount = math.exp(math.pow(0.333-y,2) * -5)
        #pitch+roll discount
        pitch_roll_discount = abs(pitch) + abs(roll)

        #right_step_pose_delta=np.exp(-np.sum(np.power(np.subtract(list(self.robot.RIGHT_STEP.values()),state[28:68]),2)))
        #x = self._clock()
        #if x > 0.5: #para hacer una funcion simetrica
        #    x = 1 - x
        #right_step_factor = np.exp(-((x)*4)**2)
        clock = self.timeout/10 % 1

        if(clock<0.25):
            test = np.sum(np.power(np.subtract(list(self.robot.LEFT_MID.values()),state[28:68]),2))
        elif(clock < 0.5):
            test = np.sum(np.power(np.subtract(list(self.robot.LEFT_STEP.values()),state[28:68]),2))
        elif(clock < 0.75):
            test = np.sum(np.power(np.subtract(list(self.robot.RIGHT_MID.values()),state[28:68]),2))
        else:
            test = np.sum(np.power(np.subtract(list(self.robot.RIGHT_STEP.values()),state[28:68]),2))

        pose_reward = math.exp(-test)

        #left_step_pose_delta=np.exp(-np.sum(np.power(np.subtract(list(self.robot.LEFT_STEP.values()),state[28:68]),2)))
        #x = self._clock() - 0.5 #defasar medio ciclo
        #if x > 0.5:
        #    x = 1 - x
        #left_step_factor = np.exp(-((x)*4)**2)

        #self.robot.display_write(3, 'L:%.3f R:%.3f'%(left_step_factor * left_step_pose_delta, right_step_factor * right_step_pose_delta))
        #pose_reward = (left_step_factor * left_step_pose_delta + right_step_factor * right_step_pose_delta)


        forward_discount = math.exp(math.pow(x-(2+self.robot.INITIAL_TRANS[0]),2) * -2)

        if math.isnan(roll) and math.isnan(y_position) and math.isnan(pitch):
            f = 0
        else:
            f = -0.1*torque_discount + 0.3*height_discount + 0.6*pose_reward + 0.1*forward_discount

        self.fallen = self.hasFallen(y, roll, pitch)
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
