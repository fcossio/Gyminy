import pybullet as p
import time
import numpy as np
import gym
import math
from time import time
from .nao import Nao
from .nao_plotter import NaoPlotter
from gym import error, spaces, utils
from gym.utils import seeding
from controller import *
import csv

class NaoStandUp1Env(gym.Env):

    def __init__(self):

        self.robot = Nao()
        self.freeze_upper_body = True
        self.exit = 0
        self.motors = self.robot.motor_names
        self.torques = dict()
        #plot in simulation time
        self.position_vector = [[0,0,0]]
        self.plot = False
        if self.plot:
            self.time_vector =[[0,0,0]]

            self.velocity_vector = [[0,0,0,0,0,0]]
            #self.plotter = NaoPlotter(title='Nao StandUp1', xlabel='time[s]', ylabel='distance[m]')
        self.plot_at_end_of_episode = False
        #for further plotting
        self.saveTorquesToCSV = False
        self.torques_CSV_path = 'torques.csv'
        self.pos_rot_CSV_path = 'pos_rot.csv'
        if self.saveTorquesToCSV:

            with open(self.torques_CSV_path, 'w+') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                row=list(self.robot.motor_names)
                row.insert(0,'Time')
                writer.writerow(row)

        self.previous_action = self.robot.INITIAL_MOTOR_POS
        self.jointPositions = self.robot.INITIAL_MOTOR_POS

        if self.freeze_upper_body:
            high = np.ones([12])
        else:
            high = np.ones([24])
        self.action_space = spaces.Box(-high, high)
        high = np.inf*np.ones([70])
        self.observation_space = spaces.Box(-high, high)

        self.node = self.robot.getSelf()
        self.translation = self.node.getField("translation")
        self.rotation = self.node.getField("rotation")

        self.init_translation = [0, 0.331, 0]
        self.init_rotation = [1, 0, 0, -1.5708]

        self.timeout = 0
        self.fallen = False

    def step(self, action):

        self.previous_action = self.jointPositions
        self.jointPositions = dict()
        defase = 0
        for j in range(len(self.motors)):
            if ( 7 <= j and j <= 14 ) or (20 <= j and j <= 27): #is a Phalanx. No action taken
                self.jointPositions[self.motors[j]] = -1.0
                defase +=1
            elif self.freeze_upper_body and j < 27: #freeze upper body
                self.jointPositions[self.motors[j]] = self.robot.INITIAL_MOTOR_POS[self.motors[j]]
                defase += 1
            else:
                self.jointPositions[self.motors[j]] = action[j-defase]
            self.robot.setJointPositions(self.jointPositions)

        now = self.robot.getTime()
        if self.timeout == 0:
            self.time_vector=[[now,now,now]]
            self.position_vector=self.position_vector
            self.velocity_vector=[self.robot.getRobotVelocity()]
        else:
            self.time_vector.append([now,now,now])
            self.position_vector.append(self.robot.getRobotPosition())
            self.velocity_vector.append(self.robot.getRobotVelocity())

        if self.plot:
            if not self.plot_at_end_of_episode:
                x = np.matrix(self.time_vector[self.timeout])
                y = np.matrix(self.position_vector[self.timeout])

        if self.saveTorquesToCSV:
            with open(self.torques_CSV_path, 'a+') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for i in range(5):
                    self.torques = self.robot.getTorques()
                    now = self.robot.getTime()
                    self.exit = self.robot.step(self.robot.timeStep)
                    row = list(self.torques.values())
                    row.insert(0,now)
                    writer.writerow(row)
        for i in range(5):
            self.exit = self.robot.step(self.robot.timeStep)

        readings = self.robot.getAllReadings()

        ax,ay,az = readings[0:3]
        roll,pitch = readings[5:7]
        gyro = readings[3:5]
        x,y,z = self.robot.getPos()
        motor_values = readings[21:]

        readings.append(self._clock())
        readings.append(0)
        if self._clock() >= 0.5:
            readings[-1]=0

        self.robot.display_write(1,"Time: " + str(self.timeout/10))
        self.robot.display_write(2,"Clock: " + str(self._clock()))
        obs = readings
        reward =self._get_reward(obs, action)
        self.robot.display_write(0,"R: " + str(reward))
        self.timeout += 1

        return obs, reward, self._is_done(obs), dict()

    def _get_reward(self, state, action):
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
        test = np.sum(np.power(np.subtract(list(self.robot.STAND_UP.values()),state[28:68]),2))
        pose_reward = math.exp(-test)

        forward_discount = math.exp(math.pow(x-(2+self.robot.INITIAL_TRANS[0]),2) * -2)

        if math.isnan(roll) and math.isnan(y_position) and math.isnan(pitch):
            f = 0
        else:
            f = -0.1*torque_discount + 0.4*height_discount + 0.7*pose_reward

        self.fallen = self.hasFallen(y, roll, pitch)

        return f

    def _clock(self):
        return self.timeout/10 % 1

    def _is_done(self, state):
        if self.timeout >= 50:
            return True
        else:
            return (self.fallen)

    def hasFallen(self, y, roll, pitch):
        if (y < 0.16 and (abs(roll)>0.2 or abs(pitch)>0.2)) or (abs(roll)>0.5 or abs(pitch)>0.5):
            # print('I fell :(')
            return True
        else:
            return False

    def isStill(self):
        if len(self.velocity_vector) > 3:
            if abs(np.mean(self.velocity_vector[-4:-1])) < 0.001:
                return True
        else:
            return False

    def reset(self, master=False):
        #print("=================================EPISODE COMPLETE===================================")
        if master:
            self.robot.simulationReset()
        else:
            if self.plot and self.plot_at_end_of_episode:
                x = np.matrix(self.time_vector)
                y = np.matrix(self.position_vector)
                start_time = np.min(x)
                end_time = np.max(x)
                delta_time = end_time - start_time
                x = x - start_time
                #self.plotter.ranges(0,max(delta_time,3), -1,1)
                #self.plotter.plotPosition(x,y)
                self.time_vector = [[0,0,0]]
                self.position_vector = [[0,0,0]]
                self.velocity_vector = [[0,0,0,0,0,0]]
                #self.plotter.resetPlot()
            for i in range(50):
                self.exit = self.robot.step(self.robot.timeStep)

            self.node.resetPhysics()
            #self.exit = self.robot.step(self.robot.timeStep)

            self.robot.resetRobotPosition()
            self.robot.simulationResetPhysics()

            self.timeout = 0

        self.previous_action = self.robot.INITIAL_MOTOR_POS
        self.jointPositions = self.robot.INITIAL_MOTOR_POS
        readings = self.robot.getAllReadings()
        ax,ay,az = readings[0:3]
        roll,pitch = readings[5:7]
        gyro = readings[3:5]
        x,y,z = self.robot.getPos()
        motor_values = readings[21:]
        readings.append(self._clock())
        readings.append(0)
        obs = readings
        return obs

    def getExit(self):
        return self.exit
