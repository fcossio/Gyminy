import gym
import numpy as np
import random
import time, thread

from gym import error, spaces, utils
from gym.utils import seeding

from nxt.sensor import *
from nxt.motor import *
import nxt.bluesock

class NXTDriver():

    def __init__(self, MAC='00:16:53:06:37:7B'):
        self.brick_MAC = MAC

    def bumper_car(self, m_left, m_right, b_left, b_right, ultrasonic, compass):
        car = {}

        print 'connecting to ', self.brick_MAC, '...'
        b = nxt.bluesock.BlueSock(self.brick_MAC).connect()
        print (b)

        #returns a dict with the parts of the bumper car
        car['left_wheel'] = Motor(b, m_left)
        car['right_wheel'] = Motor(b, m_right)
        car['left_bumper'] = Touch(b, b_left)
        car['right_bumper'] = Touch(b, b_right)
        car['ultrasonic'] = Ultrasonic(b, ultrasonic)
        car['compass']= Ultrasonic(b,compass)
        return car

    def turnmotor(self, m1, power1, m2, power2):
        a=0 #(np.random.rand()-0.5) * 20
        #a = m.get_tacho().tacho_count
        m1.run(power1)
        m2.run(power2)
        time.sleep(0.1)
        m1.idle()
        m2.idle()
        #b = m.get_tacho().tacho_count
        #print(b)
        return True


class NXTSimpleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ultrasonic=255, direction=180, terminal_reward=100, step_reward=1, target_direction=50):
        self.terminal_reward = terminal_reward
        self.done = False
        self.past_state = None

        self.num_states = (2^2)*(256)*(180)         #*(701^2)
        self.target_direction = target_direction

        self.action_space = spaces.Box(low=np.array([-127, -127]), high=np.array([128, 128]), dtype='int16')
        # self.observation_space = spaces.Dict({
        #     'bumpers':spaces.MultiBinary(2),
        #     'ultrasonic':spaces.Discrete(256),
        #     'direction':spaces.Discrete(180)
        #     #'distance':spaces.Box(low=np.array([-350, -350]), high=np.array([350, 350]), dtype='int16')
        # })

        self.state = [0,0,0,0]

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, 1, 255, 179]),
            dtype='int8'
            #'distance':spaces.Box(low=np.array([-350, -350]), high=np.array([350, 350]), dtype='int16')
        )

        self.driver = NXTDriver()

        self.car=self.driver.bumper_car(PORT_C, PORT_B, PORT_4, PORT_3, PORT_1, PORT_2)
        self.encoders_old = [self.car['left_wheel'].get_tacho().tacho_count,self.car['right_wheel'].get_tacho().tacho_count]


    def step(self, action):
        assert self.action_space.contains(action)

        self.past_state = self.state

        #If the agent reaches a terminal state
        if self.state[0]==1 or self.state[1]==1:          #antes 1 in state[]'bumpers'
            self.done = True
            return self.state, self._get_reward(), self.done, None

        thread.start_new_thread(self.driver.turnmotor, (self.car['left_wheel'], action[0], self.car['right_wheel'], action[1]))

        is_moving = True
        while is_moving:
            encoders_moving = [self.car['left_wheel'].get_tacho().tacho_count,self.car['right_wheel'].get_tacho().tacho_count]
            time.sleep(0.001)
            encoders_moving = np.subtract([self.car['left_wheel'].get_tacho().tacho_count,self.car['right_wheel'].get_tacho().tacho_count], encoders_moving)
            #print encoders_moving
            if encoders_moving[0]==0 and encoders_moving[1]==0: is_moving = False

        obstacle = self.car['ultrasonic'].get_sample()
        direction = self.car['compass'].get_sample()
        encoders_new = [self.car['left_wheel'].get_tacho().tacho_count,self.car['right_wheel'].get_tacho().tacho_count]
        encoders_dif = np.subtract(encoders_new , self.encoders_old)
        bumpers = [self.car['left_bumper'].get_sample(), self.car['right_bumper'].get_sample()]
        #print 'bumpers ', bumpers , 'ultrasonic ', distance, 'direction', direction, 'encoders', encoders_dif

        # self.state = {
        #     'bumpers' : bumpers,
        #     'ultrasonic' : obstacle,
        #     'direction' : direction
        # }

        self.state[0] = bumpers[0]
        self.state[1] = bumpers[1]
        self.state[2] = obstacle
        self.state[3] = direction

        reward = self._get_reward(encoders_dif, self.state)
        return self.state, reward, self.done, None


    def _get_reward(self, encoders_dif, new_state):

        if self.done:
            return -500

        distance = np.mean(encoders_dif)
        deviation = 90 - abs(abs(new_state[3]-self.target_direction) - 90)      #antes new_state['direction']
        difference = abs(encoders_dif[0] - encoders_dif[1]) + 1

        print "Distance moved: ", distance

        reward = distance*(1/difference) - 2*deviation

        return reward

    def reset(self):
        # self.state = {
        #     'bumpers' : [0, 0],
        #     'ultrasonic' : 255,
        #     'direction' : 0
        # }
        self.state[0] = 0
        self.state[1] = 0
        self.state[2] = self.car['ultrasonic'].get_sample()
        self.state[3] = self.car['compass'].get_sample()
        self.done = False
        return self.state
