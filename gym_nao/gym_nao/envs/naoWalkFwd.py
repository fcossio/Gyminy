import pybullet as p
import time
import numpy as np
import gym

from gym import error, spaces, utils
from gym.utils import seeding

class NaoForwardWalkEnv(gym.Env):
    def __init__(self):

        startPos = [0,0,.35]
        self.nao = p.loadURDF("/home/nebugate/Nebugate/Gyminy/gym_nao/gym_nao/nao_description/urdf/naoV50_generated_urdf/nao.urdf",startPos, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT )
        self.important_joints = [1, 2, 13, 14, 15, 16, 17, 
                                18, 26, 27, 28, 29, 30, 31,
                                39, 40, 41, 42, 43, 44, 48,
                                49, 50, 51, 52, 53, 54, 55,
                                56, 57, 58, 59, 60, 61, 65,
                                66, 67, 68, 69, 70, 71, 72]

        self.uninportant_joints = [1, 2, 39, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 54, 55,
                                    56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72]
        high = np.ones([len(self.important_joints)])
        self.action_space = spaces.Box(-high, high)
        high = np.inf*np.ones([90])
        self.observation_space = spaces.Box(-high, high)

        for i in range (p.getNumJoints(self.nao)):
            info = p.getJointInfo(self.nao,i)
            print(info[0], info[1])
            p.setJointMotorControl2(self.nao,i,p.POSITION_CONTROL,targetPosition=0,force=10)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        self.timeStep=1./240.
        p.setTimeStep(self.timeStep)

    def step(self, action):
        p.setJointMotorControlArray(self.nao,self.important_joints,p.POSITION_CONTROL,targetPositions=action,forces=[50]*len(self.important_joints))

        p.stepSimulation()

        obs = []
        base_pos = p.getLinkState(self.nao, 0)[0]
        obs.append(base_pos[0])
        obs.append(base_pos[1])
        obs.append(base_pos[2])
        head_pos = p.getLinkState(self.nao, 3)[0]
        obs.append(head_pos[0])
        obs.append(head_pos[1])
        obs.append(head_pos[2])

        for i in self.important_joints:
            info = p.getJointState(self.nao, i)
            #print(info)
            obs.append(info[0])
            obs.append(info[1])

        return obs, self._get_reward(obs), self._is_done(obs), None

    def reset(self):
        pi=3.1415
        shoulderPitch = pi/2
        shoulderRoll = np.random.rand()
        p.setJointMotorControl2(self.nao,56,p.POSITION_CONTROL,targetPosition=shoulderPitch,force=1000)
        p.setJointMotorControl2(self.nao,39,p.POSITION_CONTROL,targetPosition=shoulderPitch,force=1000)
        p.setJointMotorControl2(self.nao,57,p.POSITION_CONTROL,targetPosition=-shoulderRoll,force=1000)
        p.setJointMotorControl2(self.nao,40,p.POSITION_CONTROL,targetPosition=shoulderRoll,force=1000)
        p.setGravity(0,0,-10)

        obs = []
        base_pos = p.getLinkState(self.nao, 0)[0]
        obs.append(base_pos[0])
        obs.append(base_pos[1])
        obs.append(base_pos[2])
        head_pos = p.getLinkState(self.nao, 3)[0]
        obs.append(head_pos[0])
        obs.append(head_pos[1])
        obs.append(head_pos[2])

        for i in self.important_joints:
            info = p.getJointState(self.nao, i)
            obs.append(info[0])
            obs.append(info[1])
        return obs

    def render(self):
        return None

    def _get_reward(self, state):
        reward = [0,0,0,0]
        reward[0] = None
        reward[1] = None
        reward[2] = None
        reward[3] = None
        if state[5] < 0.4:
            return -100

        xPos = state[0]
        yPos = state[1]

        headZPos = state[5]

        f = xPos*2 - yPos + headZPos

        return 0

    def _is_done(self, obs):
        base_position,base_orientation = p.getBasePositionAndOrientation(self.nao)
        print('base position ' + str(base_position))
        print('base orientation' + str(base_orientation))
        if obs[5] < 0.4:
            return True
        else:
            if obs[1] >= 5:
                return True
            return False
