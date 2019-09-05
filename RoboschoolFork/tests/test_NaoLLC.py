import gym, roboschool, roboschoolfork_nao
import numpy as np
env = gym.make('NaoLLC-v1')
env.reset()
INITIAL_MOTOR_POS = {'HeadYaw': 0.0, #[0]
'HeadPitch': 0.0, #[1]
'LHipYawPitch': -0.0, #[2]
'LHipRoll': -0.25, #[3]
'LHipPitch': 0.7, #[4]
'LKneePitch': -0.9, #[5]
'LAnklePitch': -0.0,#[6]
'LAnkleRoll': -0.4,#[7]
'RHipYawPitch': -0.0,#[8]
'RHipRoll': 0.25,#[9]
'RHipPitch': 0.7,#[10]
'RKneePitch': -0.9,#[11]
'RAnklePitch': -0.0,#[12]
'RAnkleRoll': 0.4,#[13]
'LShoulderPitch': 0.8,#[14]
'LShoulderRoll': -0.75,#[15]
'LElbowYaw': -0.8,#[16]
'LElbowRoll': -0.6,#[17]
'LWristYaw': 0.0,#[18]
'LHand':0.0,#[19]
# 'LPhalanx1': 0.0,#[20]
# 'LPhalanx2': 0.0,#[21]
# 'LPhalanx3': 0.0,#[22]
# 'LPhalanx4': 0.0,#[23]
# 'LPhalanx5': 0.0,#[24]
# 'LPhalanx6': 0.0,#[25]
# 'LPhalanx7': 0.0,#[26]
# 'LPhalanx8': 0.0,#[27]
'RShoulderPitch': 0.8,#[28]
'RShoulderRoll': 0.75,#[29]
'RElbowYaw': 0.8,#[30]
'RElbowRoll': 0.6,#[31]
'RWristYaw': -3.43941389813196e-08,#[32]
'RHand':0.0,#[33]
# 'RPhalanx1': 0.0,#[34]
# 'RPhalanx2': 0.0,#[35]
# 'RPhalanx3': 0.0,#[36]
# 'RPhalanx4': 0.0,#[37]
# 'RPhalanx5': 0.0,#[38]
# 'RPhalanx6': 0.0,#[39]
# 'RPhalanx7': 0.0,#[40]
# 'RPhalanx8': 0.0}#[41]
}
# print(np.array(list(INITIAL_MOTOR_POS.values())))
obs, rewards, dones, info = env.step(np.array(list(INITIAL_MOTOR_POS.values())))
for _ in range(5000):
    env.render()
    obs, rewards, dones, info = env.step(np.array(list(INITIAL_MOTOR_POS.values())))
    #obs, rewards, dones, info = env.step(env.action_space.sample())
    #print(rewards)
    # env.step(env.action_space.sample())
    if dones:
        obs = env.reset()
        env.render()
    #     print(dones)
env.close()
