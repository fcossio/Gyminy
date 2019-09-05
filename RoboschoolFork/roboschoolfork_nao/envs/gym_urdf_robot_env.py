import gym, gym.spaces, gym.utils, gym.utils.seeding
from roboschool.scene_abstract import cpp_household
import numpy as np
import os

class RoboschoolUrdfEnv(gym.Env):
    """
    Base class for URDF robot actor in a Scene.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    VIDEO_W = 600  # for video showing the robot, not for camera ON the robot
    VIDEO_H = 400

    def __init__(self, model_urdf, robot_name, action_dim, obs_dim, fixed_base, self_collision):
        self.scene = None

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self.seed()

        self.model_urdf = model_urdf
        self.fixed_base = fixed_base
        self.self_collision = self_collision
        self.robot_name = robot_name
        self.initial_joint_position = REL_JOINT_POSITIONS()
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.scene is None:
            self.scene = self.create_single_player_scene()
        if not self.scene.multiplayer:
            self.scene.episode_restart()

        pose = cpp_household.Pose()
        #import time
        #t1 = time.time()
        #print("fixed base: ", self.fixed_base)
        self.urdf = self.scene.cpp_world.load_urdf(
            os.path.join(os.path.dirname(__file__), "models_robot", self.model_urdf),
            pose,
            self.fixed_base,
            self.self_collision)
        #t2 = time.time()
        #print("URDF load %0.2fms" % ((t2-t1)*1000))

        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        r = self.urdf
        self.cpp_robot = r
        if dump: print("ROBOT '%s'" % r.root_part.name)
        if r.root_part.name==self.robot_name:
            self.robot_body = r.root_part
        for part in r.parts:
            if dump: print("\tPART '%s'" % part.name)
            self.parts[part.name] = part
            if part.name==self.robot_name:
                self.robot_body = part
        for j in r.joints:

            if j.name[1:7] != "Finger" and j.name[1:6] != "Thumb":
                if dump: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                if j.name[:6]=="ignore":
                    j.set_motor_torque(0)
                    continue
                j.power_coef, j.max_velocity = j.limits()[2:4]
                self.ordered_joints.append(j)
                self.jdict[j.name] = j
            #else:
                #print(j.name, "was ignored")

        #print('joints length:', len(self.ordered_joints))
        self.robot_specific_reset()
        self.cpp_robot.query_position()
        s = self.calc_state()    # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()
        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        return s

    def render(self, mode='human'):
        if mode=="human":
            self.scene.human_render_detected = True
            return self.scene.cpp_world.test_window()
        elif mode=="rgb_array":
            self.camera_adjust()
            rgb, _, _, _, _ = self.camera.render(True, True, False) # render_depth, render_labeling, print_timing)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
            return rendered_rgb
        else:
            assert(0)

    def calc_potential(self):
        return 0
    def HUD(self, s, a, done):
        active = self.scene.actor_is_active(self)
        if active and self.done<=2:
            self.scene.cpp_world.test_window_history_advance()
            self.scene.cpp_world.test_window_observations(s.tolist())
            self.scene.cpp_world.test_window_actions(a.tolist())
            self.scene.cpp_world.test_window_rewards(self.rewards)
        if self.done<=1: # Only post score on first time done flag is seen, keep this score if user continues to use env
            s = "%04i %07.1f %s" % (self.frame, self.reward, "DONE" if self.done else "")
            if active:
                self.scene.cpp_world.test_window_score(s)
            #self.camera.test_window_score(s)  # will appear on video ("rgb_array"), but not on cameras istalled on the robot (because that would be different camera)
def REL_JOINT_POSITIONS():
    '''edit to make robot appear in a different position'''
    return(
        {'HeadYaw': 0.0, #[0]
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
    )
