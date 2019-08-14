from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import numpy as np

class RoboschoolAtlasForwardWalk(RoboschoolForwardWalker, RoboschoolUrdfEnv):
    '''
    3-D humanoid robot similar to Boston Dynamics ATLAS robot.
    The task is to make the robot run as fast as possible
    '''
    random_yaw = False
    foot_list = ["r_ankle", "l_ankle"]

    def __init__(self):
        RoboschoolForwardWalker.__init__(self, power=0.3)
        RoboschoolUrdfEnv.__init__(self,
            "nao_description/urdf/naoV50_generated_urdf/nao.urdf",
            "torso",
            action_dim=26, obs_dim=62,
            fixed_base=False,
            self_collision=True)

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.033/16, frame_skip=16)   # 8 instead of 4 here

    def alive_bonus(self, z, pitch):
        # This is debug code to fix unwanted self-collisions:
        #for part in self.parts.values():
        #    contact_names = set(x.name for x in part.contact_list())
        #    if contact_names:
        #        print("CONTACT OF '%s' WITH '%s'" % (part.name, ",".join(contact_names)) )

        x,y,z = self.head.pose().xyz()
        # Failure mode: robot doesn't bend knees, tries to walk using hips.
        # We fix that by a bit of reward engineering.
        knees = np.array([j.current_relative_position() for j in [self.jdict["LKneePitch"], self.jdict["RKneePitch"]]], dtype=np.float32).flatten()
        knees_at_limit = np.count_nonzero(np.abs(knees[0::2]) > 0.99)
        return +3-knees_at_limit if z > 0.35 else -1 #Editado por Fer original: +6 y z>1.3

    def robot_specific_reset(self):
        RoboschoolForwardWalker.robot_specific_reset(self)
        self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi)
        self.head = self.parts["Head"]

    random_yaw = False

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        cpose = cpp_household.Pose()
        if not self.random_yaw:
            yaw = yaw_center
        else:
            yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)

        cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.40)
        cpose.set_rpy(0, 0, yaw)  # just face random direction, but stay straight otherwise
        self.cpp_robot.set_pose_and_speed(cpose, 0,0,0)
        self.initial_z = 0.35
