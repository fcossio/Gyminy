import roboschool
import gym
from gym.spaces import prng
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines.common import set_global_seeds

    
def main():


    def make_env(env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
        

    n_cpu = 1

    envId = 'RoboschoolHumanoid-v1'
    record_video = False
    video_folder = '.'
    render_time_steps = 20000

    new_model = True
    learning_time_steps = 100000

    env = gym.make(envId)
    #env.camera_dramatic()
    env = DummyVecEnv([lambda: env])

    if record_video:
        env = VecVideoRecorder(env, video_folder,
                        record_video_trigger=lambda x: x == 0, video_length=render_time_steps,
                        name_prefix="PPO2-{}".format(envId))
        model = PPO2.load('Humanoid.pkl', env)
        obs = env.reset()
        
        for _ in range(render_time_steps + 1):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
        env.close()

    
    if new_model:
        model = PPO2(MlpPolicy, env, verbose = 1,tensorboard_log = 'log2')
    else:
        model = PPO2.load('Humanoid.pkl', env)

    

    #model.learn(total_timesteps = learning_time_steps, log_interval=100)
    #model.save('RoboschoolHumanoid-v2.pkl')x



    obs = env.reset()
    print('Learning is over, rendering results for ' + str(render_time_steps))
    for i in range(render_time_steps+1):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    main()
