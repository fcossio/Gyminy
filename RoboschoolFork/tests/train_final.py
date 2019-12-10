import sys
n_cpu = int(sys.argv[1])
total_train_timesteps = int(sys.argv[2])
multiplier = int(sys.argv[3])
import gym, roboschool, roboschoolfork_nao
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from time import time
import imageio
from tqdm import tqdm


def make_env(env_id, rank, seed=0):
  """
  Utility function for multiprocessed env.

  :param env_id: (str) the environment ID
  :param num_env: (int) the number of environments you wish to have in subprocesses
  :param seed: (int) the inital seed for RNG
  :param rank: (int) index of the subprocess
  """
  def _init():
      env = gym.make(env_id)
      env.seed(seed + rank)
      return env
  return _init

def lr_func():
    def lr(fraction):
        learning_rate = 0.0002*fraction**4+0.00015
        return learning_rate
    return lr
def clip_func():
    def clp(fraction):
        clip = 0.05*fraction**4 + 0.2
        return clip
    return clp
# multiprocess environment

def main():
    env = SubprocVecEnv([make_env('NaoLLC-v1', i) for i in range(n_cpu)])
    # initial_timestep = 0
    # activation_function = tf.nn.relu
    # net_arch = [512,256,64]
    # # checkpoint_timesteps = 10000
    # start_time = time()
    # policy_kwargs = dict(
    #     #act_fun=activation_function,
    #     net_arch = net_arch)
    # model = PPO2(MlpPolicy, env, verbose=2,
    #     tensorboard_log="./train_final_log",
    #     learning_rate = lr_func(),
    #     cliprange = clip_func(),
    #     nminibatches = n_cpu,
    #     n_steps = 128,
    #     gamma = 0.97,
    #     policy_kwargs=policy_kwargs)
    # # model = PPO2.load("ppo2_NaoForwardWalk11jul.pkl", env = env, tensorboard_log="./ppo2_NaoForwardWalk")
    # # try:
    # model.learn(total_timesteps=total_train_timesteps)
    # # except:
    # #     print("training aborted")
    # model.save("0.pkl")
    # print("Saved 0.pkl")
    # end_time = time()

    for repetition in range(multiplier):
        # import gym, roboschool, roboschoolfork_nao
        #env = SubprocVecEnv([make_env('NaoLLC-v1', i) for i in range(n_cpu)])
        print(str(repetition) + ".pkl")
        model = PPO2.load(str(repetition) + ".pkl", env = env)
        print("LOADED!!!!")
        model.nminibatches = 8
        model.learning_rate = lr_func()
        model.cliprange = clip_func()
        model.gamma = 0.96
        try:
            model.learn(total_timesteps=total_train_timesteps)
        except:
            print("training aborted")
        model.save(str(repetition+1)+".pkl")
        print("Saved "+str(repetition+1)+".pkl")
        env2 = gym.make('NaoLLC-v1')
        # env = wrappers.Monitor(env, './recording/' + str(time()) + '/')
        # Enjoy trained agent
        obs = env2.reset()
        episodes = 0
        with imageio.get_writer(str(repetition+1)+'.gif', mode='I', fps=30) as writer:
            for i in tqdm(range(300)):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env2.step(action)
                if dones:
                    obs = env2.reset()
                    episodes += 1
                writer.append_data(env2.render(mode='rgb_array'))
        del model
    # del model # remove to demonstrate saving and loading
    #
    # model = PPO2.load("ppo2_NaoForwardWalk12jul.pkl")
    # env = gym.make('RoboschoolNaoForwardWalk-v1')
    # # Enjoy trained agent
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     if dones:
    #         obs = env.reset()
    #     env.render()

if __name__ == '__main__':
    main()
