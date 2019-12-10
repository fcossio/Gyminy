import sys
n_cpu = int(sys.argv[1])
total_train_timesteps = int(sys.argv[2])
model_path = sys.argv[3]
import gym, roboschool, roboschoolfork_nao
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from time import time


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

# multiprocess environment
def main():
    env = SubprocVecEnv([make_env('NaoLLC-v1', i) for i in range(n_cpu)])
    initial_timestep = 0

    # checkpoint_timesteps = 10000
    start_time = time()
    model = PPO2.load(model_path, env = env)
    model.nminibatches = 8
    model.cliprange = 0.2
    model.learning_rate = 0.00050
    model.tensorboard_log = "./exponential_rewards"
    model.gamma = 0.965
    try:
        model.learn(total_timesteps=total_train_timesteps)
    except:
        print("training aborted")
    model.save(model_path.split(".")[0] + ".more.pkl")
    print("Saved")
    end_time = time()

    print("Elapsed time training", end_time - start_time)
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
