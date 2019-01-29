import numpy as np
import gym
import random
import time
from IPython.display import clear_output
env = gym.make("FrozenLake-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size,action_space_size))
print(q_table)

num_episodes = 100000
max_steps_per_episode = 1000

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 0.8
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
env.reset()
env.render()
rewards_all_episodes = []
# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    #print('episode: ' + str(episode))
    state = env.reset()
    
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode): 
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
            #print(action)
        else:
            action = env.action_space.sample()
        # Take new action
        new_state, reward, done, info = env.step(action)
        # Update Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        # Set new state
        state = new_state
        rewards_current_episode += reward
        # Add new reward
        if done == True: 
            break
        #env.render()
    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)
    #print('rewards current episode: ' + str(rewards_current_episode))
    #if episode % 10 == 0:
        #print(q_table)
# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("\n\n********Q-table********\n")
print(q_table)
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000