import gym
import gym_nxt
import time
from DQNAgent import DQNAgent
import numpy as np

env = gym.make('nxt-v0')

#print "Ejemplo de accion: ", env.action_space.sample()
state_size = env.observation_space.shape[0]
action_size = 256

motor1 = DQNAgent(state_size, action_size)
motor2 = DQNAgent(state_size, action_size)

#print "Ejemplo de accion: ", env.action_space.sample()
#print "Ejemplo de estado: ", env.observation_space.sample()['bumpers']

for i in range(1000):

    state = env.reset()
    score = 0
    done = False

    while done == False:

        state = np.reshape(state, [1, state_size])
        action1 = motor1.act(state) - 128
        action2 = motor2.act(state) - 128

        action = np.array([action1, action2])

        #action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)

        new_state = np.reshape(new_state, [1, state_size])

        # save the sample <s, a, r, s'> to the replay memory
        #motor1.append_sample(state, action1, reward, new_state, done)
        #motor2.append_sample(state, action2, reward, new_state, done)
        # every time step do the training
        #motor1.train_model()
        #motor2.train_model()
        score += reward
        state = new_state

        print ""
        print "Action: ", action
        print "Current State: ", state
        print "Reward: ", reward
        print "Done: ", done
        print "=================================="
    print ""
    print "Episode ", i, " complete! Score: ", score
