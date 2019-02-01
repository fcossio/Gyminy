import gym
import gym_gridworld
import time
import random
import numpy as np

class QLearningAgent():
    """
      Q-Learning Agent

      Instance variables
        - self.epsilon (exploration prob)
        - self.lr (learning rate)
        - self.gamma (discount rate)
    """
    def __init__(self, actions):
        self.epsilon = 0.2      #Exploration
        self.e_decay = 0.5
        self.lr = 0.5           #Learning rate
        self.gamma = 0.8        #Discount rate
        self.actions = actions
        self.qvalues = dict()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Returns 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) in self.qvalues:
            return self.qvalues[(state, action)]

        return 0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """
        qValues = dict()
        if len(self.actions) > 0:
            for a in self.actions:
                qValues[a] = self.getQValue(state, a)
            return qValues[max(qValues, key=qValues.get)]
        else:
            return 0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
        """
        max = float('-inf')
        list = []
        qValues = dict()
        if(len(self.actions) > 0):
            for a in self.actions:
                qValues[a] = self.getQValue(state, a)
                if(qValues[a] > max):
                    max = qValues[a]
            for a in self.actions:
                qValues[a] = self.getQValue(state, a)
                if(qValues[a] == max):
                    list.append(a)
            return random.choice(list)
        else:
            return 0

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
        """
        legalActions = self.actions
        action = None

        if len(legalActions) > 0:
            if np.random.sample() > self.epsilon:
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          state = action => nextState and reward transition.

          Q-Value updates happen here
        """
        qValues = dict()
        for a in self.actions:
            qValues[a] = self.getQValue(nextState, a)
        sample = reward + self.gamma*qValues[max(qValues, key=qValues.get)]
        if (state, action) in self.qvalues:
            self.qvalues[(state, action)] = (1-self.lr)*self.qvalues[(state, action)] + self.lr*sample
        else:
            self.qvalues[(state, action)] = 0

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

env = gym.make('nxt-v0')

agent = QLearningAgent(env.actions)

for i in range(1000):    #100 episodes
    state = env.reset()
    done = False
    while done == False:
        time.sleep(0.05)
        env.render()
        prevState = state
        action = agent.getAction(state)
        state, reward, done, info = env.step(action)
        agent.update(prevState, action, state, reward)
    time.sleep(0.2)
    print("")
    print("Terminal state reached: ", state)
    print("Episode ", i, " Complete! Reward: ", reward)
    print("======================================")
