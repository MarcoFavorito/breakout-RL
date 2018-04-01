import random
import numpy as np
from breakout_RL.exploration_policies.ExplorationPolicy import ExplorationPolicy
import math


class CounterBasedPolicyWithDecay(ExplorationPolicy):

    def __init__(self, env, n_actions, epsilon_start=1.0, epsilon_end=0.0, exploration_steps=1000000, alpha=0.01):
        super().__init__()
        self.env = env
        self.n_actions = n_actions
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_steps = exploration_steps
        self.alpha = alpha

        self.epsilon = self.epsilon_start
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

        self.iteration = 0
        self.state2count = {}
        self.state2decay = {}
        self.state2lastupdate = {}
        self.state2action2state = {}


    def update(self, state, action, state2):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        if state not in self.state2action2state:
            self.state2action2state[state] = {}
        self.state2action2state[state][action] = state2
        self.iteration += 1

    def explore(self, state, Q_values):
        if state not in self.state2count:
            self.state2count[state] = 0
            self.state2lastupdate[state] = self.iteration

        self.state2count[state]+= 1


        # if state not in self.state2action2state:
            # next_states = {action:None for action in range(self.n_actions)}
            # self.state2action2state[state] = next_states

        eval_a = [self.alpha * Q_values[action] +
                  self._get_count(state)/self._get_count(self.state2action2state.get(state, {}).get(action, -1))
                  for action in range(self.n_actions)]
        # for s in self.state2count:
        #     self.state2count[s] *= 0.9
        #     if self.state2count[s]<1.0:
        #         self.state2count[s] = 1.0


        return np.argmax(eval_a)

    def _get_count(self, state):
        if state not in self.state2count:
            return 1.0
        self.state2count[state] *= math.pow(0.95, self.iteration - self.state2lastupdate[state])
        self.state2lastupdate[state] = self.iteration
        if self.state2count[state]<1.0:
            self.state2count[state] = 1.0
        return self.state2count[state]