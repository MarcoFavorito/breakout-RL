"""Abstract class for every RL agent."""
from abc import ABC, abstractmethod

import _pickle as pickle

from breakout_RL.exploration_policies.ExplorationPolicy import ExplorationPolicy
from breakout_RL.misc.Brain import Brain


class RLAgent(ABC):

    def __init__(self, env, exploration_policy:ExplorationPolicy, brain:Brain):
        self.env = env
        self.exploration_policy = exploration_policy
        self.brain = brain

    def act(self, state):
        action = self.exploration_policy.explore()
        if action is None:
            action = self.brain.best_action(state)
        return action


    def observe(self, state, action, reward, state2):
        """Called at each observation"""
        self.brain.observe(state, action, reward, state2)
        self.exploration_policy.update(state, action, reward, state2)

    def replay(self):
        self.brain.learn()

    def reset(self):
        self.brain.reset()
        self.exploration_policy.reset()

    def save(self, filepath):
        with open(filepath + "/exploration_policy.dump", "wb") as fout:
            pickle.dump(self.exploration_policy, fout)
        with open(filepath + "/brain.dump", "wb") as fout:
            pickle.dump(self.brain, fout)

    def load(self, filepath):
        with open(filepath + "/exploration_policy.dump", "rb") as fout:
            self.exploration_policy = pickle.load(fout)
        with open(filepath + "/brain.dump", "rb") as fout:
            self.brain = pickle.load(fout)
