import random

from breakout_RL.exploration_policies.ExplorationPolicy import ExplorationPolicy
import numpy as np

class SemiUniformExploration(ExplorationPolicy):

    def __init__(self, env, n_actions, p_best_start=0.0, p_best_end=0.9, exploration_steps=100000000):
        super().__init__()
        self.env = env
        self.n_actions = n_actions
        self.p_best_start = p_best_start
        self.p_best_end = p_best_end
        self.exploration_steps = exploration_steps

        self.p_best = self.p_best_start
        self.decay_step = (self.p_best_end - self.p_best_start) \
                                  / self.exploration_steps


    def update(self, *args):
        if self.p_best < self.p_best_end:
            self.p_best += self.decay_step

    def explore(self, state, Q_values):
        best_action_id = np.argmax(Q_values)
        distribution = [self.p_best + (1-self.p_best)/self.n_actions if action_id==best_action_id
                        else (1-self.p_best)/self.n_actions
                        for action_id in range(self.n_actions)]
        action_id = np.random.choice(self.n_actions, p=distribution)
        return action_id