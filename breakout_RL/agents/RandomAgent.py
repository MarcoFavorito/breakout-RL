"""Abstract class for every RL agent."""

from breakout_RL.agents.RLAgent import RLAgent
import random

class RandomAgent(RLAgent):

    def __init__(self, actions):
        super().__init__(None, None)
        self.actions = actions


    def act(self, state):
        return random.choice(self.actions)


    def observe(self, state, action, reward, state2):
        pass

    def replay(self):
        pass

    def save(self, *args):
        pass

    def load(self, *args):
        pass