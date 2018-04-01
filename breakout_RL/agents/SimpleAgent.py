"""Abstract class for every RL agent."""

from breakout_RL.agents.RLAgent import RLAgent
import random

class SimpleAgent(RLAgent):

    def __init__(self):
        super().__init__(None, None)


    def act(self, state):
        """state vector representation"""

        ball_x = state[2]
        paddle_x = state[0]

        if paddle_x < ball_x:
            return 2
        else:
            return 3


    def observe(self, state, action, reward, state2):
        pass

    def replay(self):
        pass

    def save(self, *args):
        pass

    def load(self, *args):
        pass