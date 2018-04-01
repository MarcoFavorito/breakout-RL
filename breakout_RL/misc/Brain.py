from abc import ABC, abstractmethod


class Brain(ABC):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @abstractmethod
    def best_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    @abstractmethod
    def observe(self, *args):
        raise NotImplementedError

    def reset(self):
        """action performed at the end of each episode"""
        raise NotImplementedError