from random import randrange

import keras
from keras.layers import Dense, Input
from keras.optimizers import Adam
import numpy as np
import random

from breakout_RL.agents.RLAgent import RLAgent


class NNAgent(RLAgent):

    ID2ACTION = {0: 2, 1: 3, 2:0}
    ACTION2ID = {2: 0, 3: 1, 0:2}

    """Breakout RL with function approximation"""
    def __init__(self, input_size=1 + 4, hidden_size=256, batch_size=16, epsilon=0.9):
        super().__init__(None, None)

        # paddle_x + ball features + flatten tiles matrix
        self.nactions = 3

        # self.input_size = 1 + 4
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.Q = self.build_model()

        self.alpha = 0.1
        self.gamma = 0.98

        self.epsilon = epsilon
        self.epsilon_start, self.epsilon_end = 1.0, 0.1

        # self.epsilon = 0.0
        # self.epsilon_start, self.epsilon_end = 0.0, 0.0

        self.exploration_steps = 100000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

        self.iteration = 0

        self.history = []


    def build_model(self):

        state_input = Input((self.input_size,), name='states')
        actions_input = Input((self.nactions,), name='mask')

        dense_1 = keras.layers.Dense(self.hidden_size)(state_input)
        # dense_2 = Dense(self.hidden_size)(dense_1)

        output = Dense(self.nactions)(dense_1)

        filtered_output = keras.layers.multiply([output, actions_input])

        model = keras.models.Model(inputs=[state_input, actions_input], outputs=filtered_output)

        # optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        # optimizer = keras.optimizers.SGD(0.0005)
        optimizer = Adam()
        model.compile(optimizer, loss='mse')

        self.Q = model
        self.Q.summary()
        return self.Q

    def act(self, state):
        if random.random() < self.epsilon:
            action_id = randrange(self.nactions)
        else:
            estimated_Q_values = self.Q.predict([state.reshape(1,len(state)), np.ones((1, self.nactions,))])
            best_action = estimated_Q_values.argmax()
            action_id = best_action
            if random.random()<0.001:
                print(estimated_Q_values, best_action, state[:5])

        return self.ID2ACTION[action_id]


    def observe(self, state, action, reward, next_state):
        if len(self.history) > 16:
            self.history.pop(0)
        # print((state, self.ACTION2ID[action], reward, next_state))
        self.history.append((state, self.ACTION2ID[action], reward, next_state))


    def replay(self):
        if self.batch_size > len(self.history):
            return
        batch = random.sample(self.history, self.batch_size)
        states, actions, rewards, next_states = list(map(np.array, list(zip(*batch))))

        one_hot_encoded_actions = np.zeros((actions.size, self.nactions))
        one_hot_encoded_actions[np.arange(actions.size), actions] = 1

        next_Q_values = self.Q.predict([next_states, np.ones(one_hot_encoded_actions.shape)])

        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)

        self.Q.fit([states, one_hot_encoded_actions], one_hot_encoded_actions*Q_values[:, None],
                   epochs=1, batch_size=len(states), verbose=0)


        self.iteration += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step


    def save(self, filepath):
        self.Q.save_weights(filepath)

    def load(self, filepath):
        self.Q.load_weights(filepath)



    # def fit_batch(self):
    #     new_Qs = []
    #     states = []
    #     for (state, action, reward, next_state) in self.history:
    #         old_Q = self.Q[action_id].predict(state.reshape((1,self.input_size)))
    #
    #         # a_prime = self.choose_action(next_state)
    #         # next_Q = self.Q[a_prime].predict(next_state.reshape((1,self.input_size)))
    #         # new_Q = old_Q + self.alpha*(reward + self.gamma * next_Q - old_Q)
    #
    #         new_Q = old_Q + self.alpha*(reward + self.gamma * self._Qa(next_state).max() - old_Q)
    #         new_Qs.append(new_Q)
    #         states.append(state)
    #
    #     new_Qs = np.asarray(new_Qs).reshape(len(new_Qs), 1)
    #     states = np.asarray(states).reshape((len(states), self.input_size))
    #     print(new_Qs[:3], self.iteration)
    #     self.Q[action_id].fit(states, new_Qs)
    #     self.iteration += 1
    #
    #     self.history = {
    #         0:[],
    #         1:[]
    #     }



