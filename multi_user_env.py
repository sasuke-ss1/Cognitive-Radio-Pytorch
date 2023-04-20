import os
import random
import sys
import numpy as np


TIME_SLOTS = 1
NUM_CHANNELS = 3
NUM_USER = 5
ATTEMPT_PROB = 0.6
GAMMA =0.99

class env():
    def __init__(self, num_users: int, num_channels: int, reward: int,attempt_prob: float):
        self.num_users = num_users
        self.num_channels = num_channels
        self.attempt_prob = attempt_prob
        self.reward = reward

        self.action_space = np.arange(self.num_channels+1)
        self.users_action = np.zeros((self.num_users), dtype=np.int32)
        self.users_observation = np.zeros([self.num_users], dtype=np.int32)

    def reset(self):
        pass

    def sample(self):
        return np.random.choice(self.action_space, self.num_users)

            
    def step(self, action: np.ndarray):
        assert action.size == self.num_users, "F bhai"

        channel_alloc_freq = np.zeros([self.num_channels+1], np.int32) # 0 for no channel access

        obs = []
        reward = np.zeros([self.num_users])
        j = 0

        for act in action:
            prob = random.uniform(0,1)
            if prob <= self.attempt_prob:
                self.users_action[j] = act

                channel_alloc_freq[act] += 1
            
            j+= 1

        for i in range(1, len(channel_alloc_freq)):
            if channel_alloc_freq[i] > 1:
                channel_alloc_freq[i] = 0

        for i in range(len(action)):
            
            self.users_observation[i] = channel_alloc_freq[self.users_action[i]]
            if self.users_action[i] == 0:
                self.users_observation[i] = 0

            if self.users_observation[i] == 1:
                reward[i] = self.reward

            obs.append((self.users_observation[i], reward[i]))

        residual_channel_capacity = 1 - channel_alloc_freq[1:]
        obs.append(residual_channel_capacity)

        return obs
    






