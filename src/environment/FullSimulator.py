import numpy as np


class FullSimulator:
    def __init__(self, observation_space, action_space, starting_pos, full_map):
        self.observation_space = observation_space

        self.action_space = action_space

        self.position = starting_pos

        self.map = full_map

    def step(self, action: int) -> np.array:
        if self.position + action >= len(self.map):
            self.position = self.position
        elif self.position + action < 0:
            self.position = self.position
        elif self.map[self.position + action] == 1:
            self.position = self.position
        else:
            self.position = self.position + action

        start_index = self.position - 1
        end_index = self.position + 2

        # Handle edge cases if start or end indices go out of bounds
        if start_index < 0:
            obs = np.pad(self.map[:end_index], (3 - len(self.map[:end_index]), 0), mode='constant', constant_values=-1)
        elif end_index > len(self.map):
            obs = np.pad(self.map[start_index:], (0, 3 - len(self.map[start_index:])), mode='constant', constant_values=-1)
        else:
            obs = self.map[start_index:end_index]

        return obs
