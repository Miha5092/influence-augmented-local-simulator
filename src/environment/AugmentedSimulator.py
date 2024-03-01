import numpy as np


class AugmentedSimulator:
    def __init__(self, observation_space, action_space, starting_pos, initial_obs, predictor):
        self.observation_space = observation_space

        self.action_space = action_space

        self.position = starting_pos
        self.obs = initial_obs

        self.predictor = predictor

    def step(self, action: int) -> np.array:
        return self.obs
