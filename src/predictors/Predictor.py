import numpy as np


class Predictor:
    def __init__(self, observation_space, action_space, seed=42):
        np.random.seed(seed)

        self.observation_space = observation_space
        self.observation_space = action_space

    def predict(self, obs: np.array, action: int):
        pass
