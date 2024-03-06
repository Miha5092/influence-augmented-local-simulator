from abc import ABC, abstractmethod

import numpy as np


class Predictor(ABC):
    def __init__(self, observation_space, action_space, seed=42):

        np.random.seed(seed)
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def predict(self, obs: np.array, action: int):
        pass
