import numpy as np

from src.predictors.Predictor import Predictor


class RandomUniformPredictor(Predictor):
    def __init__(self, observation_space, action_space, seed=42):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         seed=seed)
        np.random.seed(seed)

    def predict(self, obs: np.array, action: int) -> int:
        return np.random.choice(self.observation_space)
