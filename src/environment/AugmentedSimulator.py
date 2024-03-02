import numpy as np

from src.predictors.Predictor import Predictor


class AugmentedSimulator:
    def __init__(self, observation_space, action_space, starting_pos, map_shape, initial_obs, predictor: Predictor):
        self.observation_space = observation_space

        self.action_space = action_space

        self.position = starting_pos
        self.map_shape = map_shape
        self.obs = initial_obs

        self.predictor = predictor

    def step(self, action: int) -> np.array:

        # If the action would make the agent exit the world do not permit it
        if self.position + action < 0:
            return self.obs
        elif self.position + action >= self.map_shape[0]:
            return self.obs

        # Perform the action and predict
        self.position = self.position + action
        predicted_value = self.predictor.predict(self.obs, action)
        print(f"Predicted value: {predicted_value}")

        # Update the observation
        if action == -1:
            obs = np.insert(self.obs, 0, predicted_value)
            obs = obs[:3]  # Resize to be of size 3
        elif action == 1:
            obs = np.insert(self.obs, np.shape(self.obs)[0], predicted_value)
            obs = obs[-3:]  # Resize to be of size 3
        else:
            raise ValueError(f"Invalid action: {action}, must be {self.action_space}")

        self.obs = obs
        return self.obs
