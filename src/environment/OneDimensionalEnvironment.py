import numpy as np

from src.environment.AugmentedSimulator import AugmentedSimulator
from src.environment.FullSimulator import FullSimulator
from src.predictors.RandomUniformPredictor import RandomUniformPredictor


class OneDimensionalEnvironment:

    def __init__(self, map_size=10, seed=42):
        np.random.seed(seed)

        self.action_space = np.array([-1, 1])
        self.observation_space = [0, 1]
        self.observation_space_prob = np.array([0.8, 0.2])

        self.map = np.random.choice(self.observation_space, size=map_size, p=self.observation_space_prob)

        self.position = np.random.choice(np.where(self.map == 0)[0])

    def obtain_initial_obs(self) -> np.array:
        start_index = self.position - 1
        end_index = self.position + 2

        # Handle edge cases if start or end indices go out of bounds
        if start_index < 0:
            obs = np.pad(self.map[:end_index], (3 - len(self.map[:end_index]), 0), mode='constant', constant_values=-1)
        elif end_index > len(self.map):
            obs = np.pad(self.map[start_index:], (0, 3 - len(self.map[start_index:])), mode='constant',
                         constant_values=-1)
        else:
            obs = self.map[start_index:end_index]

        return obs

    def create_full_simulator(self) -> tuple[FullSimulator, np.array]:
        return (FullSimulator(observation_space=self.observation_space,
                              action_space=self.action_space,
                              starting_pos=self.position,
                              full_map=self.map),
                self.obtain_initial_obs())

    def create_augmented_simulator(self) -> tuple[AugmentedSimulator, np.array]:
        predictor = RandomUniformPredictor(self.observation_space, self.action_space)

        initial_obs = self.obtain_initial_obs()

        return (AugmentedSimulator(observation_space=self.observation_space,
                                   action_space=self.action_space,
                                   starting_pos=self.position,
                                   map_shape=np.shape(self.map),
                                   initial_obs=initial_obs,
                                   predictor=predictor),
                initial_obs)
