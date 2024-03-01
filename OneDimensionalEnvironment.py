import numpy as np


class OneDimensionalEnvironment:
    """
    An environment
    """

    def __init__(self, seed=42, augmented=False):
        np.random.seed(seed)
        self.augmented = augmented

        self.action_space = np.array([0, 1])
        self.observation_space = [0, 1]
        self.observation_space_prob = np.array([0.8, 0.2])

        self.position = np.random.choice(1, 1)[0]

        if not augmented:
            self.map = np.random.choice(a=self.observation_space, size=10, p=self.observation_space_prob)

    def action_space(self) -> np.array:
        return self.action_space

    def observation_space(self) -> np.array:
        return self.observation_space

    def step(self, action: int) -> np.array:

        print(self.position - 1, self.position + 1)

        if not self.augmented:
            return self.map[self.position - 1: self.position + 1]
        else:
            return self.map[self.position - 1: self.position + 1]
