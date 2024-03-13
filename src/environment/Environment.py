import numpy as np


def valid_shape(board_shape: tuple) -> bool:
    """
    Check if board shape is valid

    :param board_shape: tuple with shape of board
    :return: true if the desired board shape allows for moving space and border walls, false otherwise
    """
    if len(board_shape) != 2:
        return False

    for dimension in board_shape:
        if dimension <= 2:
            return False

    return True


class Environment:

    def __init__(self,
                 board_configuration: np.array = None,
                 board_shape: tuple[int, int] = (7, 7),
                 global_simulator: bool = True,
                 local_simulator: bool = True,
                 seed: int = 42,
                 verbose: bool = False) -> None:
        np.random.seed(seed)
        self.verbose = verbose

        # Zero represents free space
        # One represents blocked spaces
        self.observation_space = [0, 1]
        self.observation_prob = [0.8, 0.2]

        # Up, right, down, left
        self.action_space = [0, 1, 2, 3]

        if not valid_shape(board_shape):
            raise ValueError(f"{board_shape} is invalid shape for environment. Must be 2D with values larger than 2.")

        self.board_shape = board_shape
        self.board_configuration = board_configuration

        self.global_simulator = global_simulator
        self.local_simulator = local_simulator

        self.position = None
        self.global_observation = None
        self.local_observation = None

    def _create_board(self) -> None:
        """
        Private method used to create an initial configuration of the game board based on the input parameters.

        :return: None
        """
        if self.board_configuration is None:

            if self.verbose:
                print(f"Generating random board of shape {self.board_shape}")

            self.board_configuration = np.random.choice(
                self.observation_space,
                size=self.board_shape,
                p=self.observation_prob)

            # Create the walls of the world
            self.board_configuration[0, :] = 1
            self.board_configuration[self.board_shape[0] - 1, :] = 1

            self.board_configuration[:, 0] = 1
            self.board_configuration[:, self.board_shape[0] - 1] = 1

            if self.verbose:
                print(self.board_configuration)

        else:
            self.board_configuration = self.board_configuration

        # Find valid starting positions
        available_placements = []
        for row in range(self.board_shape[0]):
            for col in range(self.board_shape[1]):
                if self.board_configuration[row][col]:
                    available_placements.append((row, col))

        if available_placements:
            self.position = available_placements[np.random.choice(len(available_placements))]
        else:
            raise ValueError("Could not find valid starting position for given starting conditions.")

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Private method to return the observations of the environment for the current game state for the global
        and local simulators.

        :return: tuple containing the observations for the global and local simulators
        """
        pass

    def possible_actions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Method to return the possible actions for the current game state for the global and local simulators.

        :return: tuple containing the possible actions for the global and local simulators
        """
        pass

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Method used to reset the environment.

        :return: tuple containing the initial observations for the global and local simulators
        """
        self._create_board()

        return self._observe()

    def __str__(self):
        pass

    def __repr__(self):
        pass