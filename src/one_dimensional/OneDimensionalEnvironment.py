import yaml
import numpy as np
from cpt import CPT


class OneDimensionalEnvironment:
    def __init__(self,
                 file_path: str = 'src/one_dimensional/initial_config.yaml',
                 global_simulator: bool = True,
                 local_simulator: bool = True):
        self.file_path = file_path

        # Decide which simulators to run
        self.global_simulator = global_simulator
        self.local_simulator = local_simulator

        # Load the YAML file
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Extract the values of the nodes
        self.map_values = []

        # Iterate over nodes
        for _, node_data in data.items():
            # Append the value of the node to the node_values list
            self.map_values.append(node_data['initial_value'])

        self.cpt = CPT()

        self.position = 2
        obs = (
            self.map_values[self.position - 1],
            self.map_values[self.position],
            self.map_values[self.position + 1])

        self.global_obs: tuple[int, int, int] = obs
        self.local_obs: tuple[int, int, int] = obs

    def _is_possible_move(self, action: int, obs: tuple[int, int, int]) -> bool:
        pass

    def _get_observation_dependencies(self, action) -> tuple[dict, dict]:
        global_dict = dict()
        local_dict = dict()

        if self.global_simulator and self._is_possible_move(action, self.global_obs):
            if action is 1:
                global_dict[0] = self.global_obs[1]
                global_dict[1] = self.global_obs[2]
                global_dict[2] = self.map_values[self.position + 2]
            else:
                global_dict[0] = self.map_values[self.position - 2]
                global_dict[1] = self.global_obs[0]
                global_dict[2] = self.global_obs[1]

        if self.local_simulator and self._is_possible_move(action, self.local_obs):
            if action is 1:
                local_dict[0] = self.local_obs[1]
                local_dict[1] = self.local_obs[2]
                local_dict[2] = None    # Value needs to be predicted
            else:
                local_dict[0] = None    # Value needs to be predicted
                local_dict[1] = self.local_obs[0]
                local_dict[2] = self.local_obs[1]

        return global_dict, local_dict

    def step(self, action: int) -> tuple[np.array, np.array]:
        global_dict, local_dict = self._get_observation_dependencies(action)

        global_obs = None
        if global_dict:
            global_obs = (self.cpt.evaluate(global_dict.get(0)),
                          self.cpt.evaluate(global_dict.get(1)),
                          self.cpt.evaluate(global_dict.get(2)))

        local_obs = None
        if local_dict:
            pass

        return global_obs, local_obs
