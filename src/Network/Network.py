import numpy as np


class Node:

    def __init__(self,
                 init_value: int,
                 cpt,
                 name: str = 'Default Name') -> None:
        self.value = init_value
        self.neighbours = []
        self.parents = []
        self.cpt = cpt

        self.name = name

    def add_neighbours(self, neighbours: list['Node']):
        self.neighbours = neighbours

    def add_parents(self, parents: list['Node']):
        self.parents = parents

    def next_value(self, useful_values: dict['Node', int]) -> int:
        influence_values = [useful_values[parent] for parent in self.parents]

        return self.cpt(influence_values)

    def next_node(self, action: int) -> 'Node':
        return self.neighbours[action]

    def __str__(self):
        parent_names = [parent.name for parent in self.parents]
        return f'{self.name}(val: {self.value}, parents: {parent_names})'

    def __repr__(self):
        return self.name


class Network:

    def __init__(self,
                 nodes: list[Node],
                 obs: Node,
                 local_simulation: bool = False) -> None:
        self.nodes = nodes
        self.obs = obs
        self.local_simulation = local_simulation

    def step(self, action: int) -> tuple[int, int]:
        if self.local_simulation:
            return self.local_step(action)
        else:
            return self.global_step(action)

    def global_step(self, action: int) -> tuple[int, int]:
        self.obs = self.obs.next_node(action)

        old_values = dict()
        for node in self.nodes:
            old_values[node] = node.value

        for node in self.nodes:
            node.value = node.next_value(old_values)

        reward = 1 if self.obs.value == 1 else 0
        return self.obs.value, reward

    def local_step(self, action: int) -> tuple[int, int]:
        useful_values = dict()              # A dict with all the nodes and the values which are going to be used
        useful_values[self.obs] = self.obs.value

        known_nodes = {self.obs}            # The nodes whose values we know (currently observation space is size 1)

        self.obs = self.obs.next_node(action)               # Perform the action and observed location

        to_predict = set(self.obs.parents) - known_nodes    # Nodes that influence the local region that need predicted

        for node in to_predict:
            useful_values[node] = np.random.choice([0, 1])

        self.obs.value = self.obs.next_value(useful_values)

        reward = 1 if self.obs.value == 1 else 0
        return self.obs.value, 1

    def update_network(self):
        new_values = dict()
        for node in self.nodes:
            new_values[node] = node.next_value()

        for (node, new_value) in new_values.items():
            node.value = new_value

    def __str__(self):
        node_strs = [str(node) for node in self.nodes]
        return f'Network( {", ".join(node_strs)} )'

    def __repr__(self):
        vals = [node.value for node in self.nodes]
        return repr(self.obs) + ' ' + str(vals)
