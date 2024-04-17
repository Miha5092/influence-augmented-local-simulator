import numpy as np

from src.Network.Network import Node, Network


def left_wave_cpl(inputs: list[int]) -> int:
    if inputs[0] == 1:
        return 0
    else:
        return np.random.choice([0, 1], p=[0.35, 0.65])


def wave_cpl_constructor(dissipation_chance: float = 0):
    def wave_cpl(inputs: list[int]):
        if inputs[0] == 1:
            u = np.random.uniform(0, 1)

            if u < dissipation_chance:
                return 0
            else:
                return 1
        else:
            return 0

    return wave_cpl


class WaveWorld:
    def __init__(self,
                 world_size: int = 10,
                 local_simulation: bool = False):

        if world_size <= 2:
            raise ValueError("World size must be larger than 2.")

        self.world_size = world_size

        # Create nodes

        nodes = []

        left_node = Node(0, left_wave_cpl, name='Node 0')
        nodes.append(left_node)
        for i in range(1, world_size):
            nodes.append(Node(0, wave_cpl_constructor(), name=f'Node {i}'))

        # Create relationships between nodes

        left_node.add_neighbours([left_node, nodes[1]])
        left_node.add_parents([left_node, nodes[1]])
        for i in range(1, world_size - 1):
            nodes[i].add_neighbours([nodes[i-1], nodes[i+1]])
            nodes[i].add_parents([nodes[i-1], nodes[i], nodes[i+1]])
        nodes[-1].add_neighbours([nodes[-2], nodes[-1]])
        nodes[-1].add_parents([nodes[-2], nodes[-1]])

        self.network = Network(nodes, np.random.choice(nodes), local_simulation)

    def step(self, action: int, verbose: bool = False) -> tuple[int, int]:
        res = self.network.step(action)

        if verbose:
            print(repr(self.network))

        return res
