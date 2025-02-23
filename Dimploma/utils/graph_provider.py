from torch_geometric.data import Data

from Dimploma.util import generate_random_graph_add_method


class GraphProvider:
    def __init__(self, fixed_graph: Data = None, device='cpu', nodes: int = 10, edges: int = 55, position=False):
        self.generate = False
        self.device = device
        self.nodes = nodes
        self.edges = edges
        self.position = position

        if fixed_graph is None:
            if nodes < 0:
                raise ValueError("generate_graph_size must be greater than or equal to 0")
            else:
                self.generate = True
        else:
            self.fixed_graph = fixed_graph

    def get_graph(self):
        if self.generate:
            return generate_random_graph_add_method(self.nodes, self.edges, device=self.device, position=self.position)
        else:
            if self.fixed_graph is None:
                raise ValueError("generate_graph_size must be greater than or equal to 0")
            else:
                return self.fixed_graph.clone()

    def set_fixed_graph(self, fixed_graph):
        self.fixed_graph = fixed_graph
        if fixed_graph is None:
            self.generate = True

    def set_size(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
