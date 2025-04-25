from torch_geometric.data import Data

from Dimploma.util import generate_random_graph_add_method


class GraphProvider:
    def __init__(self, fixed_graph: Data = None, device='cpu', nodes: int = 10, edges: int = 55, position=False, min_val=1, max_val=10):
        self.generate = False
        self.device = device
        self.nodes = nodes
        self.edges = edges
        self.position = position
        self.fixed_graph = None
        self.current_graph = None
        self.min_val = min_val
        self.max_val = max_val

        if fixed_graph is None and nodes < 0:
            raise ValueError("generate_graph_size must be greater than or equal to 0")
        else:
            self.set_fixed_graph(fixed_graph)

    def get_graph(self, new_graph=True):
        if self.generate:
            if self.current_graph is None or new_graph:
                self.current_graph = generate_random_graph_add_method(self.nodes, self.edges, self.min_val, self.max_val, self.device, self.position)
            return self.current_graph.clone()
        else:
            if self.fixed_graph is None:
                raise ValueError("Not supposed to generate graph and have no fixed graph set")
            else:
                return self.fixed_graph.clone()

    def set_fixed_graph(self, fixed_graph):
        self.fixed_graph = fixed_graph
        if fixed_graph is None:
            self.generate = True
        else:
            self.generate = False
            self.nodes = self.fixed_graph.x.shape[0]
            self.edges = self.fixed_graph.edge_index.shape[1]

    def set_size(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
