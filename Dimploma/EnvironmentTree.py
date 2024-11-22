import torch
from torch_geometric.data import Data
from networkx.algorithms.tree import minimum_spanning_tree
from torch_geometric.utils import from_networkx

from Dimploma.util import show_graph
from Dimploma.utils.graph_provider import GraphProvider
from util import my_to_networkx


class EnvMinimalTree:
    def __init__(self, graph_provider: GraphProvider, device='cpu', process_i=-1, env_i=-1):
        self.device = device
        self.graph_provider = graph_provider
        self.graph = self.graph_provider.get_graph()
        self.processI = process_i
        self.envI = env_i
        self.min_tree_score = 0
        self.min_tree = None
        self.steps = 0
        self.parent = torch.arange(self.graph.x.shape[0])
        self.calculate_min_span_tree()

    def calculate_min_span_tree(self):
        g = my_to_networkx(self.graph)
        min_tree = minimum_spanning_tree(g, 'edge_weight')
        self.min_tree = from_networkx(min_tree)
        self.min_tree_score = self.min_tree.edge_weight.sum().item() / 2
        return self.min_tree_score

    def reset(self):
        self.steps = 0
        self.graph = self.graph_provider.get_graph()
        self.calculate_min_span_tree()
        self.parent = torch.arange(self.graph.x.shape[0])
        cl = self.graph.clone()
        return cl, (cl.edge_attr[:, 1] != 1)

    def step(self, action):
        self.steps += 1
        self.graph.edge_attr[action, 1] = 1
        cl = self.graph.clone()

        sel_graph_g = self.get_selected_treex()

        terminal = self.steps >= self.graph.x.shape[0] - 1
        reward = 0

        # cycles masking
        i, j = self.graph.edge_index[:, action]
        self.parent[self.parent == self.parent[i]] = self.parent[j]

        # cycles = len(list(nx.simple_cycles(sel_graph_g)))
        # if cycles != 0:
        #     terminal = True
        #     reward = -1

        if terminal and reward == 0:
            reward = (self.compute_objective_function() / -self.min_tree_score + 2) * terminal  # +2 so that best reward is 1

        # if terminal:
        #     print("reward: ", reward)

        base_mask = cl.edge_attr[:, 1] != 1
        temp = self.parent[self.graph.edge_index]
        cycle_mask = temp[0] != temp[1]

        return cl, torch.logical_and(base_mask, cycle_mask), reward, terminal, -1

    def compute_objective_function(self):
        return torch.sum(self.graph.edge_attr[:, 1] * self.graph.edge_attr[:, 0]).item()

    def get_selected_treex(self):
        mask = torch.argwhere(self.graph.edge_attr[:, 1] == 1)[:, 0]
        sel_graph = Data(x=self.graph.x, edge_index=self.graph.edge_index[:, mask],
                         edge_weight=self.graph.edge_weight[mask])
        return my_to_networkx(sel_graph)

    def show_selected_tree(self):
        show_graph(self.get_selected_treex())


class EnvMinimalTreeTwoStep(EnvMinimalTree):
    def __init__(self, graph_provider: GraphProvider, device='cpu', process_i=-1, env_i=-1):
        super().__init__(graph_provider, device, process_i, env_i)
        self.last_step = -1

    def reset(self):
        self.last_step = -1
        cl, _ = super().reset()
        return cl, torch.ones(cl.x.shape[0], dtype=torch.bool)

    def step(self, action):
        if self.last_step == -1:
            self.last_step = action
            cl = self.graph.clone()

            # sub_mask1 = cl.edge_index[0] == self.last_step
            # sub_mask2 = cl.edge_index[1] == self.last_step
            # attr_mask1 = cl.edge_attr[:, 1] != 1
            # mask1 = cl.edge_index[1][torch.logical_and(sub_mask1, attr_mask1)]
            # mask2 = cl.edge_index[0][torch.logical_and(sub_mask2, attr_mask1)]
            # mask = torch.logical_or(torch.any(cl.x[:, 0].unsqueeze(1) == mask1, dim=1),
            #                         torch.any(cl.x[:, 0].unsqueeze(1) == mask2, dim=1))

            mask = self.parent != self.parent[self.last_step]

            # only setting the mark of last step in the clone so i don't have to reset it in the env
            cl.x[self.last_step, 1] = 1

            return cl, mask, 0, False, -1
        else:
            edge_index = self.find_edge(self.last_step, action)
            self.last_step = -1
            cl, _, reward, terminal, info = super().step(edge_index)
            return cl, torch.ones(cl.x.shape[0], dtype=torch.bool), reward, terminal, info

    def find_edge(self, p1, p2):
        mask = ((self.graph.edge_index[0] == p1) & (self.graph.edge_index[1] == p2)) | \
               ((self.graph.edge_index[0] == p2) & (self.graph.edge_index[1] == p1))
        return torch.arange(self.graph.edge_index.shape[1])[mask]
