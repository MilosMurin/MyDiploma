import torch
from torch_geometric.data import Data
from networkx.algorithms.tree import minimum_spanning_tree, maximum_spanning_tree
from torch_geometric.utils import from_networkx

from Dimploma.util import show_graph, my_to_networkx, data_to_matrix
from Dimploma.utils.graph_provider import GraphProvider

class EnvInfo:
    def __init__(self, graph_provider: GraphProvider, matrix_env=False):
        self.graph_provider = graph_provider
        self.matrix_env = matrix_env

    def create_graph(self, device='cpu'):
        return self.graph_provider.get_graph().to(device)

    def get_observation(self, graph: Data, matrix, steps, last_step=-1):
        cl = graph.clone().cpu()
        if last_step != -1:
            # only setting the mark of last step in the clone so i don't have to reset it in the env
            cl.x[last_step, 1] = 1
        return cl

class MatrixEnvInfo(EnvInfo):
    def __init__(self, graph_provider: GraphProvider, edge_info=False, node_info=False, step_info=False, adj_matrix
    =False):
        super().__init__(graph_provider, True)
        self.edge_info = edge_info
        self.node_info = node_info
        self.step_info = step_info
        self.adj_matrix = adj_matrix

    def get_observation(self, graph: Data, matrix, steps, last_step=-1):
        cl = super().get_observation(graph, matrix, steps, last_step)
        obs = []
        if self.adj_matrix:
            obs.append(matrix.clone().cpu().flatten())
        if self.edge_info:
            obs.append(graph.edge_attr[:, 1])
        if self.node_info:
            obs.append(cl.x[:, 1])
        if self.step_info:
            obs.append(torch.tensor([steps]))
        return torch.cat(obs)


class EnvMinimalTree:
    def __init__(self, env_info: EnvInfo, device='cpu', process_i=-1, env_i=-1):
        self.device = device
        self.env_info = env_info
        self.graph = self.env_info.create_graph()
        self.matrix = data_to_matrix(self.graph)
        self.processI = process_i
        self.envI = env_i
        self.min_tree_score = 0
        self.min_tree = None
        self.steps = 0
        self.parent = torch.arange(self.graph.x.shape[0], device=self.device)
        self.calculate_min_span_tree()

    def calculate_min_span_tree(self):
        g = my_to_networkx(self.graph.clone().cpu())
        min_tree = minimum_spanning_tree(g, 'edge_weight')
        self.min_tree = from_networkx(min_tree)
        self.min_tree_score = self.min_tree.edge_weight.sum().item() / 2
        return self.min_tree_score

    def reset(self):
        self.steps = 0
        self.graph = self.env_info.create_graph()
        self.matrix = data_to_matrix(self.graph)
        self.calculate_min_span_tree()
        self.parent = torch.arange(self.graph.x.shape[0], device=self.device)
        cl = self.env_info.get_observation(self.graph, self.matrix, self.steps)
        return cl, (self.graph.edge_attr[:, 1] != 1)

    def calculate_reward(self):
        return self.compute_objective_function() / -self.min_tree_score + 2  # +2 so that best reward is 1

    def step(self, action):
        self.steps += 1
        self.graph.edge_attr[action, 1] = 1
        cl = self.env_info.get_observation(self.graph, self.matrix, self.steps)

        # sel_graph_g = self.get_selected_treex()

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
            reward = self.calculate_reward() * terminal

        # if terminal:
        #     print("reward: ", reward)

        base_mask = self.graph.edge_attr[:, 1] != 1
        temp = self.parent[self.graph.edge_index]
        cycle_mask = temp[0] != temp[1]

        # print("Base mask", base_mask)
        # print("Cycle mask", cycle_mask)
        return cl, torch.logical_and(base_mask, cycle_mask).cpu(), reward, terminal, -1

    def compute_objective_function(self):
        return torch.sum(self.graph.edge_attr[:, 1] * self.graph.edge_attr[:, 2]).item()

    def get_selected_treex(self):
        mask = torch.argwhere(self.graph.edge_attr[:, 1] == 1)[:, 0]
        sel_graph = Data(x=self.graph.x, edge_index=self.graph.edge_index[:, mask],
                         edge_weight=self.graph.edge_weight[mask])
        return my_to_networkx(sel_graph)

    def show_selected_tree(self):
        show_graph(self.get_selected_treex())


class EnvMinimalTreeTwoStep(EnvMinimalTree):
    def __init__(self, env_info: EnvInfo, device='cpu', process_i=-1, env_i=-1):
        super().__init__(env_info, device, process_i, env_i)
        self.last_step = -1

    def reset(self):
        self.last_step = -1
        cl, _ = super().reset()
        return cl, torch.ones(self.graph.x.shape[0], dtype=torch.bool)

    def step(self, action):
        if self.last_step == -1:
            self.last_step = action

            sub_mask1 = self.graph.edge_index[0] == self.last_step
            sub_mask2 = self.graph.edge_index[1] == self.last_step
            attr_mask1 = self.graph.edge_attr[:, 1] != 1
            mask1 = self.graph.edge_index[1][torch.logical_and(sub_mask1, attr_mask1)]
            mask2 = self.graph.edge_index[0][torch.logical_and(sub_mask2, attr_mask1)]
            exist_edge_mask = torch.logical_or(torch.any(self.graph.x[:, 0].unsqueeze(1) == mask1, dim=1),
                                    torch.any(self.graph.x[:, 0].unsqueeze(1) == mask2, dim=1))

            cycle_mask = self.parent != self.parent[self.last_step]

            cl = self.env_info.get_observation(self.graph, self.matrix, self.steps, self.last_step)

            return cl, torch.logical_and(exist_edge_mask, cycle_mask).cpu(), 0, False, -1
        else:
            edge_index = self.find_edge(self.last_step, action)
            self.last_step = -1
            cl, edge_mask, reward, terminal, info = super().step(edge_index)

            mask = self.graph.edge_index.T[edge_mask].flatten()
            # print("Mask ", mask)

            final_mask = torch.any(self.graph.x[:,0].unsqueeze(1) == mask.flatten(), axis=1)


            return cl, final_mask.cpu(), reward, terminal, info

    def find_edge(self, p1, p2):
        mask = ((self.graph.edge_index[0] == p1) & (self.graph.edge_index[1] == p2)) | \
               ((self.graph.edge_index[0] == p2) & (self.graph.edge_index[1] == p1))
        return torch.arange(self.graph.edge_index.shape[1], device=self.device)[mask]


class EnvMinimalTreeTwoStepRew(EnvMinimalTreeTwoStep):
    def __init__(self, env_info: EnvInfo, device='cpu', process_i=-1, env_i=-1):
        super().__init__(env_info, device, process_i, env_i)


    def step(self, action):
        if self.last_step == -1:
            return super().step(action) # no last edge means first pick -> no reward based on the min tree

        # get last step and see if the edge (last_step and action) is part of the min tree
        last_step = self.last_step
        cl, mask, rew, term, info = super().step(action)

        if rew == 0 and not term:
            # test if the edge is part of minimal tree
            if torch.any(((self.graph.edge_index[0] == last_step) & (self.graph.edge_index[1] == action)) | \
                   ((self.graph.edge_index[0] == action) & (self.graph.edge_index[1] == last_step))):
                rew = 1
        return cl, mask, rew, term, info

class EnvMinimalTreeTwoStepHeur(EnvMinimalTreeTwoStep):
    def __init__(self, env_info: EnvInfo, device='cpu', process_i=-1, env_i=-1):
        super().__init__(env_info, device, process_i, env_i)

    def step(self, action):
        if self.last_step == -1:
            # no last edge - pass whatever comes from parent
            return super().step(action)

        index = self.find_edge(self.last_step, action)  # the chosen edge
        temp = self.parent[self.graph.edge_index]
        cycle_mask = temp[0] != temp[1]  # covers also if the edge was already picked
        min_pos = self.graph.edge_weight[cycle_mask].min()  # the lowest price for an edge

        cl, mask, rew, term, info = super().step(action)

        if rew == 0 and not term: # if reward is 0 and not term - just a normal step with no reward from anything else - i can edit the reward
            if self.graph.edge_weight[index] == min_pos: # if an edge with the lowest possible value was picked
                rew = .1
            else:
                rew = -.1

        return cl, mask, rew, term, info

class EnvMaximalTreeTwoStep(EnvMinimalTreeTwoStep):
    def __init__(self, env_info: EnvInfo, device='cpu', process_i=-1, env_i=-1):
        super().__init__(env_info, device, process_i, env_i)

    def calculate_min_span_tree(self): # calculating maximal span tree
        g = my_to_networkx(self.graph.clone().cpu())
        min_tree = maximum_spanning_tree(g, 'edge_weight')
        self.min_tree = from_networkx(min_tree)
        self.min_tree_score = self.min_tree.edge_weight.sum().item() / 2
        return self.min_tree_score

    def calculate_reward(self):
        return self.compute_objective_function() / self.min_tree_score


class EnvMaximalTreeTwoStepHeur(EnvMaximalTreeTwoStep):
    def __init__(self, env_info: EnvInfo, device='cpu', process_i=-1, env_i=-1):
        super().__init__(env_info, device, process_i, env_i)

    def step(self, action):
        if self.last_step == -1:
            # no last edge - pass whatever comes from parent
            return super().step(action)

        index = self.find_edge(self.last_step, action)  # the chosen edge
        temp = self.parent[self.graph.edge_index]
        cycle_mask = temp[0] != temp[1]  # covers also if the edge was already picked
        max_pos = self.graph.edge_weight[cycle_mask].max()  # the lowest price for an edge

        cl, mask, rew, term, info = super().step(action)

        if rew == 0 and not term: # if reward is 0 and not term - just a normal step with no reward from anything else - i can edit the reward
            if self.graph.edge_weight[index] == max_pos: # if an edge with the lowest possible value was picked
                rew = 1

        return cl, mask, rew, term, info




