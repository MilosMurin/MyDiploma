from random import randint

import networkx as nx
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def plot_training(path):
    scores = pd.read_csv(path + '/score.csv')
    plt.plot(scores["iteration"], scores["avg_reward"])
    plt.plot(scores["iteration"], scores["best_avg_reward"])
    plt.plot(scores["iteration"], scores["best_reward"])
    plt.show()

def get_node_sums(graph: Data):
    data = torch.zeros(graph.x.shape[0])
    for i in range(graph.x.shape[0]):
        data[i] = graph.edge_attr[
            torch.logical_or(graph.edge_index[0] == i, graph.edge_index[1] == i), 0].sum()
    sorted_data = torch.stack((data.sort().indices, data.sort().values))
    return sorted_data

def show_data(da):
    gr = my_to_networkx(da)
    show_graph(gr)


def show_graph(gr):
    gr_pos = nx.spring_layout(gr)
    nx.draw_networkx(gr, gr_pos, with_labels=True)

    gr_labels = nx.get_edge_attributes(gr, 'edge_weight')
    nx.draw_networkx_edge_labels(gr, gr_pos, edge_labels=gr_labels)

    plt.show()


def my_to_networkx(graph: Data):
    return to_networkx(graph, to_undirected=True, edge_attrs=['edge_weight'])

def generate_random_full_graph(node_amount, edge_value_min=1, edge_value_max=10, device='cpu'):
    max_edge_amount = torch.sum(torch.arange(node_amount)).item()
    x = torch.zeros((node_amount, 2), device=device)
    edges = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.int64)
    edges_attr = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.float32)
    edges_weight = torch.zeros(max_edge_amount, device=device, dtype=torch.float32)
    e = 0
    for i in range(node_amount):
        x[i, 0] = i
        for j in range(i, node_amount):
            if i == j:
                continue
            edges[e, 0] = i
            edges[e, 1] = j
            edges_weight[e] = randint(edge_value_min, edge_value_max) if i != j else 0
            edges_attr[e, 0] = edges_weight[e]
            edges_attr[e, 1] = 0
            e += 1

    return Data(x=x, edge_index=edges.T, edge_attr=edges_attr, edge_weight=edges_weight)


def generate_random_graph_remove_method(node_amount, max_edge_amount=-1, edge_value_min=1, edge_value_max=10, device='cpu'):
    full_graph = generate_random_full_graph(node_amount, edge_value_min, edge_value_max, device)
    t_max = torch.sum(torch.arange(node_amount)).item()

    if max_edge_amount < node_amount - 1:
        raise Exception("Cannot create a graph for training that will have more than one component")

    if max_edge_amount == -1 or max_edge_amount > t_max:
        return full_graph

    # TODO remove edges that do not create two components when removed until the desired amount


    pass


def generate_random_graph_add_method(node_amount, max_edge_amount=-1, edge_value_min=1, edge_value_max=10, device='cpu'):
    t_max = torch.sum(torch.arange(node_amount)).item()

    if max_edge_amount < node_amount - 1:
        raise Exception("Cannot create a graph for training that will have more than one component")

    if max_edge_amount == -1 or max_edge_amount > t_max:
        return generate_random_full_graph(node_amount, edge_value_min, edge_value_max, device)

    x = torch.stack((torch.arange(node_amount), torch.zeros(node_amount))).T.to(device)
    parent = torch.arange(node_amount, device=device, dtype=torch.int)
    edge_index = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.int64)
    edges_attr = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.float32)
    edges_weight = torch.zeros(max_edge_amount, device=device, dtype=torch.float32)
    e = 0

    # make a basic tree
    while not torch.all(parent == parent[0]):
        mask = parent == parent[0]

        from_nodes = x[mask, 0]
        to_nodes = x[~mask, 0]

        random_from = int(from_nodes[randint(0, from_nodes.shape[0] - 1)].item())
        random_to = int(to_nodes[randint(0, to_nodes.shape[0] - 1)].item())

        edge_index[e] = torch.tensor([random_from, random_to])
        edges_weight[e] = randint(edge_value_min, edge_value_max)
        edges_attr[e, 0] = edges_weight[e]
        e += 1
        parent[random_to] = parent[random_from]

    # fill out the rest with random picks
    while e < max_edge_amount:

        node_from = randint(0, node_amount - 1)

        possibilities = torch.arange(node_amount)

        # filter out self loops
        possibilities = possibilities[possibilities != node_from]

        # filter out already existing edges
        banned_nodes1 = edge_index[edge_index[:, 0] == node_from, 1]
        banned_nodes2 = edge_index[edge_index[:, 1] == node_from, 0]
        mask = torch.logical_not(
            torch.any(possibilities.unsqueeze(1) == torch.unique(torch.cat((banned_nodes1, banned_nodes2))), dim=1))

        possibilities = possibilities[mask]

        if possibilities.shape[0] > 0:
            # pick one and create edge
            node_to = randint(0, possibilities.shape[0] - 1)
            edge_index[e] = torch.tensor([node_from, possibilities[node_to]])
            edges_weight[e] = randint(edge_value_min, edge_value_max)
            edges_attr[e, 0] = edges_weight[e]

            e += 1

    return Data(x=x, edge_index=edge_index.T, edge_attr=edges_attr, edge_weight=edges_weight)


def print_graph_info(graph: Data):
    for x in graph.x:
        print('Node: ', x)
        edges = 0
        for e in graph.edge_index:
            if e[0] == x:
                print(f'Edge {x}->{e[1]} - Price: {graph.edge_attr[edges, 0]}, Mark: {graph.edge_attr[edges, 1]}')
            edges += 1
