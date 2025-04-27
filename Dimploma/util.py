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


def get_out_edges(graph: Data, to_sort=False):
    data = torch.zeros(graph.x.shape[0], dtype= torch.int32)
    for i in range(graph.x.shape[0]):
        data[i] = torch.logical_or(graph.edge_index[0] == i, graph.edge_index[1] == i).sum()
    if to_sort:
        sorted_data = torch.stack((data.sort().indices, data.sort().values))
        return sorted_data
    return data

def get_node_sums_norm(graph: Data):
    sums = torch.zeros(graph.x.shape[0])
    degrees = torch.zeros(graph.x.shape[0])
    for i in range(graph.x.shape[0]):
        sums[i] = graph.edge_attr[torch.logical_or(graph.edge_index[0] == i, graph.edge_index[1] == i), 0].sum()
        degrees[i] = torch.logical_or(graph.edge_index[0] == i, graph.edge_index[1] == i).sum()
    sums /= degrees
    sorted_data = torch.stack((sums.sort().indices, sums.sort().values))
    return sorted_data

def get_node_sums(graph: Data, to_sort=False):
    data = torch.zeros(graph.x.shape[0])
    for i in range(graph.x.shape[0]):
        data[i] = graph.edge_attr[
            torch.logical_or(graph.edge_index[0] == i, graph.edge_index[1] == i), 2].sum()
    if to_sort:
        sorted_data = torch.stack((data.sort().indices, data.sort().values))
        return sorted_data
    return data

def show_data(da):
    gr = my_to_networkx(da)
    show_graph(gr)


def show_graph(gr, widths=None, with_labels=True):
    gr_pos = nx.spring_layout(gr)
    nx.draw_networkx(gr, gr_pos, with_labels=True)

    gr_labels = nx.get_edge_attributes(gr, 'edge_weight')
    if widths is not None:
        nx.draw_networkx_edges(gr, gr_pos, width=widths)
    if with_labels:
        nx.draw_networkx_edge_labels(gr, gr_pos, edge_labels=gr_labels)

    plt.show()


def my_to_networkx(graph: Data):
    return to_networkx(graph, to_undirected=True, edge_attrs=['edge_weight'])

def generate_random_full_graph(node_amount, edge_value_min=1, edge_value_max=10, device='cpu', position=False):
    max_edge_amount = torch.sum(torch.arange(node_amount)).item()
    x = torch.zeros((node_amount, 4 if position else 2), device=device)
    edges = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.int64)
    edges_attr = torch.zeros((max_edge_amount, 3), device=device, dtype=torch.float32)
    edges_weight = torch.zeros(max_edge_amount, device=device, dtype=torch.float32)
    e = 0
    for i in range(node_amount):
        x[i, 0] = i
        if position:
            x[i, 2] = randint(-node_amount, node_amount)
            x[i, 3] = randint(-node_amount, node_amount)

    for i in range(node_amount):
        for j in range(i, node_amount):
            if i == j:
                continue
            edges[e, 0] = i
            edges[e, 1] = j
            if position:
                edges_weight[e] = torch.dist(x[i, 2:4], x[j, 2:4]).item()
            else:
                edges_weight[e] = randint(edge_value_min, edge_value_max) if i != j else 0
            edges_attr[e, 2] = edges_weight[e]
            edges_attr[e, 1] = 0
            e += 1

    max_distance = torch.max(edges_attr[:, 2])
    edges_attr[:, 0] = edges_attr[:, 2] / max_distance


    return Data(x=x, edge_index=edges.T, edge_attr=edges_attr, edge_weight=edges_weight)


def generate_random_graph_add_method(node_amount, max_edge_amount=-1, edge_value_min=1, edge_value_max=10, device='cpu', position=False):
    t_max = torch.sum(torch.arange(node_amount)).item()

    if max_edge_amount < node_amount - 1:
        raise Exception("Cannot create a graph for training that will have more than one component")

    if max_edge_amount == -1 or max_edge_amount > t_max:
        return generate_random_full_graph(node_amount, edge_value_min, edge_value_max, device, position)

    x = torch.stack((torch.arange(node_amount), *[torch.zeros(node_amount)] * (3 if position else 1)), dim=1).to(device)
    parent = torch.arange(node_amount, device=device, dtype=torch.int)
    edge_index = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.int64)
    edges_attr = torch.zeros((max_edge_amount, 3), device=device, dtype=torch.float32)
    edges_weight = torch.zeros(max_edge_amount, device=device, dtype=torch.float32)
    e = 0

    for i in range(node_amount):
        if position:
            x[i, 2] = randint(-node_amount, node_amount)
            x[i, 3] = randint(-node_amount, node_amount)

    # make a basic tree
    while not torch.all(parent == parent[0]):
        mask = parent == parent[0]

        from_nodes = x[mask, 0]
        to_nodes = x[~mask, 0]

        random_from = int(from_nodes[randint(0, from_nodes.shape[0] - 1)].item())
        random_to = int(to_nodes[randint(0, to_nodes.shape[0] - 1)].item())

        edge_index[e] = torch.tensor([random_from, random_to], device=device)
        if position:
            edges_weight[e] = torch.dist(x[random_from, 2:4], x[random_to, 2:4]).item()
        else:
            edges_weight[e] = randint(edge_value_min, edge_value_max)
        edges_attr[e, 2] = edges_weight[e]
        e += 1
        parent[random_to] = parent[random_from]

    # fill out the rest with random picks
    while e < max_edge_amount:

        node_from = randint(0, node_amount - 1)

        possibilities = torch.arange(node_amount, device=device)

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
            node_to = possibilities[randint(0, possibilities.shape[0] - 1)]
            edge_index[e] = torch.tensor([node_from, node_to], device=device)
            if position:
                edges_weight[e] = torch.dist(x[node_from, 2:4], x[node_to, 2:4]).item()
            else:
                edges_weight[e] = randint(edge_value_min, edge_value_max)
            edges_attr[e, 2] = edges_weight[e]
            e += 1

    max_distance = torch.max(edges_attr[:, 2])
    edges_attr[:, 0] = edges_attr[:, 2] / max_distance

    return Data(x=x, edge_index=edge_index.T, edge_attr=edges_attr, edge_weight=edges_weight)


def print_graph_info(graph: Data):
    for x in graph.x:
        print('Node: ', x)
        edges = 0
        for e in graph.edge_index:
            if e[0] == x:
                print(f'Edge {x}->{e[1]} - Price: {graph.edge_attr[edges, 0]}, Mark: {graph.edge_attr[edges, 1]}')
            edges += 1


def data_to_matrix(data: Data, normalized = True):
    matrix = torch.zeros((data.x.shape[0], data.x.shape[0]))
    matrix[data.edge_index[0], data.edge_index[1]] = data.edge_attr[:, 0] if normalized else data.edge_weight
    matrix[data.edge_index[1], data.edge_index[0]] = data.edge_attr[:, 0] if normalized else data.edge_weight
    return matrix

def data_to_edge_graph(graph: Data):
    x = torch.arange(graph.x.shape[0] + graph.edge_index.shape[1], device=graph.x.device)
    x[graph.x.shape[0]:] = -x[graph.x.shape[0]:] + 9
    x = torch.stack([x, torch.zeros(x.shape[0])])
    x[1, graph.x.shape[0]:] = graph.edge_attr[:, 0]

    edge_index = torch.zeros((2, graph.edge_index.shape[1] * 2), device=graph.x.device, dtype=torch.int64)
    edge_index[0, :graph.edge_index.shape[1]] = graph.edge_index[0, :]
    edge_index[0, graph.edge_index.shape[1]:] = x[0, graph.x.shape[0]:]
    edge_index[1, :graph.edge_index.shape[1]] = x[0, graph.x.shape[0]:]
    edge_index[1, graph.edge_index.shape[1]:] = graph.edge_index[1, :]

    return Data(x=x, edge_index=edge_index)
