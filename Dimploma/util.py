from random import randint

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


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


def generate_random_graph(node_amount, max_edge_amount=-1, edge_value_min=1, edge_value_max=10, device='cpu'):
    t_max = torch.sum(torch.arange(node_amount)).item()

    if max_edge_amount == -1 or max_edge_amount > t_max:
        max_edge_amount = t_max
    x = torch.zeros((node_amount, 2), device=device)
    edges = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.int64)
    edges_attr = torch.zeros((max_edge_amount, 2), device=device, dtype=torch.float32)
    edges_weight = torch.zeros(max_edge_amount, device=device, dtype=torch.float32)
    e = 0
    generate = False
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
            if not generate:
                e += 1
                if e >= max_edge_amount:
                    generate = True

            if generate:
                e = randint(0, max_edge_amount - 1)

    return Data(x=x, edge_index=edges.T, edge_attr=edges_attr, edge_weight=edges_weight)


def print_graph_info(graph: Data):
    for x in graph.x:
        print('Node: ', x)
        edges = 0
        for e in graph.edge_index:
            if e[0] == x:
                print(f'Edge {x}->{e[1]} - Price: {graph.edge_attr[edges, 0]}, Mark: {graph.edge_attr[edges, 1]}')
            edges += 1
