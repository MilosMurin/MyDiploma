import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch
 

class HybridConv(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.state_size = in_features

        self.lin = nn.Linear(in_features, out_features)
        self.conv = GATConv(in_features * 2, out_features)

    def out_channels(self):
        return self.lin.out_features + self.conv.out_channels

    def forward(self, x_lin, x_conv, edge_index, edge_attr=None):

        x_combined = torch.cat([x_lin, x_conv], dim=-1)
        x_conv = self.conv(x_combined, edge_index, edge_attr=edge_attr)

        x_lin = self.lin(x_lin)

        return x_lin, x_conv

# This is the same model as in model.py, but with an additional attention mechanism 
# for the actor head that gives every node information about the global state of the graph
class HybridNetworkGlobal(torch.nn.Module):
    def __init__(self, state_size, node_count, remove_index=False, position=False):
        super().__init__()

        self.state_size = state_size
        self.node_count = node_count
        self.remove_index = remove_index
        self.position = position

        self.h1 = HybridConv(state_size, 16)
        self.h2 = HybridConv(16, 16)
        self.h3 = HybridConv(16, 16)
        self.h4 = HybridConv(16, 16)

        self.weighting_attention = nn.Linear(self.h4.out_channels(), 1)

        self.a1 = nn.Linear(self.h4.out_channels() * 2, 32)
        self.a2 = nn.Linear(32, 1)

        self.v1 = nn.Linear(self.h4.out_channels(), 16)
        self.v2 = nn.Linear(16, 1)

    def forward(self, data):
        x = data.x
        if not self.position:
            x = x[:, :2]
        if self.remove_index:
            x = x[:, 1:]
        if data.edge_index is not None:
            data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
            data.edge_attr = torch.cat([data.edge_attr, -data.edge_attr], dim=0)

        x, edge_index = x, data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        x_lin, x_conv = self.h1(x, x, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_lin, x_conv = self.h2(x_lin, x_conv, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_lin, x_conv = self.h3(x_lin, x_conv, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_lin, x_conv = self.h4(x_lin, x_conv, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_combined = torch.cat([x_lin, x_conv], dim=-1)

        weights = self.weighting_attention(x_combined) # Learned weights
        weights = weights.view(-1, self.node_count)
        weights = F.softmax(weights, dim=1).unsqueeze(2)  

        # sum features over all nodes
        graph = x_combined.view(-1, self.node_count, x_combined.shape[1])
        graph = graph * weights
        graph_embedding = torch.sum(graph, dim=1)  # Weighted sum
        
        # Actor head
        graph_embedding_expanded = graph_embedding.unsqueeze(1).expand(-1, self.node_count, -1).reshape(-1, graph_embedding.shape[1])
        node_features = torch.cat([x_combined, graph_embedding_expanded], dim=-1)
        x = F.relu(self.a1(node_features))
        x = self.a2(x)
        X = x.view(-1, self.node_count)

        # Critic head
        V = F.relu(self.v1(graph_embedding))
        V = self.v2(V)

        return X, V#, V.clone().detach()