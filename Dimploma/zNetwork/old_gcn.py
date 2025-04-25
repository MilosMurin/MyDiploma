
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)


class GCN31(torch.nn.Module):
    def __init__(self, out_size, num_node_features, rem_index=False):
        super().__init__()

        self.out_size = out_size
        self.num_node_features = num_node_features
        self.rem_index = rem_index

        self.conv1 = nng.GATConv(num_node_features, 16)
        self.conv2 = nng.GATConv(16 + num_node_features, 16)
        self.conv3 = nng.GATConv(16 + num_node_features, 16)

        self.conv_p2 = nng.GATConv(16 + num_node_features, 1)

        self.fc_v1 = nn.Linear(16, 16)
        self.fc_v2 = nn.Linear(16, 1)

        self.apply(_init_weights)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr[:,
                                                              :2]  # take only the normalized distances with edge_attr[:, 0]
        if self.rem_index:
            x = x[:, 1:]

        xa = F.relu(self.conv1(x, edge_index, edge_weight))
        xa = F.relu(self.conv2(torch.cat((xa, x), dim=1), edge_index, edge_weight))

        xa = F.relu(self.conv3(torch.cat((xa, x), dim=1), edge_index, edge_weight))

        px = F.relu(self.conv_p2(torch.cat((xa, x), dim=1), edge_index, edge_weight))

        px = px.view(-1, min(self.out_size, px.shape[0]))

        xa = xa.unsqueeze(0)
        xa = xa.view(-1, min(self.out_size, xa.shape[1]), self.conv1.out_channels)
        v = xa.mean(dim=1)
        vx = F.relu(self.fc_v1(v))
        vx = self.fc_v2(vx)

        return px, vx


class GCN32(torch.nn.Module):
    def __init__(self, out_size, num_node_features):
        super().__init__()

        self.out_size = out_size
        self.num_node_features = num_node_features

        self.conv1 = nng.GATConv(num_node_features, 16)
        self.conv2 = nng.GATConv(16 + num_node_features, 16)
        self.conv3 = nng.GATConv(16 + num_node_features, 16)

        self.conv_p1 = nng.GATConv(16 + num_node_features, 16)
        self.conv_p2 = nng.GATConv(16 + num_node_features, 1)

        self.fc_v1 = nn.Linear(16, 16)
        self.fc_v2 = nn.Linear(16, 1)

        self.apply(_init_weights)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr[:,
                                                              :2]  # take only the normalized distances with edge_attr[:, 0]

        xa = F.relu(self.conv1(x, edge_index, edge_weight))
        xa = F.relu(self.conv2(torch.cat((xa, x), dim=1), edge_index, edge_weight))

        xa = F.relu(self.conv3(torch.cat((xa, x), dim=1), edge_index, edge_weight))

        px = F.relu(self.conv_p1(torch.cat((xa, x), dim=1), edge_index, edge_weight))
        px = F.relu(self.conv_p2(torch.cat((px, x), dim=1), edge_index, edge_weight))

        px = px.view(-1, min(self.out_size, px.shape[0]))

        xa = xa.unsqueeze(0)
        xa = xa.view(-1, min(self.out_size, xa.shape[1]), self.conv1.out_channels)
        v = xa.mean(dim=1)
        vx = F.relu(self.fc_v1(v))
        vx = self.fc_v2(vx)

        return px, vx

class GCN21(torch.nn.Module):
    def __init__(self, out_size, num_node_features):
        super().__init__()

        self.out_size = out_size
        self.num_node_features = num_node_features

        self.conv1 = nng.GATConv(num_node_features, 16)
        self.conv2 = nng.GATConv(16 + num_node_features, 16)

        self.conv_p2 = nng.GATConv(16 + num_node_features, 1)

        self.fc_v1 = nn.Linear(16, 16)
        self.fc_v2 = nn.Linear(16, 1)

        self.apply(_init_weights)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr[:,
                                                              :2]  # take only the normalized distances with edge_attr[:, 0]

        xa = F.relu(self.conv1(x, edge_index, edge_weight))
        xa = F.relu(self.conv2(torch.cat((xa, x), dim=1), edge_index, edge_weight))

        px = F.relu(self.conv_p2(torch.cat((xa, x), dim=1), edge_index, edge_weight))

        px = px.view(-1, min(self.out_size, px.shape[0]))

        xa = xa.unsqueeze(0)
        xa = xa.view(-1, min(self.out_size, xa.shape[1]), self.conv1.out_channels)
        v = xa.mean(dim=1)
        vx = F.relu(self.fc_v1(v))
        vx = self.fc_v2(vx)

        return px, vx