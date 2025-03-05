
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)


class GCN(torch.nn.Module):
    def __init__(self, out_size, num_node_features, cat=True, conv_layers=3, conv_p_layers=1, linear_layers=2, remove_index=False):
        super().__init__()

        self.out_size = out_size
        self.num_node_features = num_node_features
        self.cat = cat
        self.conv_layers = conv_layers
        self.conv_p_layers = conv_p_layers
        self.linear_layers = linear_layers
        to_add = num_node_features if cat else 0
        self.remove_index = remove_index

        self.conv1 = nng.GATConv(num_node_features, 16)
        if conv_layers > 1:
            self.conv2 = nng.GATConv(16 + to_add, 16)
        if conv_layers > 2:
            self.conv3 = nng.GATConv(16 + to_add, 16)
        if conv_layers > 3:
            self.conv4 = nng.GATConv(16 + to_add, 16)

        if conv_p_layers == 1:
            self.conv_p1 = nng.GATConv(16 + to_add, 1)
        elif conv_p_layers > 1:
            self.conv_p1 = nng.GATConv(16 + to_add, 16)
            self.conv_p2 = nng.GATConv(16 + to_add, 1)

        if conv_layers == 1:
            self.fc_v1 = nn.Linear(16, 1)
        if conv_layers > 1:
            self.fc_v1 = nn.Linear(16, 16)
            self.fc_v2 = nn.Linear(16, 1)

        self.apply(_init_weights)

    def forward(self, data):
        x = data.x
        if self.remove_index:
            x = x[:, 1:]
        edge_index = data.edge_index
        edge_weight = data.data.edge_attr[:, :2] # take only the normalized distances with edge_attr[:, 0]
        # print(data.num_node_features)
        # print(f'x start: {x.shape}')
        # print(f'Edge index start: {edge_index.shape}')
        # print(f'Edge index start: {edge_weight}')

        xa = F.relu(self.conv1(x, edge_index, edge_weight))
        # print(f'Edge index after conv1: {edge_weight}')
        # print(f'x conv1: {x.shape}')
        # print(f'xa conv1: {xa.shape}')
        # print(f'Edge index conv1: {edge_index.shape}')
        # x = F.dropout(x, training=self.training)
        if self.conv_layers > 1:
            xa = F.relu(self.conv2(torch.cat((xa, x), dim=1) if self.cat else xa, edge_index, edge_weight))
        # print(f'Edge index after conv2: {edge_weight}')
        # print(f'x conv2: {x.shape}')
        # print(f'xa conv2: {xa.shape}')
        # print(f'Edge index conv2: {edge_index.shape}')

        # x = F.sigmoid(self.conv3(x, edge_index, edge_weight))
        if self.conv_layers > 2:
            xa = F.relu(self.conv3(torch.cat((xa, x), dim=1) if self.cat else xa, edge_index, edge_weight))
        # # print(f'Edge index after conv3: {edge_weight}')
        # # print(f'x conv3: {x.shape}')
        if self.conv_layers > 3:
            xa = F.relu(self.conv4(torch.cat((xa, x), dim=1) if self.cat else xa, edge_index, edge_weight))
        # x = F.sigmoid(self.conv4(x, edge_index, edge_weight))
        # print(f'Edge index after conv4: {edge_weight}')

        px = F.relu(self.conv_p1(torch.cat((xa, x), dim=1) if self.cat else xa, edge_index, edge_weight))
        # print(f'x convp1: {x.shape}')
        # px = F.relu(self.conv_p2(torch.cat((xa, x), dim=1), edge_index, edge_weight))
        if self.conv_p_layers > 1:
            px = F.relu(self.conv_p2(torch.cat((px, x), dim=1) if self.cat else px, edge_index, edge_weight))
        # print(f'x convp2: {x.shape}')

        px = px.view(-1, min(self.out_size, px.shape[0]))
        # X = X.view(-1, self.node_count)

        xa = xa.unsqueeze(0)
        xa = xa.view(-1, min(self.out_size, xa.shape[1]), self.conv1.out_channels)
        v = xa.mean(dim=1)
        vx = self.fc_v1(v)
        if self.linear_layers > 1:
            vx = F.relu(vx)
            vx = self.fc_v2(vx)

        # print(f'x result: {X.shape}')
        # print(f'v result: {V.shape}')
        # print(f'Edge index end: {edge_weight}')
        return px, vx