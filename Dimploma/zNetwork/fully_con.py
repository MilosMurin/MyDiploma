import torch
import torch.nn.functional as F
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, node_amount, node_features, edge_amount, main_layers=3, p_layers=2, v_layers=2, edge_info=False, node_info=False, step_info=False, adj_matrix
    =False):
        super().__init__()
        self.node_amount = node_amount
        self.node_features = node_features
        self.edge_amount = edge_amount
        self.main_layers = main_layers
        self.p_layers = p_layers
        self.v_layers = v_layers
        self.edge_info = edge_info
        self.node_info = node_info
        self.step_info = step_info
        self.adj_matrix = adj_matrix

        matrix_size = ((node_amount * node_amount) if adj_matrix else 0) + (self.edge_amount if edge_info else 0) + (self.node_amount if node_info else 0) + (1 if self.step_info else 0)

        self.linear1 = nn.Linear(matrix_size, matrix_size * (2 if self.main_layers > 1 else 1))

        if self.main_layers > 1:
            self.linear2 = nn.Linear(self.linear1.out_features, matrix_size * (2 if self.main_layers > 2 else 1))
        if self.main_layers > 2:
            self.linear3 = nn.Linear(self.linear2.out_features, matrix_size * (2 if self.main_layers > 3 else 1))
        if self.main_layers > 3:
            self.linear4 = nn.Linear(self.linear3.out_features, matrix_size * (2 if self.main_layers > 4 else 1))
        if self.main_layers > 4:
            self.linear5 = nn.Linear(self.linear4.out_features, matrix_size * (2 if self.main_layers > 5 else 1))
        if self.main_layers > 5:
            self.linear6 = nn.Linear(self.linear5.out_features, matrix_size * 1)

        self.linear_p = nn.Linear(matrix_size, (matrix_size if p_layers > 1 else node_amount))
        if self.p_layers > 1:
            self.linear_p2 = nn.Linear(matrix_size, node_amount)

        self.lin_v1 = nn.Linear(matrix_size, (matrix_size if v_layers > 1 else 1))
        if self.v_layers > 1:
            self.lin_v2 = nn.Linear(matrix_size, 1)


    def forward(self, matrix):
        # if len(matrix.shape) > 2:
        #     matrix = matrix.flatten(start_dim=1)
        # else:
        #     matrix = matrix.flatten()

        x = F.relu(self.linear1(matrix))

        if self.main_layers > 1:
            x = F.relu(self.linear2(x))
        if self.main_layers > 2:
            x = F.relu(self.linear3(x))
        if self.main_layers > 3:
            x = F.relu(self.linear4(x))
        if self.main_layers > 4:
            x = F.relu(self.linear5(x))
        if self.main_layers > 5:
            x = F.relu(self.linear6(x))

        px = F.relu(self.linear_p(x))
        if self.p_layers > 1:
            px = F.relu(self.linear_p2(x))

        vx = self.lin_v1(x)
        if self.v_layers > 1:
            vx = F.relu(vx)
            vx = self.lin_v2(vx)

        return px, vx
