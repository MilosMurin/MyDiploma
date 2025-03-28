import torch
import torch.nn.functional as F
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, node_amount, node_features):
        super().__init__()
        self.node_amount = node_amount
        self.node_features = node_features

        matrix_size = node_amount * node_features

        self.linear1 = nn.Linear(matrix_size, matrix_size * 2)
        self.linear2 = nn.Linear(matrix_size * 2, matrix_size * 2)
        self.linear3 = nn.Linear(matrix_size * 2, matrix_size)

        self.linear_p = nn.Linear(matrix_size, node_amount)

        self.lin_v1 = nn.Linear(matrix_size * 2, matrix_size)


    def forward(self, matrix):

        x = F.relu(self.linear1(matrix.flatten()))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        px = F.relu(self.linear_p(x))

        vx = self.lin_v1(x)

        return px, vx
