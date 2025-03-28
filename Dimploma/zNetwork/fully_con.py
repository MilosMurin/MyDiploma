import torch
import torch.nn.functional as F
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, node_amount, node_features):
        super().__init__()
        self.node_amount = node_amount
        self.node_features = node_features