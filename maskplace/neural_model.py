# neural_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        agg = torch.matmul(adj, x)   # neighbor aggregation
        return F.relu(self.linear(agg))


class GNNOrderingModel(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, num_layers=3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(in_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)

        scores = self.mlp(x).squeeze(-1)  # [N]
        return scores