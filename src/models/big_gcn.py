import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from args import MyArgs
from torch_geometric.utils import get_laplacian
from typing import Optional
from torch_geometric.nn import GCNConv


class BigGCN(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        multi: int,
        num_layers: int,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate
        self.multi = multi

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        self.gcns = nn.ModuleList(
            [GCNConv(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.multi * self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.multi * self.hidden_dim, self.hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)

        self.dropout = nn.Dropout(self.dropout_rate)

    def reset_parameters(self):
        return

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)

        N, H = x.shape

        # Pre-compute adjacency
        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        )
        # Laplacian
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=N)  # type: ignore

        for i in range(self.num_layers):
            x = self.gcns[i](x, edge_index, edge_weight)
            x = x.reshape(N, H)
            x = self.layers[i](x)
            x = x.reshape(N, H)

        out = F.layer_norm(x, x.shape)
        out = self.decoder(out)

        dummy_loss = torch.tensor(0)

        return F.log_softmax(out, dim=-1), dummy_loss
