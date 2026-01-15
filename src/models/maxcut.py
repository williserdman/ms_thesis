import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from args import MyArgs
from torch_geometric.utils import get_laplacian
from typing import Optional


class MaxCutClusters(nn.Module):
    def __init__(self, dataset, args: MyArgs):
        super(MaxCutClusters, self).__init__()

        if args.hidden <= 32:
            num_heads = 4

        self.hidden_dim = args.hidden_dim
        self.num_classes = dataset.num_classes
        self.dprate = args.dprate

        self.encoder = nn.Linear(dataset.num_features, self.hidden_dim)

        self.clusters = nn.Parameter(torch.zeros(dataset.N, self.num_classes))  # (N, c)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.num_classes, self.hidden_dim * 4),
            nn.LeakyReLU(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)

        self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        return

    def forward(self, data):

        x = data.x
        x = self.encoder(x)
        N, H = x.shape

        edge_index = data.edge_index

        # Pre-compute adjacency
        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        )

        clusters = F.softmax(self.clusters, dim=-1)

        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, edge_weight, num_nodes=N  # type: ignore
        )

        lap = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight_lap,
            (N, N),
            device=clusters.device,
            dtype=clusters.dtype,
        ).coalesce()

        Lc = torch.sparse.mm(lap, clusters)  # (N, c)
        qtLq = clusters.t() @ Lc  # (c, c)
        loss = torch.trace(qtLq)  # scalar

        dummy_loss = loss

        x = torch.concat([x, clusters], dim=-1)  # (N, 2h)
        # print(x.shape)
        out = self.mlp(x)
        out = self.decoder(out)

        return F.log_softmax(out, dim=-1), dummy_loss
