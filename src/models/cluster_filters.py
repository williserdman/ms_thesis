import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from src.args import MyArgs
from torch_geometric.utils import get_laplacian
from typing import Optional
from src.models.GARNOLDI import GArnoldi_prop


class ClusterWiseFilters(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
        multi: int,
        num_iters,
        num_clusters: int,
        loss_lambda: float,
    ):
        super().__init__()

        self.K = K
        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate
        self.num_clusters = num_clusters
        self.loss_lambda = loss_lambda

        self.clusters = nn.Parameter(
            torch.zeros(network_info.N, self.num_clusters)
        )  # (N, c)

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        self.cheb_diff = GArnoldi_prop()
        self.cheb_diff = DiffusionStep("chebyshev", self.K, self.hidden_dim)

        # num_iters = 4
        self.attn_layers = nn.ModuleList(
            [
                AttentionBlock(self.hidden_dim, self.K, self.dropout_rate, 4, multi)
                for _ in range(num_iters)
            ]
        )
        self.cluster_bias_proj = nn.Sequential(
            nn.Linear(self.num_clusters, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.K + 1),
        )

        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * multi),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim * multi, self.hidden_dim),
                )
                for _ in range(num_iters)
            ]
        )

        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)

        self.dropout = nn.Dropout(self.dropout_rate)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        [al.reset_parameters() for al in self.attn_layers]  # type: ignore

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)

        N, H = x.shape

        # Pre-compute adjacency
        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        )
        # Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(edge_index, edge_weight, num_nodes=N)  # type: ignore

        clusters = F.softmax(self.clusters, dim=-1)
        lap = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight_lap,
            (N, N),
            device=clusters.device,
            dtype=clusters.dtype,
        ).coalesce()

        Lc = torch.sparse.mm(lap, clusters)  # (N, c)
        qtLq = clusters.t() @ Lc  # (c, c)
        cluster_loss = torch.trace(qtLq)  # scalar

        msgs = self.cheb_diff(x, edge_index, edge_weight)  # list of (N, H)
        msgs = torch.stack(msgs, dim=1)  # (N, K+1, H)

        cluster_bias = self.cluster_bias_proj(clusters)  # (N, K+1)

        tokens = msgs
        orig_tokens = F.layer_norm(msgs, tokens.shape[-1:])

        out = orig_tokens.clone()
        for idx, attn_l in enumerate(self.attn_layers):
            tokens = F.layer_norm(out, tokens.shape[-1:])
            # print(tokens.shape)

            out = attn_l(N, H, tokens, cluster_bias) + orig_tokens
            out = F.layer_norm(out, out.shape[-1:])
            out = self.ffns[idx](out) + orig_tokens

        out = torch.sum(out, dim=1)  # (N, H)
        out = F.layer_norm(out, out.shape[-1:])
        out = self.decoder(out)

        cluster_loss = cluster_loss * torch.tensor(self.loss_lambda)

        return out, cluster_loss
