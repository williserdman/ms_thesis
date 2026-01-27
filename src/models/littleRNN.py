import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from args import MyArgs
from torch_geometric.utils import get_laplacian
from typing import Optional


class DiffusionStep(MessagePassing, nn.Module):
    """
    Message passing layer for Graph Diffusion based attention.
    Performs X' = P * X where P is the proppagation matrix.
    """

    def __init__(self, prop_type: str, K: int, hidden_dim):
        super(DiffusionStep, self).__init__(aggr="sum")
        assert prop_type in {"monomial", "chebyshev", "mlp"}, prop_type

        self.prop_type = prop_type
        self.K = K

    def forward(self, x, edge_index, edge_weight) -> list[torch.Tensor]:  # type: ignore
        """
        Docstring for forward

        :param self: Description
        :param x: (N, num_channels)
        :param edge_index: (2, num_edges)
        :param edge_weight: (E), computed through gcn_norm
        """

        out = [x]
        if self.K <= 0:
            return out

        if self.prop_type == "monomial":
            h = x
            for _ in range(self.K):
                h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
                out.append(h)
            return out

        elif self.prop_type == "chebyshev":
            # chebyshev
            # T_0 = x
            # T_1 = L x = x - A_norm x
            # T_k = 2 * L * T_{k-1} - T_{k-2}
            A_x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            L_x = x - A_x
            out.append(L_x)

            if self.K == 1:
                return out

            T_k_minus_two = x
            T_k_minus_one = L_x
            for _ in range(2, self.K + 1):
                A_tm1 = self.propagate(
                    edge_index, x=T_k_minus_one, edge_weight=edge_weight
                )
                L_tm1 = T_k_minus_one - A_tm1
                T_k = 2.0 * L_tm1 - T_k_minus_two
                out.append(T_k)
                T_k_minus_two, T_k_minus_one = T_k_minus_one, T_k

            return out

    def message(self, x_j, edge_weight):  # type: ignore
        """
        Docstring for message

        :param self: Description
        :param x_j: features of the source node
        :param edge_weight: will be applied
        """

        return edge_weight.reshape(-1, 1) * x_j


class LittleRNN(nn.Module):
    def __init__(
        self, network_info, hidden_dim: int, dropout_rate: float, K: int, multi: int
    ):
        super().__init__()

        self.K = K  # args.K
        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        self.cheb_diff = DiffusionStep("chebyshev", self.K, self.hidden_dim)
        # self.mono_diff = DiffusionStep("monomial", self.K, self.hidden_dim)

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * multi),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim * multi, self.hidden_dim),
                )
                for _ in range(K + 1)
            ]
        )
        self.lns = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(K + 1)])

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
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=N)  # type: ignore

        for i in range(len(self.mlps)):
            x = self.lns[i](x)
            x = self.mlps[i](x)
            x = x + torch.sum(
                torch.stack(self.cheb_diff.forward(x, edge_index, edge_weight), dim=1),
                dim=1,
            )

        out = F.layer_norm(x, x.shape)
        out = self.decoder(out)

        dummy_loss = torch.tensor(0)

        return F.log_softmax(out, dim=-1), dummy_loss
