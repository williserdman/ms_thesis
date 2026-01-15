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

        if prop_type == "mlp":
            self.basis = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LeakyReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(K + 1)
                ]
            )

    def forward(
        self, x, edge_index, edge_weight, old_info: Optional[torch.Tensor] = None
    ) -> list[torch.Tensor]:
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

        elif self.prop_type == "mlp":
            h = x  # (N, H)
            h = torch.concat([h, old_info[:, 0, :]], dim=-1)  # (N, 2H)
            h = F.layer_norm(h, h.shape)
            h = self.basis[0](h)
            out = [h]
            for i in range(1, self.K + 1):
                h = torch.concat([h, old_info[:, 0, :]], dim=-1)  # (N, 2H)
                h = F.layer_norm(h, h.shape)
                h = self.basis[i](h)
                h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
                out.append(h)
            return out

    def message(self, x_j, edge_weight):
        """
        Docstring for message

        :param self: Description
        :param x_j: features of the source node
        :param edge_weight: will be applied
        """

        return edge_weight.reshape(-1, 1) * x_j


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, K, dprate, num_heads):
        super(AttentionBlock, self).__init__()
        """
        Docstring for __init__

        :param self: Description
        :param hidden_dim: will be used for both input and output
        :param max_hops: number of hops
        :param num_heads: number of heads, size of head dim == hidden_dim // num_heads
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim % num_heads == 0"

        self.head_dim = hidden_dim // num_heads
        self.K = K

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(K + 1)
            ]
        )

        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)

        self.fflm1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fflrelu = nn.LeakyReLU()
        self.fflm2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.dropout = nn.Dropout(dprate)

        self.B = nn.Parameter(torch.ones(1, self.K + 1))
        self.head_bias = nn.Parameter(torch.ones(num_heads, self.K + 1))

    def reset_parameters(self):
        """
        resets parameters for this module

        :param self: Description
        """

        self.fflm1.reset_parameters()
        self.fflm2.reset_parameters()
        with torch.no_grad():
            self.local_alpha.data.fill_(1.0 / (self.K + 1))

    def forward(self, N: int, H: int, tokens: torch.Tensor):
        tokens = torch.stack(
            [layer(tokens[:, idx, :]) for idx, layer in enumerate(self.linear_layers)],
            dim=1,
        )

        Qs = (
            self.W_Q(tokens.reshape(-1, H))
            .reshape(N, self.K + 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (N, heads, K+1, d_head)
        Ks = (
            self.W_K(tokens.reshape(-1, H))
            .reshape(N, self.K + 1, self.num_heads, self.head_dim)
            .permute(0, 2, 3, 1)
        )  # (N, heads, d_head, K+1)
        Vs = tokens.reshape(N, self.K + 1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (N, heads, K+1, head_dim)

        score_logit = (Qs @ Ks) / self.head_dim**0.5  # (N, heads, K+1, K+1)
        scores = torch.tanh(score_logit)
        scores = self.dropout(scores)

        bias = self.B * self.head_bias  # (num_heads, K+1)
        Vs = Vs * bias.reshape(1, self.num_heads, self.K + 1, 1)

        out = (
            (scores @ Vs).permute(0, 2, 1, 3).reshape(N, self.K + 1, -1)
        )  # (N, heads, K+1, head_dim) -> (N, K+1, heads, head_dim) -> (N, K+1, H)

        out = F.layer_norm(out + tokens, out.shape)
        out = self.dropout(out)

        return self.fflm2(self.fflrelu(self.fflm1(out)))


class DiffusedAttention(nn.Module):
    def __init__(self, dataset, args: MyArgs, attn_layers=2):
        super(DiffusedAttention, self).__init__()

        if args.hidden <= 32:
            num_heads = 4

        self.K = 2  # args.K
        self.hidden_dim = args.hidden_dim
        self.num_classes = dataset.num_classes
        self.dprate = args.dprate

        self.encoder = nn.Linear(dataset.num_features, self.hidden_dim)

        # self.cheb_diff = DiffusionStep("chebyshev", self.K, self.hidden_dim)
        self.mono_diff = DiffusionStep("monomial", self.K, self.hidden_dim)

        num_iters = 4
        self.attn_layers = [
            AttentionBlock(self.hidden_dim, self.K, self.dprate, 4)
            for _ in range(num_iters)
        ]
        # self.poly_attn = AttentionBlock(self.hidden_dim, self.K, self.dprate, 4)
        # self.poly_attn2 = AttentionBlock(self.hidden_dim, self.K, self.dprate, 4)

        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)

        self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        [al.reset_parameters() for al in self.attn_layers]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)

        N, H = x.shape

        # Pre-compute adjacency
        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        )
        # Laplacian
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=N)

        for idx, attn_l in enumerate(self.attn_layers):
            # compute diffused messages and stack into (N, K+1, H)
            mono_msgs = self.mono_diff(x, edge_index, edge_weight)
            mono_tokens = torch.stack(mono_msgs, dim=1)  # (N, K+1, H)
            tokens = F.layer_norm(mono_tokens, mono_tokens.shape)
            # print(tokens.shape)

            out = attn_l(N, H, tokens)
            out = torch.sum(out, dim=1)
            x = out

        out = F.layer_norm(x, x.shape)
        out = self.decoder(out)

        dummy_loss = torch.tensor(0)

        return F.log_softmax(out, dim=-1), dummy_loss
