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
    Performs X' = P @ X where P is the proppagation matrix.
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
    ) -> list[torch.Tensor]:  # type: ignore
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


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, K, dprate, num_heads, multi):
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

        assert hidden_dim % num_heads == 0, f"{hidden_dim} % {num_heads} == 0"

        self.head_dim = hidden_dim // num_heads
        self.K = K

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * multi),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim * multi, hidden_dim),
                )
                for _ in range(K + 1)
            ]
        )

        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dprate)

        self.B = nn.Parameter(torch.ones(1, self.K + 1))
        self.head_bias = nn.Parameter(torch.ones(num_heads, self.K + 1))

    def reset_parameters(self):
        with torch.no_grad():
            self.local_alpha.data.fill_(1.0 / (self.K + 1))  # type: ignore

    def forward(self, N: int, H: int, tokens: torch.Tensor, cluster_bias: torch.Tensor):
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

        # (N, heads, K+1, K+1)
        # Note: Ensure Ks is transposed if it isn't already: (Qs @ Ks.transpose(-1, -2))
        score_logit = (Qs @ Ks) / self.head_dim**0.5
        scores = torch.tanh(score_logit)
        scores = self.dropout(scores)

        # 1. Prepare Static Bias (Global)
        # self.B: (K+1) -> (num_heads, K+1)
        static_bias = self.B * self.head_bias
        # Broadcast to: (1, num_heads, K+1, 1) to match Vs
        static_bias = static_bias.reshape(1, self.num_heads, self.K + 1, 1)

        # 2. Prepare Cluster Bias (Node-Specific)
        # Assuming cluster_bias is (N, K+1). Reshape to broadcast over heads and dim.
        # Target: (N, 1, K+1, 1)
        N = scores.shape[0]  # Get batch size
        cluster_bias_broadcast = cluster_bias.view(N, 1, self.K + 1, 1)

        # 3. Apply Combined Bias to Values
        # We add the cluster bias to the static bias. This shifts the filter coefficients
        # for each node based on its cluster.
        # Result: (N, heads, K+1, head_dim)
        Vs = Vs * (static_bias + cluster_bias_broadcast)

        out = (
            (scores @ Vs).permute(0, 2, 1, 3).reshape(N, self.K + 1, -1)
        )  # (N, heads, K+1, head_dim) -> (N, K+1, heads, head_dim) -> (N, K+1, H)

        out = F.layer_norm(out + tokens, out.shape[-1:])
        out = self.dropout(out)

        return out


class DiffusedAttention(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
        multi: int,
        num_iters,
        num_clusters: int,
        num_heads_main: int,
        loss_lambda: float,
    ):
        super().__init__()

        self.K = K  # args.K
        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate
        self.num_clusters = num_clusters
        self.loss_lambda = loss_lambda

        self.clusters = nn.Parameter(
            torch.zeros(network_info.N, self.num_clusters)
        )  # (N, c)

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

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
