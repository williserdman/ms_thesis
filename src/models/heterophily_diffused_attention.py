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

        elif self.prop_type == "mlp":
            h = x  # (N, H)
            h = torch.concat([h, old_info[:, 0, :]], dim=-1)  # type: ignore (N, 2H)
            h = F.layer_norm(h, h.shape)
            h = self.basis[0](h)
            out = [h]
            for i in range(1, self.K + 1):
                h = torch.concat([h, old_info[:, 0, :]], dim=-1)  # type: ignore (N, 2H)
                h = F.layer_norm(h, h.shape)
                h = self.basis[i](h)
                h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
                out.append(h)
            return out

    def message(self, x_j, edge_weight):  # type: ignore
        """
        Docstring for message

        :param self: Description
        :param x_j: features of the source node
        :param edge_weight: will be applied
        """

        return edge_weight.reshape(-1, 1) * x_j


class PolyAttn(nn.Module):
    def __init__(self, hidden_dim, num_heads, K, dropout_rate, q=0.25, multi=4.0):
        super(PolyAttn, self).__init__()
        self.K = K + 1
        self.norm = nn.LayerNorm(hidden_dim)
        self.n_head = num_heads
        self.multi = multi
        self.d_head = hidden_dim // num_heads

        self.token_wise_network = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, int(hidden_dim * self.multi)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim * self.multi), hidden_dim),
                )
                for _ in range(self.K)
            ]
        )

        self.W_Q = nn.Linear(hidden_dim, self.n_head * self.d_head, bias=False)
        self.W_K = nn.Linear(hidden_dim, self.n_head * self.d_head, bias=False)

        self.bias_scale = nn.Parameter(torch.ones(self.n_head, self.K))
        self.bias = torch.tensor([((j + 1) ** q) ** (-1) for j in range(self.K)])
        self.register_buffer("bias_buffer", self.bias)

        self.dprate = dropout_rate
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.token_wise_network:
            layer[0].reset_parameters()  # type: ignore
            layer[2].reset_parameters()  # type: ignore
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()

    def forward(self, src):
        batch_size = src.shape[0]
        origin_src = src
        src = self.norm(src)
        token = src
        value = src
        token = torch.stack(
            [
                layer(token[:, idx, :])
                for idx, layer in enumerate(self.token_wise_network)
            ],
            dim=1,
        )
        query = self.W_Q(token)
        key = self.W_K(token)
        q_heads = query.view(batch_size, self.K, self.n_head, self.d_head).transpose(
            1, 2
        )  # [n,n_head,k,d_head]
        k_heads = key.view(batch_size, self.K, self.n_head, self.d_head).transpose(1, 2)
        v_heads = value.view(batch_size, self.K, self.n_head, -1).transpose(1, 2)
        attention_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (
            self.d_head**0.5
        )
        attention_scores = torch.tanh(attention_scores)
        attn_mask = torch.einsum("hk,k->hk", self.bias_scale, self.bias_buffer)
        attention_scores = torch.einsum("nhij,hj->nhij", attention_scores, attn_mask)
        attention_scores = F.dropout(
            attention_scores, p=self.dprate, training=self.training
        )
        context_heads = torch.matmul(attention_scores, v_heads)
        context_sequence = (
            context_heads.transpose(1, 2).contiguous().view(batch_size, self.K, -1)
        )
        src = F.dropout(context_sequence, p=self.dprate, training=self.training)
        src = src + origin_src
        return src


class FFNNetwork(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super(FFNNetwork, self).__init__()
        self.lin1 = nn.Linear(hidden_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(ffn_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x


class FFN(nn.Module):
    def __init__(self, hidden_dim, dropout_rate, d_ffn=None):
        super(FFN, self).__init__()
        if d_ffn is None:
            d_ffn = hidden_dim * 4
        self.dropout = dropout_rate
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_net = FFNNetwork(hidden_dim, d_ffn)

    def forward(self, src):
        origin_src = src
        src = self.ffn_norm(src)
        src = self.ffn_net(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        src = src + origin_src
        return src


class PolyFormerBlock(nn.Module):
    def __init__(self, hidden_dim, K, dropout_rate, num_heads):
        super(PolyFormerBlock, self).__init__()
        self.attnmodule = PolyAttn(hidden_dim, num_heads, K, dropout_rate)
        self.ffnmodule = FFN(hidden_dim, dropout_rate)

    def reset_parameters(self):
        self.attnmodule.reset_parameters()
        self.ffnmodule.ffn_net.reset_parameters()

    def forward(self, src):
        src = self.attnmodule(src)
        src = self.ffnmodule(src)
        return src


class DiffusedAttention(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
    ):
        super().__init__()

        self.K = K  # args.K
        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        self.cheb_diff = DiffusionStep("chebyshev", self.K, self.hidden_dim)
        # self.mono_diff = DiffusionStep("monomial", self.K, self.hidden_dim)

        num_iters = 4
        self.attn_layers = nn.ModuleList(
            [
                PolyFormerBlock(self.hidden_dim, self.K, self.dropout_rate, 4)
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
        # edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=N)  # type: ignore

        # compute diffused messages and stack into (N, K+1, H)
        mono_msgs = self.cheb_diff(x, edge_index, edge_weight)
        mono_tokens = torch.stack(mono_msgs, dim=1)  # (N, K+1, H)
        tokens = F.layer_norm(mono_tokens, (H,))
        # print(tokens.shape)

        for _, attn_l in enumerate(self.attn_layers):
            tokens = attn_l(tokens)
            # tokens = F.layer_norm(out, out.shape) # PolyFormer block handles normalization internally

        # Pool over K
        out = torch.sum(tokens, dim=1)  # type: ignore
        out = F.layer_norm(out, (H,))
        out = self.decoder(out)

        dummy_loss = torch.tensor(0)

        return F.softmax(out, dim=-1), dummy_loss
