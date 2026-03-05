import numpy as np
import torch
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.models._arnoldi import *


class GArnoldi_prop(MessagePassing):
    """
    Propagation class for GPR_GNN
    """

    def __init__(
        self,
        K,
        alpha,
        Init,
        nameFunc,
        homophily,
        Vandermonde,
        lower,
        upper,
        Gamma=None,
        bias=True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.homophily = homophily
        self.Vandermonde = Vandermonde
        self.nameFunc = nameFunc
        self.lower = lower
        self.upper = upper

        valid_inits = [
            "Monomial",
            "Chebyshev",
            "Legendre",
            "Jacobi",
            "PPR",
            # "SChebyshev",
            "WS",
        ]
        if Init not in valid_inits:
            raise ValueError(f"Init must be one of {valid_inits}")

        func_map = {
            "g_0": g_0,
            "g_1": g_1,
            "g_2": g_2,
            "g_3": g_3,
            "g_4": g_4,
            "g_band_rejection": g_band_rejection,
            "g_band_pass": g_band_pass,
            "g_low_pass": g_low_pass,
            "g_high_pass": g_high_pass,
            "g_comb": g_comb,
        }
        func = func_map.get(nameFunc, g_fullRWR)

        # Apply coefficients based on Initialization type
        if Init in {"Monomial", "Chebyshev", "Legendre", "Jacobi"}:
            self.coeffs = compare_fit_panelA(
                func, Init, Vandermonde, self.K, self.lower, self.upper
            )
            self.coeffs = filter_jackson(self.coeffs)
            TEMP = self.coeffs

        elif Init == "SChebyshev":
            raise NotImplemented
            self.coeffs = compare_fit_panelA(func, Init, self.K)
            TEMP = self.coeffs

        elif Init == "PPR":
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K

        elif Init == "WS":
            TEMP = Gamma

        # Ensure TEMP is a tensor before making it a Parameter
        if not isinstance(TEMP, torch.Tensor):
            TEMP = torch.tensor(TEMP, dtype=torch.float32)

        self.temp = Parameter(TEMP)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)

        if self.Init == "Monomial":
            self.temp.data = torch.tensor(
                m_polynomial_zeros(self.lower, self.upper, self.K)
            )
        elif self.Init == "Chebyshev":
            self.temp.data = torch.tensor(
                t_polynomial_zeros(self.lower, self.upper, self.K)
            )
        elif self.Init == "Legendre":
            self.temp.data = torch.tensor(p_polynomial_zeros(self.K))
        elif self.Init == "Jacobi":
            self.temp.data = torch.tensor(j_polynomial_zeros(self.K, 0, 1))
        else:
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        edge_index1, norm1 = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            dtype=x.dtype,
            num_nodes=x.size(self.node_dim),
        )

        # 2I - L
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0, num_nodes=x.size(self.node_dim)
        )

        hidden = self.temp[self.K - 1] * x

        for k in range(self.K - 2, -1, -1):
            if self.homophily:
                x = self.propagate(edge_index, x=x, norm=norm)
            else:
                x = self.propagate(edge_index1, x=x, norm=norm1)

            gamma = self.temp[k]
            x = x + gamma * hidden

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f"{self.__class__.__name__}(K={self.K}, temp={self.temp})"


class GARNOLDI(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GARNOLDI, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        # if args.Arnoldippnp == "PPNP":
        #     self.prop1 = APPNP(args.K, args.alpha)
        if args.Arnoldippnp == "GArnoldi_prop":
            self.prop1 = GArnoldi_prop(
                args.K,
                args.alpha,
                args.ArnoldiInit,
                args.FuncName,
                args.homophily,
                args.Vandermonde,
                args.lower,
                args.upper,
                args.Gamma,
            )

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.FuncName = args.FuncName

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
