import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def _laplacian_trace_from_edges(
    edge_index_lap: torch.Tensor,
    edge_weight_lap: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    src = edge_index_lap[0]
    dst = edge_index_lap[1]
    pair_inner = (q[src] * q[dst]).sum(dim=-1)
    return (edge_weight_lap * pair_inner).sum()


class ClusterGArnoldi_prop(MessagePassing):
    def __init__(self, K, num_clusters, alpha, homophily, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.homophily = homophily

        # Shape: (num_clusters, K + 1)
        self.temp = nn.Parameter(torch.Tensor(self.num_clusters, self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # We need to break symmetry here.
        # You can initialize using your existing filter functions (g_0, g_1, etc.)
        # Here is a generic approach: PPR baseline + distinct Gaussian noise per cluster
        torch.nn.init.zeros_(self.temp)

        for c in range(self.num_clusters):
            # Base PPR Initialization
            base_temp = torch.zeros(self.K + 1)
            for k in range(self.K + 1):
                base_temp[k] = self.alpha * (1 - self.alpha) ** k
            base_temp[-1] = (1 - self.alpha) ** self.K

            # Add distinct noise to break symmetry
            noise = torch.randn(self.K + 1) * 0.1
            self.temp.data[c] = base_temp + noise

    def forward(self, x, edge_index, clusters, edge_weight=None):
        # clusters shape: (N, num_clusters) - assumed to be softmaxed already

        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        edge_index1, norm1 = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            dtype=x.dtype,
            num_nodes=x.size(0),
        )

        # Base hidden state multiplied by node-specific coefficients for K-1
        # clusters @ self.temp[:, self.K - 1] -> shape (N,)
        node_coeffs_init = (clusters @ self.temp[:, self.K - 1]).unsqueeze(-1)  # (N, 1)
        hidden = node_coeffs_init * x

        for k in range(self.K - 2, -1, -1):
            if self.homophily:
                x = self.propagate(edge_index, x=x, norm=norm)
            else:
                x = self.propagate(edge_index1, x=x, norm=norm1)

            # Node-specific coefficients at step k
            node_coeffs = (clusters @ self.temp[:, k]).unsqueeze(-1)  # (N, 1)
            x = x + node_coeffs * hidden

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class ClusteredArnoldiModel(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
        num_clusters: int,
        alpha: float = 0.1,  # You can add this to Optuna or keep it fixed
        homophily: bool = True,  # You can add this to Optuna or keep it fixed
        cut_type: str = "mincut",
        loss_lambda: float = 1.0,  # Used to scale the cut loss
        **kwargs  # Catches num_iters, multi, num_heads_main if you add them later
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.cut_type = cut_type
        self.dropout_rate = dropout_rate
        self.loss_lambda = loss_lambda

        # Soft cluster assignment embeddings
        # network_info.N gives the total number of nodes in the graph
        self.clusters_logits = nn.Parameter(
            torch.zeros(network_info.N, self.num_clusters)
        )

        self.lin1 = nn.Linear(network_info.num_features, hidden_dim)

        # Instantiate the cluster-aware propagation layer
        self.prop = ClusterGArnoldi_prop(
            K=K, num_clusters=num_clusters, alpha=alpha, homophily=homophily
        )

        self.lin2 = nn.Linear(hidden_dim, network_info.num_classes)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        N = x.shape[0]

        # 1. Compute Clusters
        clusters = F.softmax(self.clusters_logits, dim=-1)  # Q matrix (N, c)

        # 2. Compute Cut Loss (MinCut or MaxCut)
        edge_index_lap, edge_weight_lap = get_laplacian(edge_index, num_nodes=N)
        edge_weight_lap = torch.as_tensor(edge_weight_lap, device=clusters.device, dtype=clusters.dtype)  # type: ignore[arg-type]
        cut_size = _laplacian_trace_from_edges(edge_index_lap, edge_weight_lap, clusters)

        # Orthogonality constraint to prevent collapse
        I = torch.eye(self.num_clusters, device=x.device, dtype=x.dtype)
        ortho_loss = torch.norm(
            clusters.t() @ clusters - (N / self.num_clusters) * I, p="fro"
        )

        if self.cut_type == "mincut":
            cluster_loss = cut_size + ortho_loss
        elif self.cut_type == "maxcut":
            cluster_loss = -cut_size + ortho_loss
        else:
            raise ValueError("cut_type must be 'mincut' or 'maxcut'")

        # 3. Forward Pass through GNN
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Pass the soft cluster assignments into the propagation layer
        x = self.prop(x, edge_index, clusters)

        logits = self.lin2(x)

        # Scale the cluster loss by lambda before returning
        return logits, self.loss_lambda * cluster_loss
