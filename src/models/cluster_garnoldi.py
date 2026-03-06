"""Two-stage cluster-conditioned GArnoldi filter bank.

Stage 1 (ClusterStage1):
    Learn soft node cluster assignments via MinCut or MaxCut graph-cut objective,
    trained jointly with a lightweight GCN encoder-decoder.

Stage 2 (ClusterGArnoldiFilterBank):
    Freeze the learned cluster assignments.  Train ``num_clusters`` separate
    GArnoldi polynomial filters, each initialized with a different spectral
    target function (low-pass, high-pass, band-pass, …).  Per-node output is a
    soft mixture of the cluster filter outputs, weighted by cluster membership.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian

from src.models.GARNOLDI import GArnoldi_prop
from src.loading.DatasetInfo import DatasetInfo

# ---------------------------------------------------------------------------
# Diverse target filters used to initialise each cluster's GArnoldi_prop with
# a different spectral shape.  The names must match keys in GArnoldi_prop's
# internal ``func_map``.
# ---------------------------------------------------------------------------
DIVERSE_FILTER_NAMES = [
    "g_low_pass",
    "g_high_pass",
    "g_band_pass",
    "g_band_rejection",
    "g_comb",
]


# ============================= helpers ======================================


class _SimpleGCNProp(MessagePassing):
    """Single-hop GCN-style message passing (used by Stage 1)."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, norm=edge_weight)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# ============================= Stage 1 =====================================


class ClusterStage1(nn.Module):
    """Learn soft cluster assignments with a MinCut / MaxCut objective.

    ``forward`` returns ``(logits, cluster_loss)`` where ``cluster_loss``
    combines:
        * ``cut_sign * tr(Q^T L Q)`` — MinCut (sign = +1) or MaxCut (sign = −1)
        * orthogonality regularisation ``||Q^T Q / N − I||_F``
    scaled by ``loss_lambda``.
    """

    def __init__(
        self,
        network_info: DatasetInfo,
        hidden_dim: int,
        dropout_rate: float,
        num_clusters: int,
        loss_lambda: float,
        cut_type: str = "maxcut",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate
        self.num_clusters = num_clusters
        self.loss_lambda = loss_lambda
        self.cut_type = cut_type

        assert cut_type in {
            "mincut",
            "maxcut",
        }, f"cut_type must be 'mincut' or 'maxcut', got '{cut_type}'"
        self._cut_sign = 1.0 if cut_type == "mincut" else -1.0

        # Soft cluster assignments (raw logits; softmax applied in forward)
        self.clusters = nn.Parameter(torch.randn(network_info.N, num_clusters) * 0.01)

        # Lightweight encoder → 1-hop GCN → decoder
        self.encoder = nn.Linear(network_info.num_features, hidden_dim)
        self.prop = _SimpleGCNProp()
        self.mid = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, network_info.num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    # ------------------------------------------------------------------
    def get_cluster_logits(self) -> torch.Tensor:
        """Return the raw cluster logits (detached, CPU) for Stage 2."""
        return self.clusters.data.detach().cpu()

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        N = x.shape[0]

        # --- classification head ---
        x = self.dropout(F.leaky_relu(self.encoder(x)))

        edge_index_norm, edge_weight = gcn_norm(
            edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        )
        x = self.prop(x, edge_index_norm, edge_weight)
        x = self.dropout(F.leaky_relu(self.mid(x)))
        logits = self.decoder(x)

        # --- cluster loss ---
        Q = F.softmax(self.clusters, dim=-1)  # (N, num_clusters)

        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index_norm, edge_weight, num_nodes=N
        )
        lap = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight_lap,
            (N, N),
            device=x.device,
            dtype=x.dtype,
        ).coalesce()

        Lc = torch.sparse.mm(lap, Q)  # (N, c)
        qtLq = Q.t() @ Lc  # (c, c)
        cut_value = torch.trace(qtLq)

        # Orthogonality: encourage Q^T Q / N ≈ I  (prevents cluster collapse)
        QtQ = Q.t() @ Q  # (c, c)
        I = torch.eye(self.num_clusters, device=x.device, dtype=x.dtype)
        ortho_loss = torch.norm(QtQ / N - I, p="fro")

        cluster_loss = (self._cut_sign * cut_value + ortho_loss) * self.loss_lambda

        return logits, cluster_loss


# ============================= Stage 2 =====================================


class ClusterGArnoldiFilterBank(nn.Module):
    """Per-cluster GArnoldi polynomial filter bank.

    Each cluster owns a separate ``GArnoldi_prop`` initialised with a *different*
    target filter function so the polynomial coefficients start at diverse
    spectral shapes.  Per-node output is a soft mixture (weighted by frozen
    cluster assignments from Stage 1) of the ``num_clusters`` filter outputs.

    ``forward`` returns ``(logits, zero_loss)`` so the interface matches
    ``ClusterStage1``.
    """

    def __init__(
        self,
        network_info: DatasetInfo,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
        num_clusters: int,
        pretrained_clusters: torch.Tensor,
        Init: str = "Chebyshev",
        alpha: float = 0.1,
        Vandermonde: bool = False,
        homophily: bool = False,
    ):
        super().__init__()

        self.K = K
        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.num_clusters = num_clusters
        self.dropout_rate = dropout_rate
        self.homophily = homophily

        # Frozen cluster assignments from Stage 1 (raw logits)
        self.register_buffer("cluster_logits", pretrained_clusters)  # (N, c)

        # Encoder
        self.encoder = nn.Linear(network_info.num_features, hidden_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Per-cluster GArnoldi filters — each initialised with a different
        # spectral target function.
        lower = -0.9 if homophily else 0.0001
        upper = 0.9 if homophily else 2.0

        self.filter_bank = nn.ModuleList()
        for c in range(num_clusters):
            fname = DIVERSE_FILTER_NAMES[c % len(DIVERSE_FILTER_NAMES)]
            self.filter_bank.append(
                GArnoldi_prop(
                    K=K,
                    alpha=alpha,
                    Init=Init,
                    nameFunc=fname,
                    homophily=homophily,
                    Vandermonde=Vandermonde,
                    lower=lower,
                    upper=upper,
                )
            )

        # Post-filter processing
        self.post_filter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        # Decoder
        self.decoder = nn.Linear(hidden_dim, network_info.num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    # ------------------------------------------------------------------
    def get_filter_coefficients(self) -> dict[str, list[float]]:
        """Return learned polynomial coefficients for inspection."""
        out: dict[str, list[float]] = {}
        for c, filt in enumerate(self.filter_bank):
            label = DIVERSE_FILTER_NAMES[c % len(DIVERSE_FILTER_NAMES)]
            out[f"cluster_{c} ({label})"] = filt.temp.data.detach().cpu().tolist()
        return out

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Encode
        x = self.dropout(F.leaky_relu(self.encoder(x)))
        x = self.encoder_norm(x)

        # Run each cluster's polynomial filter
        filtered = [filt(x, edge_index) for filt in self.filter_bank]

        # Stack → (N, num_clusters, H)
        filtered_stack = torch.stack(filtered, dim=1)

        # Soft mixture weighted by frozen cluster assignments
        Q = F.softmax(self.cluster_logits, dim=-1)  # (N, num_clusters)
        # (N, 1, c) @ (N, c, H) → (N, 1, H) → (N, H)
        out = torch.bmm(Q.unsqueeze(1), filtered_stack).squeeze(1)

        # Post-process + decode
        out = self.post_filter(out)
        logits = self.decoder(out)

        return logits, torch.tensor(0.0, device=logits.device)
