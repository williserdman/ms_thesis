"""Linear-gamma spectral GNN head (GPR-GNN lineage) for the mode-connectivity PoCs.

Why this exists
---------------
The repo's headline model (`DiffusedAttention`) buries its filter coefficients
inside a 4-block nonlinear PolyAttn stack, so it has *no* clean linear
coefficient axis. The portfolio review flagged that ~12/20 ideas assume exactly
such an axis. This module provides it: an explicit, ~(K+1)-dimensional, linear
coefficient vector `gamma` with

    Z = sum_{k=0..K} gamma_k * phi_k(op) @ H0 ,   H0 = MLP(X),

where `op` is the symmetric-normalized adjacency (domain="adj") or Laplacian
(domain="lap"), and phi_k is the monomial (P^k) or Chebyshev (T_k) basis. Because
Z is *linear in gamma*, interpolating gamma is a genuine morph of the filter
response g(lambda) = sum_k gamma_k phi_k(lambda) -- which is what the path /
barrier experiments need.

Contract: matches the rest of the repo -- `forward(batch) -> (logits, inner_loss)`,
a `pl.LightningModule`, built from a `DatasetInfo`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.utils import add_self_loops


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == y).sum().item()) / max(int(y.numel()), 1)


def sym_norm_adj(edge_index: torch.Tensor, num_nodes: int, device) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2.

    Returned as a coalesced sparse COO tensor. Eigenvalues lie in (-1, 1].
    """
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=device).scatter_add_(
        0, row, torch.ones(row.size(0), device=device)
    )
    dinv_sqrt = deg.pow(-0.5)
    dinv_sqrt[torch.isinf(dinv_sqrt)] = 0.0
    vals = dinv_sqrt[row] * dinv_sqrt[col]
    return torch.sparse_coo_tensor(
        edge_index, vals, (num_nodes, num_nodes), device=device
    ).coalesce()


def ppr_init(K: int, alpha: float) -> torch.Tensor:
    """GPR-GNN personalized-PageRank coefficient init: gamma_k = alpha (1-alpha)^k."""
    g = alpha * (1.0 - alpha) ** torch.arange(K + 1, dtype=torch.float)
    g[K] = (1.0 - alpha) ** K
    return g


class LinearSpectralGNN(pl.LightningModule):
    """Explicit linear-gamma spectral filter GNN.

    Args:
        ds_info: DatasetInfo (num_features, num_classes, class_weights, ...).
        hidden_dim: width of the feature MLP.
        K: polynomial order; the coefficient axis gamma has K+1 entries.
        basis: "cheb" (Chebyshev T_k) or "mono" (monomial P^k).
        domain: "adj" (normalized adjacency) or "lap" (normalized Laplacian).
        gamma_init: "ppr" | "ones" | "random".
        alpha: PPR teleport for gamma_init="ppr".
        dropout_rate, learning_rate, weight_decay: training knobs.
        freeze_mlp: if True, only gamma is trainable (used by subspace experiments).
    """

    def __init__(
        self,
        ds_info,
        hidden_dim: int = 64,
        K: int = 10,
        basis: str = "cheb",
        domain: str = "adj",
        gamma_init: str = "ppr",
        alpha: float = 0.1,
        dropout_rate: float = 0.5,
        learning_rate: float = 1e-2,
        weight_decay: float = 5e-4,
        freeze_mlp: bool = False,
        **kwargs,
    ):
        super().__init__()
        # class_weights is a tensor; exclude from hparam pickling cleanly
        self.save_hyperparameters(ignore=["ds_info"])

        self.num_features = ds_info.num_features
        self.num_classes = ds_info.num_classes
        self.K = K
        self.basis = basis
        self.domain = domain
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.lin1 = nn.Linear(self.num_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, self.num_classes)

        if gamma_init == "ppr":
            g0 = ppr_init(K, alpha)
        elif gamma_init == "ones":
            g0 = torch.ones(K + 1) / (K + 1)
        elif gamma_init == "random":
            g0 = torch.randn(K + 1) * 0.1
        else:
            raise ValueError(f"unknown gamma_init {gamma_init!r}")
        self.gamma = nn.Parameter(g0)

        if freeze_mlp:
            for p in self.lin1.parameters():
                p.requires_grad_(False)
            for p in self.lin2.parameters():
                p.requires_grad_(False)

        cw = torch.as_tensor(ds_info.class_weights, dtype=torch.float)
        self.register_buffer("class_weights", cw)
        self._op_cache = None  # (num_nodes, sparse op) cache for the single graph

    # ---- operator application ----
    def _apply_op(self, adj: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Apply the chosen spectral operator once to node features H."""
        AH = torch.sparse.mm(adj, H)
        if self.domain == "adj":
            return AH
        if self.domain == "lap":
            if self.basis == "cheb":
                # rescaled Laplacian L~ = L_hat - I = -A_hat (eigs -> [-1,1])
                return -AH
            # monomial Laplacian L_hat = I - A_hat
            return H - AH
        raise ValueError(f"unknown domain {self.domain!r}")

    def _propagate(self, adj: torch.Tensor, H0: torch.Tensor) -> torch.Tensor:
        """Z = sum_k gamma_k phi_k(op) H0 for the chosen basis."""
        if self.basis == "mono":
            Z = self.gamma[0] * H0
            Hk = H0
            for k in range(1, self.K + 1):
                Hk = self._apply_op(adj, Hk)
                Z = Z + self.gamma[k] * Hk
            return Z
        if self.basis == "cheb":
            T0 = H0
            Z = self.gamma[0] * T0
            if self.K >= 1:
                T1 = self._apply_op(adj, H0)
                Z = Z + self.gamma[1] * T1
                Tprev, Tcur = T0, T1
                for k in range(2, self.K + 1):
                    Tnext = 2.0 * self._apply_op(adj, Tcur) - Tprev
                    Z = Z + self.gamma[k] * Tnext
                    Tprev, Tcur = Tcur, Tnext
            return Z
        raise ValueError(f"unknown basis {self.basis!r}")

    def _get_op(self, batch) -> torch.Tensor:
        num_nodes = batch.x.size(0)
        if self._op_cache is None or self._op_cache[0] != num_nodes:
            adj = sym_norm_adj(batch.edge_index, num_nodes, batch.x.device)
            self._op_cache = (num_nodes, adj)
        return self._op_cache[1]

    def forward(self, batch):
        adj = self._get_op(batch)
        h = F.dropout(batch.x, p=self.dropout_rate, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        H0 = self.lin2(h)
        logits = self._propagate(adj, H0)
        inner_loss = logits.new_zeros(())  # keep the (logits, inner_loss) contract
        return logits, inner_loss

    # ---- lightning steps ----
    def _loss(self, logits, y, mask):
        return F.cross_entropy(logits[mask], y[mask], weight=self.class_weights)

    def training_step(self, batch, batch_idx=0):
        logits, inner = self.forward(batch)
        mask = batch.train_mask
        loss = self._loss(logits, batch.y, mask) + inner
        self.log("train_loss", loss, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx=0):
        logits, _ = self.forward(batch)
        mask = batch.val_mask
        loss = self._loss(logits, batch.y, mask)
        acc = _accuracy(logits[mask], batch.y[mask])
        self.log("val_loss", loss, prog_bar=True, batch_size=1)
        self.log("val_accuracy", acc, prog_bar=True, batch_size=1)

    def test_step(self, batch, batch_idx=0):
        logits, _ = self.forward(batch)
        mask = batch.test_mask
        loss = self._loss(logits, batch.y, mask)
        acc = _accuracy(logits[mask], batch.y[mask])
        self.log("test_loss", loss, batch_size=1)
        self.log("test_accuracy", acc, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
