"""Adaptation of the pinned MIT-licensed tunedGNN MPNN implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


_ARCHITECTURES = {"gcn": "gcn", "mlp": "mlp", "graphsage": "sage", "sage": "sage", "gat": "gat"}


class ReferenceModel(nn.Module):
    """The tunedGNN stack, with ``nn.Linear`` blocks for the explicit MLP baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 3,
        dropout: float = 0.5,
        normalization: str = "none",
        residual: bool = False,
        pre_linear: bool = False,
        heads: int = 1,
        architecture: str = "gcn",
    ) -> None:
        super().__init__()
        architecture_key = architecture.strip().lower()
        if architecture_key not in _ARCHITECTURES:
            raise ValueError(
                f"Unknown tunedgnn architecture {architecture!r}; choose gcn, mlp, graphsage, or gat"
            )
        if depth < 1:
            raise ValueError("depth must be positive")
        if normalization not in {"none", "batch", "layer"}:
            raise ValueError("normalization must be 'none', 'batch', or 'layer'")
        if heads < 1:
            raise ValueError("heads must be positive")
        if architecture_key == "gat" and heads != 1:
            raise ValueError(
                "The pinned tunedGNN stack supports heads=1 only because concatenated "
                "multi-head outputs do not match its normalization and classifier widths"
            )

        self.architecture = architecture_key
        self.dropout = dropout
        self.pre_linear = pre_linear
        self.res = residual
        self.ln = normalization == "layer"
        self.bn = normalization == "batch"

        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lin_in = nn.Linear(in_channels, hidden_channels)

        remaining_layers = depth
        if not pre_linear:
            self.local_convs.append(
                self._make_block(
                    in_channels,
                    hidden_channels,
                    heads,
                    _ARCHITECTURES[architecture_key],
                )
            )
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            remaining_layers -= 1

        for _ in range(remaining_layers):
            self.local_convs.append(
                self._make_block(
                    hidden_channels,
                    hidden_channels,
                    heads,
                    _ARCHITECTURES[architecture_key],
                )
            )
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.pred_local = nn.Linear(hidden_channels, out_channels)

    @staticmethod
    def _make_block(in_channels: int, hidden_channels: int, heads: int, kind: str) -> nn.Module:
        if kind == "gat":
            return GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=True,
                add_self_loops=False,
                bias=False,
            )
        if kind == "sage":
            return SAGEConv(in_channels, hidden_channels)
        if kind == "mlp":
            return nn.Linear(in_channels, hidden_channels)
        return GCNConv(in_channels, hidden_channels, cached=False, normalize=True)

    def reset_parameters(self) -> None:
        for module in self.local_convs:
            module.reset_parameters()
        for module in self.lins:
            module.reset_parameters()
        for module in self.lns:
            module.reset_parameters()
        for module in self.bns:
            module.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for index, block in enumerate(self.local_convs):
            if self.architecture == "mlp":
                transformed = block(x)
            else:
                transformed = block(x, edge_index)
            x = transformed + self.lins[index](x) if self.res else transformed
            if self.ln:
                x = self.lns[index](x)
            elif self.bn:
                x = self.bns[index](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.pred_local(x)
