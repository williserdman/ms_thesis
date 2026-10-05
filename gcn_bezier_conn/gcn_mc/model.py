"""GCN endpoint model used by the connectivity experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """A plain multi-layer GCN with dropout between graph convolutions."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")

        channels = [in_channels, *([hidden_channels] * (depth - 1)), out_channels]
        self.convs = nn.ModuleList(
            GCNConv(source, target, cached=False)
            for source, target in zip(channels, channels[1:])
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def build_model(model_config: dict) -> nn.Module:
    """Build a legacy GCN or the pinned tunedGNN architecture adaptation."""
    config = dict(model_config)
    family = config.pop("family", None)
    if family is None:
        config.pop("architecture", None)
        return GCN(**config)
    if family != "tunedgnn":
        raise ValueError(f"Unknown model family {family!r}")

    from .reference_models import ReferenceModel

    architecture = config.pop("architecture", None)
    if architecture is None:
        raise ValueError("tunedgnn model configuration requires an architecture")
    return ReferenceModel(architecture=architecture, **config)
