"""Fixed GArnoldi graph filters followed by a trainable MLP."""

from collections.abc import Callable, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_undirected


def _low_pass(values: np.ndarray) -> np.ndarray:
    return np.exp(-10.0 * values**2)


_FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "g_low_pass": _low_pass,
    "g_high_pass": lambda values: 1.0 - _low_pass(values),
    "g_band_pass": lambda values: np.exp(-10.0 * (values - 1.0) ** 2),
    "g_band_rejection": lambda values: 1.0
    - np.exp(-10.0 * (values - 1.0) ** 2),
}


def _fit_filters(
    degree: int, filters: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit target responses in an Arnoldi basis on Chebyshev nodes."""
    sample_count = degree + 1
    indices = np.arange(1, sample_count + 1)
    nodes = 1.0 + np.cos((2 * indices - 1) * np.pi / (2 * sample_count))

    # The legacy GArnoldi code uses the discrete inner product mean(a * b),
    # so the constant starting vector already has unit norm.
    basis = np.ones((sample_count, degree + 1), dtype=np.float64)
    recurrence = np.zeros((degree + 1, degree), dtype=np.float64)
    for k in range(degree):
        candidate = nodes * basis[:, k]
        for j in range(k + 1):
            recurrence[j, k] = np.dot(basis[:, j], candidate) / sample_count
            candidate -= recurrence[j, k] * basis[:, j]
        recurrence[k + 1, k] = np.linalg.norm(candidate) / np.sqrt(sample_count)
        if recurrence[k + 1, k] <= np.finfo(np.float64).eps:
            raise ValueError("Arnoldi basis broke down before reaching K")
        basis[:, k + 1] = candidate / recurrence[k + 1, k]

    coefficients = np.stack(
        [np.linalg.solve(basis, _FILTERS[name](nodes)) for name in filters]
    )
    return recurrence, coefficients


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


class FixedSpectralModel(pl.LightningModule):
    """Apply fixed spectral filters to raw features, then train an MLP.

    The graph is converted to an undirected graph before constructing the
    symmetric normalized Laplacian. This makes the operator real symmetric
    even for directed input datasets. The fit uses the GArnoldi target
    responses and Chebyshev sampling scheme, but retains the fitted recurrence
    and evaluates that same polynomial instead of using the legacy malformed
    propagation. Filtered features are detached and cached for one static full
    graph; only ``predictor`` has trainable parameters.
    """

    def __init__(
        self,
        ds_info,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        dropout_rate: float = 0.5,
        weight_decay: float = 5e-4,
        K: int = 10,
        filters: Sequence[str] = ("g_low_pass", "g_high_pass"),
    ) -> None:
        super().__init__()
        if K < 0:
            raise ValueError("K must be non-negative")
        filters = tuple(filters)
        if not filters:
            raise ValueError("at least one filter is required")
        unknown = set(filters) - _FILTERS.keys()
        if unknown:
            raise ValueError(f"unknown fixed filter(s): {sorted(unknown)}")

        self.save_hyperparameters(ignore=["ds_info"])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.filters = filters

        recurrence, coefficients = _fit_filters(K, filters)
        self.register_buffer("arnoldi_h", torch.from_numpy(recurrence))
        self.register_buffer("filter_coefficients", torch.from_numpy(coefficients))
        self.register_buffer(
            "class_weights", torch.as_tensor(ds_info.class_weights).float().clone()
        )
        self.register_buffer("_cached_features", torch.empty(0), persistent=False)
        self._cache_signature = None

        input_dim = ds_info.num_features * len(filters)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, ds_info.num_classes),
        )

    @staticmethod
    def _tensor_signature(value: torch.Tensor | None):
        if value is None:
            return None
        return (
            value.data_ptr(),
            tuple(value.shape),
            value.dtype,
            value.device,
            None if value.is_inference() else value._version,
        )

    def _graph_signature(self, batch) -> tuple:
        return (
            self._tensor_signature(batch.x),
            self._tensor_signature(batch.edge_index),
            self._tensor_signature(getattr(batch, "edge_weight", None)),
            self.arnoldi_h.device,
            self.predictor[0].weight.dtype,
        )

    def _precompute(self, batch) -> torch.Tensor:
        x = batch.x.detach().to(dtype=torch.float64)
        num_nodes = x.shape[0]
        edge_index = batch.edge_index
        edge_weight = getattr(batch, "edge_weight", None)
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.shape[1], dtype=torch.float64, device=edge_index.device
            )
        else:
            edge_weight = edge_weight.detach().to(dtype=torch.float64)

        edge_index, edge_weight = to_undirected(
            edge_index, edge_weight, num_nodes=num_nodes, reduce="add"
        )
        laplacian_index, laplacian_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            num_nodes=num_nodes,
        )
        laplacian = torch.sparse_coo_tensor(
            laplacian_index,
            laplacian_weight,
            (num_nodes, num_nodes),
            dtype=torch.float64,
            device=x.device,
        ).coalesce()

        recurrence = self.arnoldi_h.to(device=x.device, dtype=torch.float64)
        coefficients = self.filter_coefficients.to(device=x.device, dtype=torch.float64)
        basis = [x]
        outputs = [coefficient[0] * x for coefficient in coefficients]
        for k in range(recurrence.shape[1]):
            candidate = torch.sparse.mm(laplacian, basis[k])
            for j in range(k + 1):
                candidate = candidate - recurrence[j, k] * basis[j]
            candidate = candidate / recurrence[k + 1, k]
            basis.append(candidate)
            for output, coefficient in zip(outputs, coefficients):
                output.add_(coefficient[k + 1] * candidate)

        dtype = self.predictor[0].weight.dtype
        return torch.cat(outputs, dim=-1).to(dtype=dtype).detach()

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        signature = self._graph_signature(batch)
        if self._cache_signature != signature:
            with torch.no_grad():
                self._cached_features = self._precompute(batch)
            self._cache_signature = signature
        logits = self.predictor(self._cached_features)
        return logits, logits.new_zeros(())

    def training_step(self, batch, batch_idx=None):
        logits, _ = self(batch)
        mask = getattr(batch, "train_mask", None)
        if mask is not None:
            logits, labels = logits[mask], batch.y[mask]
        else:
            labels = batch.y
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        self.log("train_loss", loss, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx=None) -> None:
        logits, _ = self(batch)
        mask = batch.val_mask
        loss = F.cross_entropy(logits[mask], batch.y[mask], weight=self.class_weights)
        self.log("val_loss", loss, prog_bar=True, batch_size=1)
        self.log(
            "val_accuracy",
            _accuracy(logits[mask], batch.y[mask]),
            prog_bar=True,
            batch_size=1,
        )

    def test_step(self, batch, batch_idx=None) -> None:
        logits, _ = self(batch)
        mask = batch.test_mask
        loss = F.cross_entropy(logits[mask], batch.y[mask], weight=self.class_weights)
        self.log("test_loss", loss, batch_size=1)
        self.log(
            "test_accuracy", _accuracy(logits[mask], batch.y[mask]), batch_size=1
        )

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        }
