"""Lightning wrappers for the two-stage cluster-conditioned GArnoldi models."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

from src.models.cluster_garnoldi import (
    ClusterStage1,
    ClusterGArnoldiFilterBank,
    DIVERSE_FILTER_NAMES,
)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    total = int(y.numel())
    return float(correct) / max(total, 1)


# -----------------------------------------------------------------------
# Mixin — shared train / val / test step logic (no __init__)
# -----------------------------------------------------------------------


class _ClusterTrainingMixin:
    """Shared training/validation/test logic for both stages.

    Expects the concrete class to set ``self.learning_rate`` and ``self.cse``
    (CrossEntropyLoss) before the first forward pass.
    """

    learning_rate: float
    cse: nn.CrossEntropyLoss

    def __init__(self, *args, **kwargs):
        # Transparent passthrough so that MRO super().__init__ chains work.
        super().__init__(*args, **kwargs)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)  # type: ignore[attr-defined]

    def training_step(self, batch):
        logits, inner_loss = self.forward(batch)  # type: ignore[attr-defined]
        if hasattr(batch, "train_mask") and batch.train_mask is not None:
            mask = batch.train_mask
            loss = self.cse(logits[mask], batch.y[mask]) + inner_loss
        else:
            loss = self.cse(logits, batch.y) + inner_loss
        self.log("train_loss", loss)  # type: ignore[attr-defined]
        return loss

    def validation_step(self, batch):
        logits, _ = self.forward(batch)  # type: ignore[attr-defined]
        mask = batch.val_mask
        loss = self.cse(logits[mask], batch.y[mask])
        acc = _accuracy(logits[mask], batch.y[mask])
        self.log("val_loss", loss, prog_bar=True)  # type: ignore[attr-defined]
        self.log("val_accuracy", acc, prog_bar=True)  # type: ignore[attr-defined]

    def test_step(self, batch):
        logits, _ = self.forward(batch)  # type: ignore[attr-defined]
        mask = batch.test_mask
        loss = self.cse(logits[mask], batch.y[mask])
        acc = _accuracy(logits[mask], batch.y[mask])
        self.log("test_loss", loss)  # type: ignore[attr-defined]
        self.log("test_accuracy", acc)  # type: ignore[attr-defined]

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.AdamW(
                self.parameters(),  # type: ignore[attr-defined]
                lr=self.learning_rate,
            )
        }


# -----------------------------------------------------------------------
# Stage 1 Lightning Module
# -----------------------------------------------------------------------


class MyClusterStage1(
    _ClusterTrainingMixin,
    ClusterStage1,
    pl.LightningModule,
):
    """Lightning module for Stage 1: cluster learning with MinCut / MaxCut."""

    def __init__(self, ds_info, learning_rate, **kwargs):
        super().__init__(ds_info, **kwargs)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.cse = nn.CrossEntropyLoss(torch.tensor(ds_info.class_weights))


# -----------------------------------------------------------------------
# Stage 2 Lightning Module
# -----------------------------------------------------------------------


class MyClusterFilterBank(
    _ClusterTrainingMixin,
    ClusterGArnoldiFilterBank,
    pl.LightningModule,
):
    """Lightning module for Stage 2: per-cluster GArnoldi filter training."""

    def __init__(self, ds_info, learning_rate, pretrained_clusters, **kwargs):
        super().__init__(
            ds_info,
            pretrained_clusters=pretrained_clusters,
            **kwargs,
        )
        # Exclude the (potentially large) cluster tensor from hparams JSON.
        self.save_hyperparameters(ignore=["pretrained_clusters"])
        self.learning_rate = learning_rate
        self.cse = nn.CrossEntropyLoss(torch.tensor(ds_info.class_weights))

    def on_train_epoch_end(self):
        """Log per-cluster filter coefficient norms to monitor divergence."""
        for c in range(self.num_clusters):
            coeffs = self.filter_bank[c].temp
            self.log(f"filter_{c}_coeff_norm", coeffs.norm().item())
