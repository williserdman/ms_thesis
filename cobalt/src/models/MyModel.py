import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.gemini import ClusteredArnoldiModel
from pytorch_lightning.utilities import grad_norm


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculates accuracy between predictions and ground truth
    """

    # logits: (N, C), y: (N,)  — optionally may pass masked tensors only
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    total = int(y.numel())
    return float(correct) / max(total, 1)


class MyModel(
    ClusteredArnoldiModel,
    pl.LightningModule,
):
    def __init__(self, ds_info, learning_rate, **kwargs):
        super().__init__(ds_info, **kwargs)
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.learning_rate = learning_rate
        class_weights = torch.as_tensor(ds_info.class_weights, dtype=torch.float32)
        self.cse = nn.CrossEntropyLoss(weight=class_weights)

        ### MODEL DEFINITION ###

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer and log them
        # Gradients are unscaled automatically if using mixed precision (AMP)
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def training_step(self, batch, batch_idx):
        # 3. Retrieve the optimizers
        opt_cluster, opt_gnn = self.optimizers()  # type: ignore

        # Forward pass
        logits, cluster_loss = self.forward(batch)
        mask = batch.train_mask
        cse_loss = self.cse(logits[mask], batch.y[mask])

        # 4. Define your alternating schedule
        # Example: Alternating every 5 epochs
        is_cluster_phase = (self.current_epoch // 5) % 2 == 0

        if is_cluster_phase:
            # --- TRAIN CLUSTERS ---
            opt_cluster.zero_grad()
            # We use manual_backward instead of loss.backward() in PL manual optimization
            self.manual_backward(cluster_loss)
            opt_cluster.step()

            self.log("train_cluster_loss", cluster_loss, prog_bar=True)

        else:
            # --- TRAIN GNN ---
            opt_gnn.zero_grad()
            self.manual_backward(cse_loss)
            opt_gnn.step()

            self.log("train_cse_loss", cse_loss, prog_bar=True)

        # You can still log accuracy for tracking, even if not optimizing for it this phase
        acc = _accuracy(logits[mask], batch.y[mask])
        self.log("train_accuracy", acc, prog_bar=True)

    def on_validation_epoch_end(self):
        # Retrieve the schedulers
        # Note: If you have multiple schedulers, self.lr_schedulers() returns a list
        schedulers = self.lr_schedulers()

        # In Lightning, if schedulers is not None, it will return the list we defined
        if schedulers:
            sch_cluster, sch_gnn = schedulers

            # Retrieve the validation loss computed during the validation epoch
            val_loss = self.trainer.callback_metrics.get("val_loss")

            # Step the schedulers (check if not None to avoid crashing during sanity checks)
            if val_loss is not None:
                # ReduceLROnPlateau requires the monitored metric to be passed in
                sch_cluster.step(val_loss)
                sch_gnn.step(val_loss)

                # Optional: Log the current learning rates to track the drops
                opt_cluster, opt_gnn = self.optimizers()
                self.log("lr_cluster", opt_cluster.param_groups[0]["lr"], prog_bar=True)
                self.log("lr_gnn", opt_gnn.param_groups[0]["lr"], prog_bar=True)

    def test_step(self, batch):
        logits, inner_loss = self.forward(batch)
        # compute metrics only over test nodes
        mask = batch.test_mask
        loss = self.cse(logits[mask], batch.y[mask])  # + inner_loss
        acc = _accuracy(logits[mask], batch.y[mask])

        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self, lr=1e-3):  # type: ignore
        cluster_params = [self.clusters]
        gnn_params = [
            p for n, p in self.named_parameters() if n != "clusters" and p.requires_grad
        ]

        opt_cluster = torch.optim.Adam(cluster_params, lr=self.learning_rate)
        opt_gnn = torch.optim.AdamW(gnn_params, lr=self.learning_rate)

        scheduler_cluster = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_cluster, mode="min", factor=0.1, patience=10
        )
        scheduler_gnn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_gnn, mode="min", factor=0.1, patience=10
        )

        return {
            "optimizer": opt_cluster,
            "lr_scheduler": {
                "scheduler": scheduler_cluster,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }, {
            "optimizer": opt_gnn,
            "lr_scheduler": {
                "scheduler": scheduler_gnn,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
