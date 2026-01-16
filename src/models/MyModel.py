import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.heterophily_diffused_attention import DiffusedAttention
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
    DiffusedAttention,
    pl.LightningModule,
):
    def __init__(self, ds_info, learning_rate, **kwargs):
        super().__init__(ds_info, **kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.cse = nn.CrossEntropyLoss(torch.tensor(ds_info.class_weights))

        ### MODEL DEFINITION ###

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer and log them
        # Gradients are unscaled automatically if using mixed precision (AMP)
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        logits, inner_loss = self.forward(batch)
        # compute loss only over training nodes
        if hasattr(batch, "train_mask") and batch.train_mask is not None:
            mask = batch.train_mask
            loss = self.cse(logits[mask], batch.y[mask]) + inner_loss
        else:
            loss = self.cse(logits, batch.y) + inner_loss
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        logits, inner_loss = self.forward(batch)
        # compute metrics only over validation nodes
        if hasattr(batch, "val_mask") and batch.val_mask is not None:
            mask = batch.val_mask
            loss = self.cse(logits[mask], batch.y[mask]) + inner_loss
            acc = _accuracy(logits[mask], batch.y[mask])
        else:
            loss = self.cse(logits, batch.y) + inner_loss
            acc = _accuracy(logits, batch.y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def test_step(self, batch):
        logits, inner_loss = self.forward(batch)
        # compute metrics only over test nodes
        if hasattr(batch, "test_mask") and batch.test_mask is not None:
            mask = batch.test_mask
            loss = self.cse(logits[mask], batch.y[mask]) + inner_loss
            acc = _accuracy(logits[mask], batch.y[mask])
        else:
            loss = self.cse(logits, batch.y) + inner_loss
            acc = _accuracy(logits, batch.y)

        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self, lr=1e-3):  # type: ignore

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        """ total_steps = self.trainer.estimated_stepping_batches

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update scheduler every step (not every epoch)
                "frequency": 1,
            },
        } """
        return {"optimizer": optimizer}
