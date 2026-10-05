"""Shared plumbing for the spectral x mode-connectivity PoC experiments.

Provides: dataset loading (reusing the repo loader), training a LinearSpectralGNN,
flat/parameter-subset vectorization (for weight-space vs coefficient-space paths),
loss evaluation on a mask, and barrier computation along linear / Bezier paths.

Run modules from the `src/` directory so the repo's package imports resolve, e.g.

    cd src && python -m experiments.b1_barrier --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import torch
import pytorch_lightning as pl

from loading.LightningGraphLoader import load_datasets
from loading.DatasetInfo import DatasetInfo
from models.linear_spectral import LinearSpectralGNN
from mode_connectivity import paths


# ----------------------------------------------------------------------------
# reproducibility / devices
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def resolve_device(gpus: int):
    """Return (accelerator, devices, torch_device) honoring availability."""
    if gpus and torch.cuda.is_available():
        return "gpu", gpus, torch.device("cuda:0")
    return "cpu", "auto", torch.device("cpu")


def move_batch(batch, device):
    return batch.to(device)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
@dataclass
class Prepared:
    name: str
    ds_info: DatasetInfo
    datamodule: object
    batch: object  # single full-graph Data with train/val/test masks


def prepare_dataset(name: str) -> Prepared:
    network = load_datasets([name])[name]
    ds_info = DatasetInfo(
        network.num_classes,
        network.num_features,
        name,
        network.class_weights,
        network.data.data.x.shape[0],
    )
    batch = next(iter(network.data.train_dataloader()))
    return Prepared(name, ds_info, network.data, batch)


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def train_model(
    prepared: Prepared,
    model_kwargs: dict,
    max_epochs: int = 200,
    patience: int = 50,
    gpus: int = 0,
    seed: int = 0,
    verbose: bool = False,
) -> LinearSpectralGNN:
    """Train a LinearSpectralGNN on the prepared dataset; return the fitted model."""
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    set_seed(seed)
    accelerator, devices, _ = resolve_device(gpus)
    model = LinearSpectralGNN(prepared.ds_info, **model_kwargs)
    callbacks = [EarlyStopping(monitor="val_loss", patience=patience, mode="min")]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_progress_bar=verbose,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    trainer.fit(
        model,
        train_dataloaders=prepared.datamodule.train_dataloader(),
        val_dataloaders=prepared.datamodule.val_dataloader(),
    )
    return model


# ----------------------------------------------------------------------------
# parameter vectorization (whole net, or a named subset)
# ----------------------------------------------------------------------------
COEFF_NAMES = ["gamma"]  # the explicit spectral coefficient axis


def _named_subset(model, names):
    """Ordered list of (name, param) limited to `names` (None = all params)."""
    if names is None:
        return list(model.named_parameters())
    wanted = set(names)
    return [(n, p) for n, p in model.named_parameters() if n in wanted]


def get_vector(model, names=None) -> torch.Tensor:
    """Detached flat vector of the selected parameters (CPU)."""
    parts = [p.detach().reshape(-1).cpu() for _, p in _named_subset(model, names)]
    return torch.cat(parts) if parts else torch.zeros(0)


def set_vector(model, vec: torch.Tensor, names=None) -> None:
    """Load a flat vector back into the selected parameters (in place)."""
    vec = vec.to(next(model.parameters()).device)
    offset = 0
    with torch.no_grad():
        for _, p in _named_subset(model, names):
            n = p.numel()
            p.copy_(vec[offset:offset + n].view_as(p))
            offset += n
    if offset != vec.numel():
        raise ValueError(f"vector size {vec.numel()} != param size {offset}")


def gamma_vector(model) -> torch.Tensor:
    return get_vector(model, COEFF_NAMES)


def mlp_names(model):
    return [n for n, _ in model.named_parameters() if n not in set(COEFF_NAMES)]


# ----------------------------------------------------------------------------
# evaluation + barriers
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_loss_acc(model, batch, mask_name: str = "val_mask") -> tuple[float, float]:
    """Cross-entropy loss and accuracy on the given node mask."""
    was_training = model.training
    model.eval()
    logits, _ = model.forward(batch)
    mask = getattr(batch, mask_name)
    loss = model._loss(logits, batch.y, mask).item()
    preds = logits[mask].argmax(dim=-1)
    acc = float((preds == batch.y[mask]).float().mean().item())
    if was_training:
        model.train()
    return loss, acc


def barrier_along_path(
    model,
    batch,
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    names=None,
    control: torch.Tensor = None,
    n_points: int = 11,
    mask_name: str = "val_mask",
) -> dict:
    """Loss barrier between vec_a and vec_b along a linear or Bezier path.

    Interpolates only the parameters named in `names` (None = all). If `control`
    is given a quadratic Bezier path a -> control -> b is used; otherwise linear.
    Returns {ts, losses, accs, barrier, argmax_t, path}.
    """
    saved = get_vector(model, names)
    try:
        def point_at(t):
            if control is None:
                return paths.linear_interp(vec_a, vec_b, t)
            return paths.bezier_interp(vec_a, vec_b, control, t)

        def loss_of(vec):
            set_vector(model, vec, names)
            loss, _ = eval_loss_acc(model, batch, mask_name)
            return loss

        ts = paths.linspace(n_points)
        losses, accs = [], []
        for t in ts:
            set_vector(model, point_at(t), names)
            loss, acc = eval_loss_acc(model, batch, mask_name)
            losses.append(loss)
            accs.append(acc)
        barrier = paths.barrier_from_losses(losses, ts)
        argmax_t = paths.argmax_barrier_t(losses, ts)
    finally:
        set_vector(model, saved, names)  # always restore
    return {
        "ts": ts,
        "losses": losses,
        "accs": accs,
        "barrier": barrier,
        "argmax_t": argmax_t,
        "path": "bezier" if control is not None else "linear",
    }


# ----------------------------------------------------------------------------
# graph spectrum (for conditioning / steering experiments)
# ----------------------------------------------------------------------------
@torch.no_grad()
def laplacian_eigs(batch, k: int = None) -> "list[float]":
    """Eigenvalues of the symmetric-normalized Laplacian of the batch graph.

    For small PoC graphs we use a dense eig; pass k to keep only k smallest/largest
    samples (here we just return all, sorted). Returned as a python list of floats.
    """
    from models.linear_spectral import sym_norm_adj

    n = batch.x.size(0)
    adj = sym_norm_adj(batch.edge_index, n, batch.x.device).to_dense()
    lap = torch.eye(n, device=adj.device) - adj
    ev = torch.linalg.eigvalsh(lap).cpu().tolist()
    return ev


# ----------------------------------------------------------------------------
# CLI / IO
# ----------------------------------------------------------------------------
def base_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--datasets", nargs="+",
                   default=["cora", "citeseer", "squirrel", "Roman-empire"],
                   help="datasets to run on (repo loader names)")
    p.add_argument("--gpus", type=int, default=1, help="num GPUs (0 = CPU)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--basis", choices=["cheb", "mono"], default="cheb")
    p.add_argument("--domain", choices=["adj", "lap"], default="adj")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--n_points", type=int, default=11, help="samples along each path")
    p.add_argument("--out", type=str, default=None, help="output json path")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config: 1 small dataset, few epochs, K=2, CPU ok")
    return p


def apply_smoke(args):
    """Shrink everything for a fast end-to-end smoke run."""
    if args.smoke:
        args.datasets = ["cora"]
        args.epochs = 5
        args.patience = 5
        args.seeds = [0, 1]
        args.K = 2
        args.n_points = 5
        args.hidden_dim = 16
        if not torch.cuda.is_available():
            args.gpus = 0
    return args


def write_results(args, name: str, payload: dict) -> str:
    out = args.out or f"results_{name}{'_smoke' if args.smoke else ''}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[{name}] wrote results -> {out}")
    return out
