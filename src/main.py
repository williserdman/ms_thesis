from dataclasses import dataclass
import torch
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from LightningGraphLoader import load_datasets, LightningGraph
from args import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import grad_norm
import multiprocessing
import time
import json
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import get_cosine_schedule_with_warmup

SEED_VALUE = 42

# to use tensor cores
torch.set_float32_matmul_precision(
    "high"
)  # "medium" also works to take advantage of tensor cores
torch.autograd.set_detect_anomaly(True)


ALL_DATASETS = [
    "Questions",
    "Roman-empire",
    "Amazon-ratings",
    "Tolokers",
    "computers",
    "photo",
    "texas",
    "cornell",
    "Cora",
    "Citeseer",
    "Pubmed",
    "squirrel",
    "chameleon",
    "actor",
    "Minesweeper",
]

ALL_DATASETS = ["squirrel", "chameleon", "Roman-empire"]


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculates accuracy between predictions and ground truth
    """

    # logits: (N, C), y: (N,)  — optionally may pass masked tensors only
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    total = int(y.numel())
    return float(correct) / max(total, 1)


@dataclass
class DatasetInfo:
    num_classes: int
    num_features: int
    name: str
    class_weights: torch.Tensor
    N: int


# define the LightningModule
class LitGARNOLDI(pl.LightningModule):
    def __init__(self, model, ds_info, args, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model(ds_info, args)
        print(ds_info.class_weights)
        self.cse = nn.CrossEntropyLoss(torch.tensor(ds_info.class_weights))

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer and log them
        # Gradients are unscaled automatically if using mixed precision (AMP)
        norms = grad_norm(self.trainer.model, norm_type=2)  # type: ignore
        self.log_dict(norms)

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward

        logits, inner_loss = self.model(batch)
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
        logits, inner_loss = self.model(batch)
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
        logits, inner_loss = self.model(batch)
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

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

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


from pytorch_lightning.callbacks import Callback


# from models.diffused_attention import DiffusedAttention
from models.heterophily_diffused_attention import DiffusedAttention
from models.maxcut import MaxCutClusters


MODEL = MaxCutClusters
TEST_NAME = MODEL_NAME = MODEL.__name__ + "_sanity_check"


def train_job(dataset_name, sargs, gpu_id, results_list):
    pl.seed_everything(SEED_VALUE, workers=True)
    # Re-load only the needed dataset inside the subprocess to avoid pickling issues
    dm_local = load_datasets([dataset_name])[dataset_name]
    args = MyArgs(sargs)
    lr = 1e-3

    if dataset_name in {"actor", "squirrel", "Minesweeper"}:
        args.hidden = args.hidden_dim = 32

    if dataset_name in {"Questions", "Tolokers"}:
        args.hidden = args.hidden_dim = 64

    if dataset_name in {"Amazon-ratings", "Roman-empire"}:
        args.hidden = args.hidden_dim = 128

    ds_info = DatasetInfo(
        name=dataset_name,
        num_classes=dm_local.num_classes,
        num_features=dm_local.num_features,
        class_weights=dm_local.class_weights,
        N=dm_local.data.data.x.shape[0],
    )

    pl.seed_everything(SEED_VALUE, workers=True)
    lgarnoldi = LitGARNOLDI(MODEL, ds_info, args, learning_rate=lr)

    logger = TensorBoardLogger(
        TEST_NAME,
        version=dataset_name,
        name=f"{MODEL_NAME}_{dataset_name}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu_id],
        limit_train_batches=100,
        max_epochs=4000,
        callbacks=[lr_monitor],  # _explainable_test
        check_val_every_n_epoch=100,
        logger=logger,
        gradient_clip_val=1.0,
    )

    trainer.fit(model=lgarnoldi, datamodule=dm_local.data)

    test_res = trainer.test(lgarnoldi, datamodule=dm_local.data)
    results_list.append(
        (dataset_name, {"status": "ok", "test": test_res, "gpu": gpu_id})
    )
    return


if __name__ == "__main__":
    # fallbacks
    LR = 0.000005
    DPRATE = 0.5
    HIDDEN = 128

    pl.seed_everything(SEED_VALUE, workers=True)
    dm_dict = load_datasets(ALL_DATASETS)
    sargs = SimplifiedArgs("Chebyshev", "doesn't matter", LR, DPRATE, HIDDEN)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{TEST_NAME}_training_results_{timestamp}.json"

    if gpu_count <= 0:
        # fallback: run sequentially on CPU
        summary = []
        for d in dm_dict:
            dm = dm_dict[d]
            ds_info = DatasetInfo(
                name=d,
                num_classes=dm.num_classes,
                num_features=dm.num_features,
                class_weights=dm.class_weights,
                N=dm.data.data.x.shape[0],
            )
            args = MyArgs(sargs)
            args.K = 10

            lgarnoldi = LitGARNOLDI(MODEL, ds_info, args, learning_rate=LR)
            trainer = pl.Trainer(
                accelerator="cpu",
                limit_train_batches=100,
                max_epochs=1000,
                callbacks=[EarlyStopping("val_loss", patience=5)],
                check_val_every_n_epoch=10,
            )
            trainer.fit(model=lgarnoldi, datamodule=dm.data)
            test_res = trainer.test(lgarnoldi, datamodule=dm.data)
            summary.append((d, {"status": "ok", "test": test_res, "device": "cpu"}))

        print("Summary:")
        for s in summary:
            print(s)

        # write results to file
        output = []
        for dataset_name, info in summary:
            entry = {"dataset": dataset_name}
            entry.update(info)
            output.append(entry)
        with open(out_filename, "w") as fh:
            json.dump(output, fh, indent=2, default=str)
        print(f"Wrote results to {out_filename}")
    else:
        manager = multiprocessing.Manager()
        results = manager.list()

        free_gpus = list(range(gpu_count))
        processes = {}

        for d in dm_dict:
            # wait until a GPU is available
            while not free_gpus:
                # poll running processes and reclaim finished GPUs
                for p in list(processes.keys()):
                    if not p.is_alive():
                        p.join()
                        free_gpus.append(processes.pop(p))
                if not free_gpus:
                    time.sleep(1)

            gpu = free_gpus.pop(0)
            p = multiprocessing.Process(target=train_job, args=(d, sargs, gpu, results))
            p.start()
            processes[p] = gpu

        # wait for all processes to finish
        for p in list(processes.keys()):
            p.join()
            processes.pop(p)

        # Print summary of results
        print("Distributed training summary:")
        for item in list(results):
            dataset_name, info = item
            print(f"{dataset_name}: {info}")

        # write results to file
        output = []
        for item in list(results):
            dataset_name, info = item
            entry = {"dataset": dataset_name}
            entry.update(info)
            output.append(entry)
        with open(out_filename, "w") as fh:
            json.dump(output, fh, indent=2, default=str)
        print(f"Wrote results to {out_filename}")
