import argparse
import collections
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_CLASSIFICATION_DIR = REPO_ROOT / "PolyFormer" / "node_classification"
if str(NODE_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_CLASSIFICATION_DIR))

from mymodels import GCN  # noqa: E402
from utils import get_data_load, random_splits  # noqa: E402


@dataclass
class DataBundle:
    x: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    @classmethod
    def from_data(cls, data) -> "DataBundle":
        return cls(
            x=data.x,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
        )


class ModelManager:
    def __init__(self, model: torch.nn.Module, data, dataset_name: str):
        self.model = model
        self.data = data
        self.dataset_name = dataset_name.lower()

    def get_model_state(self) -> collections.OrderedDict:
        state = collections.OrderedDict()
        for key, value in self.model.state_dict().items():
            state[key] = value.detach().clone()
        return state

    def set_model_state(self, state_dict: collections.OrderedDict) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def no_touch_get_logits(self, state_dict: collections.OrderedDict) -> torch.Tensor:
        self.model.eval()
        return functional_call(self.model, state_dict, (self.data,))

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)

        out: dict[str, float] = {}
        for split_name, mask in [
            ("train", self.data.train_mask),
            ("val", self.data.val_mask),
            ("test", self.data.test_mask),
        ]:
            if self.dataset_name in {"minesweeper", "tolokers", "questions"}:
                split_logits = logits[mask].squeeze(-1)
                split_y = self.data.y[mask].to(torch.float)
                pred = (split_logits > 0).to(torch.long)
                acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
                loss = F.binary_cross_entropy_with_logits(split_logits, split_y)
            else:
                split_logits = logits[mask]
                split_y = self.data.y[mask]
                pred = split_logits.argmax(dim=1)
                acc = pred.eq(split_y).sum().item() / mask.sum().item()
                loss = F.cross_entropy(split_logits, split_y)
            out[f"{split_name}_loss"] = float(loss.item())
            out[f"{split_name}_acc"] = float(acc)
        return out


def build_model_args(dataset: str, device_idx: int = 0, hidden: int = 64) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset.lower(),
        base="mono",
        device_idx=device_idx,
        net="GCN",
        seed=42,
        idx_run=0,
        epochs=120,
        early_stopping=40,
        runs=2,
        model_dir="./saved_models",
        no_save_models=True,
        save_models=False,
        lr=0.01,
        weight_decay=5e-4,
        hidden=hidden,
        dropout=0.5,
        # Needed only because get_data_load computes PolyFormer bases.
        attn_lr=0.001,
        attn_wd=0.0001,
        n_head=1,
        d_ffn=128,
        q=1.0,
        multi=1.0,
        K=2,
        nlayer=1,
    )


@contextmanager
def _working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def apply_training_like_split(args: argparse.Namespace, dataset, data, seed: int, idx_run: int):
    if getattr(args, "use_external_splits", False):
        return data
    if args.dataset.lower() in ["citeseer", "pubmed", "cs", "physics"]:
        train_rate = 0.6
        val_rate = 0.2
        percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
        val_lb = int(round(val_rate * len(data.y)))
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb, seed)
    return data


def build_gcn_manager(
    dataset_name: str,
    run_seed: int,
    run_index: int,
    device: torch.device,
    hidden_dim: int = 64,
):
    args = build_model_args(dataset_name, device_idx=0, hidden=hidden_dim)
    args.seed = run_seed
    args.idx_run = run_index

    # PolyFormer utils use relative paths (for example ./bases), so align cwd.
    with _working_directory(NODE_CLASSIFICATION_DIR):
        dataset, data = get_data_load(args, split_index=run_index, split_seed=run_seed)
    data = apply_training_like_split(args, dataset, data, run_seed, run_index)
    model = GCN(dataset, args).to(device)
    data = data.to(device)
    bundle = DataBundle.from_data(data)
    manager = ModelManager(model=model, data=data, dataset_name=dataset_name)
    return args, dataset, data, bundle, manager


def align_state_dict_to_model(state_dict: dict[str, torch.Tensor], model: torch.nn.Module) -> collections.OrderedDict:
    model_state = model.state_dict()
    out = collections.OrderedDict()
    for key, model_tensor in model_state.items():
        if key not in state_dict:
            raise KeyError(f"Checkpoint is missing parameter '{key}'")
        out[key] = state_dict[key].to(device=model_tensor.device, dtype=model_tensor.dtype)
    return out
