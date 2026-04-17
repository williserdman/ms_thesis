from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.func import functional_call

from gcn_modeconnect.data import load_single_graph
from gcn_modeconnect.graph_transforms import GraphCondition, apply_graph_condition
from gcn_modeconnect.model import GCN


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
            split_logits = logits[mask]
            split_y = self.data.y[mask]
            pred = split_logits.argmax(dim=1)
            acc = pred.eq(split_y).sum().item() / max(1, mask.sum().item())
            loss = F.cross_entropy(split_logits, split_y)
            out[f"{split_name}_loss"] = float(loss.item())
            out[f"{split_name}_acc"] = float(acc)
        return out


def build_gcn_manager(
    dataset_name: str,
    condition: GraphCondition,
    split_index: int,
    split_seed: int,
    hidden_channels: int,
    dropout: float,
    device: torch.device,
):
    data, num_features, num_classes = load_single_graph(
        name=dataset_name,
        split_index=split_index,
        split_seed=split_seed,
    )
    data, transform_stats = apply_graph_condition(data, condition=condition, seed=split_seed)
    data = data.to(device)

    model = GCN(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        dropout=dropout,
    ).to(device)

    bundle = DataBundle.from_data(data)
    manager = ModelManager(model=model, data=data, dataset_name=dataset_name)

    meta: dict[str, Any] = {
        "num_features": num_features,
        "num_classes": num_classes,
        "condition": condition.to_dict(),
        "transform_stats": transform_stats,
    }
    return model, data, bundle, manager, meta


def align_state_dict_to_model(state_dict: dict[str, torch.Tensor], model: torch.nn.Module) -> collections.OrderedDict:
    model_state = model.state_dict()
    out = collections.OrderedDict()
    for key, model_tensor in model_state.items():
        if key not in state_dict:
            raise KeyError(f"Checkpoint is missing parameter '{key}'")
        out[key] = state_dict[key].to(device=model_tensor.device, dtype=model_tensor.dtype)
    return out
