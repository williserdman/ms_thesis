from dataclasses import dataclass
import torch


@dataclass
class DatasetInfo:
    num_classes: int
    num_features: int
    name: str
    class_weights: torch.Tensor
    N: int
