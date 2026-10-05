"""Model recipes adapted from the official REPAIR implementation.

Source: https://github.com/KellerJordan/REPAIR at commit
e90263d7a4d48376091327274ae541d8d6d34743.
"""

import math
from numbers import Real

import torch
from torch import nn


_VGG11_CONFIG = (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M")


class VGG(nn.Module):
    """CIFAR VGG model with the module layout used by the REPAIR notebook."""

    def __init__(self, width: Real = 1, num_classes: int = 10) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, Real):
            raise TypeError("width must be a positive real number")
        if not math.isfinite(float(width)) or width <= 0:
            raise ValueError("width must be a positive finite number")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")

        self.width = width
        layers: list[nn.Module] = []
        in_channels = 3
        for entry in _VGG11_CONFIG:
            if entry == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            out_channels = max(1, round(entry * width))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = outputs.reshape(outputs.shape[0], -1)
        return self.classifier(outputs)


def vgg11(width: Real = 1, num_classes: int = 10) -> nn.Module:
    """Build the REPAIR VGG11 recipe on CPU."""
    return VGG(width=width, num_classes=num_classes)


def tiny_mlp(
    input_dim: int = 16,
    hidden_dims: tuple[int, ...] = (32, 32),
    num_classes: int = 4,
) -> nn.Sequential:
    """Build a small Linear/ReLU network for synthetic experiments."""
    dimensions = (input_dim, *hidden_dims, num_classes)
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in dimensions):
        raise ValueError("all layer dimensions must be positive integers")
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one hidden layer")

    layers: list[nn.Module] = []
    for in_features, out_features in zip(dimensions[:-2], dimensions[1:-1]):
        layers.extend((nn.Linear(in_features, out_features), nn.ReLU()))
    layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
    return nn.Sequential(*layers)
