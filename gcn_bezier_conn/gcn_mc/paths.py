"""Differentiable paths through a GCN's parameter space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.func import functional_call


ParameterState = Mapping[str, torch.Tensor]


def clone_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy model parameters into a detached, name-keyed state."""
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def interpolate(
    endpoint_a: ParameterState,
    endpoint_b: ParameterState,
    t: float | torch.Tensor,
    control: ParameterState | None = None,
) -> dict[str, torch.Tensor]:
    """Return linear or quadratic Bézier parameters at ``t``."""
    if control is None:
        return {
            name: (1.0 - t) * value_a + t * endpoint_b[name]
            for name, value_a in endpoint_a.items()
        }
    return {
        name: (1.0 - t) ** 2 * value_a
        + 2.0 * (1.0 - t) * t * control[name]
        + t**2 * endpoint_b[name]
        for name, value_a in endpoint_a.items()
    }


def make_control(
    endpoint_a: ParameterState, endpoint_b: ParameterState
) -> dict[str, nn.Parameter]:
    """Create trainable quadratic Bézier controls at the arithmetic midpoint."""
    return {
        name: nn.Parameter(0.5 * value_a + 0.5 * endpoint_b[name])
        for name, value_a in endpoint_a.items()
    }


def path_logits(
    model: nn.Module,
    data,
    endpoint_a: ParameterState,
    endpoint_b: ParameterState,
    t: float | torch.Tensor,
    control: ParameterState | None = None,
) -> torch.Tensor:
    """Evaluate ``model`` at one parameter-space path position."""
    parameters = interpolate(endpoint_a, endpoint_b, t, control)
    return functional_call(
        model,
        parameters,
        args=(data.x, data.edge_index),
        strict=True,
    )


def summarize_path(ts: Sequence[float], losses: Sequence[float]) -> dict[str, float]:
    """Summarize sampled loss and its excess over the endpoint-loss chord."""
    start, end = float(losses[0]), float(losses[-1])
    excess = [
        float(loss) - ((1.0 - float(t)) * start + float(t) * end)
        for t, loss in zip(ts, losses)
    ]
    index = max(range(len(excess)), key=excess.__getitem__)
    return {
        "barrier": excess[index],
        "argmax_t": float(ts[index]),
        "max_loss": max(float(loss) for loss in losses),
    }
