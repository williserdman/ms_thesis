"""Differentiable parameter paths with isolated normalization buffers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.func import functional_call


ParameterState = Mapping[str, torch.Tensor]


def _batchnorm_modules(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def _restore_training_modes(
    modules: Sequence[nn.Module], modes: Sequence[bool]
) -> None:
    for module, training in zip(modules, modes):
        module.training = training


def _clone_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }


def _reset_batchnorm_buffers(
    model: nn.Module, buffers: Mapping[str, torch.Tensor]
) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        prefix = f"{name}." if name else ""
        running_mean = buffers.get(f"{prefix}running_mean")
        running_var = buffers.get(f"{prefix}running_var")
        batches = buffers.get(f"{prefix}num_batches_tracked")
        if running_mean is not None:
            running_mean.zero_()
        if running_var is not None:
            running_var.fill_(1)
        if batches is not None:
            batches.zero_()


def calibrate_batchnorm(model: nn.Module, graph) -> None:
    """Replace BatchNorm running statistics with one full-graph batch."""
    batchnorms = _batchnorm_modules(model)
    if not batchnorms:
        return

    modules = list(model.modules())
    modes = [module.training for module in modules]
    momenta = [module.momentum for module in batchnorms]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        model.eval()
        for module in batchnorms:
            module.train()
            module.reset_running_stats()
            module.momentum = 1.0
        with torch.no_grad():
            model(graph.x, graph.edge_index)
    except Exception:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        raise
    finally:
        for module, momentum in zip(batchnorms, momenta):
            module.momentum = momentum
        _restore_training_modes(modules, modes)


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
    buffers = _clone_buffers(model)
    state = {**parameters, **buffers}
    batchnorms = _batchnorm_modules(model)

    if model.training or not batchnorms:
        return functional_call(
            model,
            state,
            args=(data.x, data.edge_index),
            strict=True,
        )

    modules = list(model.modules())
    modes = [module.training for module in modules]
    momenta = [module.momentum for module in batchnorms]
    try:
        model.eval()
        for module in batchnorms:
            module.train()
            module.momentum = 1.0
        _reset_batchnorm_buffers(model, buffers)
        with torch.no_grad():
            functional_call(
                model,
                state,
                args=(data.x, data.edge_index),
                strict=True,
            )
        model.eval()
        return functional_call(
            model,
            state,
            args=(data.x, data.edge_index),
            strict=True,
        )
    finally:
        for module, momentum in zip(batchnorms, momenta):
            module.momentum = momentum
        _restore_training_modes(modules, modes)


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
