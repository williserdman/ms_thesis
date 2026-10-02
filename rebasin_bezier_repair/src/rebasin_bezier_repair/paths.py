"""Quadratic paths adapted from sibling gcn_mc/paths.py and fit_curve."""

import copy
import math
from collections.abc import Callable, Collection, Iterable

import torch
from git_re_basin import validate_state_pair
from torch import Tensor, nn

from .adapters.base import ArchitectureAdapter, State


def _fraction(t: float) -> float:
    t = float(t)
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("t must be finite and in [0, 1]")
    return t


def _fixed_state(a: State, b: State, parameter_names: Collection[str]) -> None:
    validate_state_pair(a, b)
    names = set(parameter_names)
    if not names <= set(a):
        raise ValueError("unknown parameter names")
    for name, value in a.items():
        if name in names:
            if not value.is_floating_point():
                raise ValueError(f"interpolated parameter must be floating point: {name}")
        elif not torch.equal(value, b[name]):
            raise ValueError(f"fixed buffer differs: {name}")


def bezier_state(
    a: State, b: State, control: State, t: float, *, parameter_names: Collection[str],
) -> dict[str, Tensor]:
    """Interpolate parameters differentiably while copying equal fixed buffers."""
    t = _fraction(t)
    names = set(parameter_names)
    _fixed_state(a, b, names)
    if set(control) != names:
        raise ValueError("control keys must equal parameter names")
    validate_state_pair({name: a[name] for name in names}, control)
    return {
        name: ((1 - t) ** 2 * value + 2 * t * (1 - t) * control[name] + t**2 * b[name])
        if name in names else value.clone()
        for name, value in a.items()
    }


def fit_curve(
    model: nn.Module, adapter: ArchitectureAdapter, a: State, b: State,
    train_data: Iterable[object], loss_fn: Callable[[Tensor, object], Tensor], *,
    steps: int = 100, lr: float = 0.01, seed: int = 0,
) -> dict[str, Tensor]:
    """Fit only a midpoint-initialized control using training batches and loss."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("curve steps must be a positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("curve learning rate must be finite and positive")
    iterator = iter(train_data)
    if iterator is train_data:
        raise TypeError("training data must be re-iterable")
    names = tuple(dict(model.named_parameters()))
    _fixed_state(a, b, names)
    a = {name: value.detach().clone() for name, value in a.items()}
    b = {name: value.detach().clone() for name, value in b.items()}
    template = copy.deepcopy(model).train()
    template.requires_grad_(False)
    control = {name: nn.Parameter((a[name] + b[name]) / 2) for name in names}
    optimizer = torch.optim.Adam(control.values(), lr=lr)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    cuda_devices = sorted({value.device.index for value in a.values() if value.is_cuda})
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_data)
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise ValueError("training data must be nonempty") from error
            optimizer.zero_grad(set_to_none=True)
            t = float(torch.rand((), generator=rng, device="cpu"))
            state = bezier_state(a, b, control, t, parameter_names=names)
            loss = loss_fn(adapter.forward(template, state, batch), batch)
            if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("training loss must be a finite scalar tensor")
            if not loss.requires_grad:
                raise ValueError("training loss must be differentiable through the control")
            loss.backward()
            optimizer.step()
    return {name: value.detach().clone() for name, value in control.items()}
