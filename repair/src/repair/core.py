"""Parameter interpolation and activation-statistics REPAIR.

Adapted from https://github.com/KellerJordan/REPAIR at commit
e90263d7a4d48376091327274ae541d8d6d34743, VGG11 notebook cells 18-27.

The ``batchnorm`` method follows the fast estimator in the authors' VGG11
notebook. It averages per-batch BatchNorm statistics, so unequal batch sizes
and earlier per-batch corrections make it an approximation. The
``sequential`` method computes dataset moments and measures each merged layer
after corrections to earlier selected layers have been installed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn


_SUPPORTED_LAYERS = (nn.Linear, nn.Conv2d)
_EPS = 1e-5


class _RepairLayer(nn.Module):
    """A selected affine layer followed by its REPAIR correction."""

    def __init__(self, layer: nn.Module, batch_norm: nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.batch_norm = batch_norm

    def forward(self, inputs: Tensor) -> Tensor:
        return self.batch_norm(self.layer(inputs))


def _validate_alpha(alpha: float) -> float:
    try:
        value = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError("alpha must be a finite number in [0, 1]") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")
    return value


def _validate_architecture(model_a: nn.Module, model_b: nn.Module) -> None:
    modules_a = [(name, type(module)) for name, module in model_a.named_modules()]
    modules_b = [(name, type(module)) for name, module in model_b.named_modules()]
    if modules_a != modules_b:
        raise ValueError("models must have the same module architecture")

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    if state_a.keys() != state_b.keys():
        raise ValueError("models must have matching state dictionaries")
    for name in state_a:
        if state_a[name].shape != state_b[name].shape:
            raise ValueError(f"state tensor {name!r} has different shapes")


def interpolate(
    model_a: nn.Module,
    model_b: nn.Module,
    alpha: float = 0.5,
) -> nn.Module:
    """Return an eval-mode copy at ``(1 - alpha) * model_a + alpha * model_b``.

    Floating-point and complex parameters and buffers are interpolated.
    Non-floating buffers come from the nearer endpoint, with ``model_b`` used
    at the midpoint. Neither input model is modified.
    """

    alpha = _validate_alpha(alpha)
    _validate_architecture(model_a, model_b)
    merged = copy.deepcopy(model_a)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged_state: dict[str, Tensor] = {}
    for name, value_a in state_a.items():
        value_b = state_b[name].to(device=value_a.device)
        if value_a.is_floating_point() or value_a.is_complex():
            value_b = value_b.to(dtype=value_a.dtype)
            merged_state[name] = value_a * (1.0 - alpha) + value_b * alpha
        else:
            merged_state[name] = value_a if alpha < 0.5 else value_b
    merged.load_state_dict(merged_state)
    return merged.eval()


def _validate_layer_names(
    model_a: nn.Module,
    model_b: nn.Module,
    layer_names: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(layer_names, (str, bytes)):
        raise TypeError("layer_names must be a sequence of module names")
    names = tuple(layer_names)
    if not names:
        raise ValueError("layer_names must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("each layer name must be a non-empty string")
    if len(set(names)) != len(names):
        raise ValueError("layer_names must be unique")
    for name in names:
        try:
            layer_a = model_a.get_submodule(name)
            layer_b = model_b.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"unknown layer name {name!r}") from error
        if not isinstance(layer_a, _SUPPORTED_LAYERS):
            raise TypeError(f"layer {name!r} must be nn.Linear or nn.Conv2d")
        if type(layer_a) is not type(layer_b):
            raise ValueError(f"layer {name!r} differs between endpoint models")
    return names


def _validate_calibration_data(calibration_data: Iterable[object]) -> None:
    try:
        iterator = iter(calibration_data)
    except TypeError as error:
        raise TypeError("calibration_data must be re-iterable") from error
    if iterator is calibration_data:
        raise TypeError("calibration_data must be re-iterable, not a one-shot iterator")
    try:
        next(iterator)
    except StopIteration as error:
        raise ValueError("calibration_data is empty") from error


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if parameter.is_floating_point() or parameter.is_complex():
            return parameter.device, parameter.dtype
    raise ValueError("model must have a floating-point or complex parameter")


def _batch_inputs(batch: object, model: nn.Module) -> Tensor:
    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(inputs, Tensor):
        raise TypeError("each calibration batch must be a tensor or (input, label) pair")
    device, dtype = _model_device_dtype(model)
    return inputs.to(device=device, dtype=dtype)


def _run_calibration(
    model: nn.Module,
    calibration_data: Iterable[object],
    max_batches: int | None,
) -> int:
    batches = 0
    with torch.no_grad():
        for batch in calibration_data:
            if max_batches is not None and batches >= max_batches:
                break
            model(_batch_inputs(batch, model))
            batches += 1
    if batches == 0:
        raise ValueError("calibration_data is empty")
    return batches


def _batch_norm_for(layer: nn.Module) -> nn.Module:
    if isinstance(layer, nn.Linear):
        batch_norm = nn.BatchNorm1d(layer.out_features, eps=_EPS)
    elif isinstance(layer, nn.Conv2d):
        batch_norm = nn.BatchNorm2d(layer.out_channels, eps=_EPS)
    else:  # pragma: no cover - callers validate selected layers
        raise TypeError("REPAIR supports only nn.Linear and nn.Conv2d")
    return batch_norm.to(device=layer.weight.device, dtype=layer.weight.dtype)


def _track_batchnorm_stats(
    model: nn.Module,
    layer_names: tuple[str, ...],
    calibration_data: Iterable[object],
    max_batches: int | None,
) -> dict[str, tuple[Tensor, Tensor]]:
    model.eval()
    trackers: dict[str, nn.Module] = {}
    handles = []
    for name in layer_names:
        tracker = _batch_norm_for(model.get_submodule(name))
        tracker.momentum = None
        tracker.reset_running_stats()
        tracker.train()
        trackers[name] = tracker

        def update(_module, _args, output, *, batch_norm=tracker):
            batch_norm(output)

        handles.append(model.get_submodule(name).register_forward_hook(update))
    try:
        _run_calibration(model, calibration_data, max_batches)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for name, tracker in trackers.items():
        if tracker.num_batches_tracked.item() == 0:
            raise ValueError(f"selected layer {name!r} did not execute during calibration")
        result[name] = (
            tracker.running_mean.detach().clone(),
            tracker.running_var.detach().sqrt().clone(),
        )
    return result


class _MomentAccumulator:
    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.total: Tensor | None = None
        self.square_total: Tensor | None = None
        self.count = 0

    def add(self, output: Tensor) -> None:
        expected_rank = 2 if isinstance(self.layer, nn.Linear) else 4
        if not isinstance(output, Tensor) or output.ndim != expected_rank:
            raise ValueError(
                f"selected {type(self.layer).__name__} must return a rank-{expected_rank} tensor"
            )
        reduce_dims = (0,) if expected_rank == 2 else (0, 2, 3)
        values = output.detach().to(dtype=torch.float64)
        batch_total = values.sum(dim=reduce_dims)
        batch_square_total = values.square().sum(dim=reduce_dims)
        if self.total is None:
            self.total = batch_total
            self.square_total = batch_square_total
        else:
            self.total += batch_total
            self.square_total += batch_square_total
        self.count += output.numel() // output.shape[1]

    def moments(self) -> tuple[Tensor, Tensor]:
        if self.count == 0 or self.total is None or self.square_total is None:
            raise ValueError("selected layer did not execute during calibration")
        mean = self.total / self.count
        variance = (self.square_total / self.count - mean.square()).clamp_min(0.0)
        dtype = self.layer.weight.dtype
        return mean.to(dtype=dtype), variance.sqrt().to(dtype=dtype)


def _measure_exact_stats(
    model: nn.Module,
    layer_names: tuple[str, ...],
    calibration_data: Iterable[object],
    max_batches: int | None,
) -> dict[str, tuple[Tensor, Tensor]]:
    model.eval()
    accumulators = {
        name: _MomentAccumulator(model.get_submodule(name)) for name in layer_names
    }
    handles = []
    for name, accumulator in accumulators.items():
        def update(_module, _args, output, *, moments=accumulator):
            moments.add(output)

        handles.append(model.get_submodule(name).register_forward_hook(update))
    try:
        _run_calibration(model, calibration_data, max_batches)
    finally:
        for handle in handles:
            handle.remove()
    return {name: accumulator.moments() for name, accumulator in accumulators.items()}


def _replace_submodule(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


def _configured_correction(
    layer: nn.Module,
    source_mean: Tensor,
    source_std: Tensor,
    goal_mean: Tensor,
    goal_std: Tensor,
) -> _RepairLayer:
    batch_norm = _batch_norm_for(layer)
    with torch.no_grad():
        batch_norm.running_mean.copy_(source_mean)
        batch_norm.running_var.copy_(source_std.square())
        batch_norm.weight.copy_(goal_std)
        batch_norm.bias.copy_(goal_mean)
    batch_norm.eval()
    return _RepairLayer(layer, batch_norm)


def _repair_batchnorm(
    model_a: nn.Module,
    model_b: nn.Module,
    merged: nn.Module,
    calibration_data: Iterable[object],
    alpha: float,
    layer_names: tuple[str, ...],
    max_batches: int | None,
) -> None:
    stats_a = _track_batchnorm_stats(model_a, layer_names, calibration_data, max_batches)
    stats_b = _track_batchnorm_stats(model_b, layer_names, calibration_data, max_batches)
    corrections: list[nn.Module] = []
    for name in layer_names:
        mean_a, std_a = stats_a[name]
        mean_b, std_b = stats_b[name]
        goal_mean = mean_a * (1.0 - alpha) + mean_b * alpha
        goal_std = std_a * (1.0 - alpha) + std_b * alpha
        layer = merged.get_submodule(name)
        correction = _configured_correction(
            layer,
            torch.zeros_like(goal_mean),
            torch.ones_like(goal_std),
            goal_mean,
            goal_std,
        )
        correction.batch_norm.momentum = None
        correction.batch_norm.reset_running_stats()
        correction.batch_norm.train()
        _replace_submodule(merged, name, correction)
        corrections.append(correction.batch_norm)

    # Only the inserted BatchNorms train. Dropout and every original module
    # remain in eval mode during calibration.
    merged.eval()
    for batch_norm in corrections:
        batch_norm.train()
    _run_calibration(merged, calibration_data, max_batches)
    merged.eval()


def _repair_sequential(
    model_a: nn.Module,
    model_b: nn.Module,
    merged: nn.Module,
    calibration_data: Iterable[object],
    alpha: float,
    layer_names: tuple[str, ...],
    max_batches: int | None,
) -> None:
    stats_a = _measure_exact_stats(model_a, layer_names, calibration_data, max_batches)
    stats_b = _measure_exact_stats(model_b, layer_names, calibration_data, max_batches)
    for name in layer_names:
        mean_a, std_a = stats_a[name]
        mean_b, std_b = stats_b[name]
        goal_mean = mean_a * (1.0 - alpha) + mean_b * alpha
        goal_std = std_a * (1.0 - alpha) + std_b * alpha
        source = _measure_exact_stats(merged, (name,), calibration_data, max_batches)[name]
        source_mean, source_std = source
        layer = merged.get_submodule(name)
        correction = _configured_correction(
            layer, source_mean, source_std, goal_mean, goal_std
        )
        _replace_submodule(merged, name, correction)
        merged.eval()


def _fuse_layer(correction: _RepairLayer) -> nn.Module:
    layer = correction.layer
    batch_norm = correction.batch_norm
    with torch.no_grad():
        scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
        reshape = (scale.shape[0],) + (1,) * (layer.weight.ndim - 1)
        layer.weight.mul_(scale.reshape(reshape))
        if layer.bias is None:
            bias = torch.zeros_like(batch_norm.running_mean)
        else:
            bias = layer.bias.detach()
        fused_bias = batch_norm.bias + scale * (bias - batch_norm.running_mean)
        layer.bias = nn.Parameter(
            fused_bias.detach().clone(), requires_grad=layer.weight.requires_grad
        )
    return layer


def _fuse_in_place(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, _RepairLayer):
            setattr(module, name, _fuse_layer(child))
        else:
            _fuse_in_place(child)


def fuse_repair(model: nn.Module) -> nn.Module:
    """Return an eval-mode copy with internal REPAIR corrections fused."""

    fused = copy.deepcopy(model)
    _fuse_in_place(fused)
    return fused.eval()


def repair(
    model_a: nn.Module,
    model_b: nn.Module,
    calibration_data: Iterable[object],
    alpha: float = 0.5,
    *,
    layer_names: Sequence[str],
    max_batches: int | None = None,
    method: str = "batchnorm",
    fuse: bool = True,
) -> nn.Module:
    """Interpolate aligned endpoints and repair selected activation moments.

    ``layer_names`` must list ``nn.Linear`` or ``nn.Conv2d`` modules in
    execution order. Calibration data must be re-iterable and yield input
    tensors or ``(input, label)`` batches. This function assumes a sequential
    feed-forward path; it does not infer residual or general graph structure.
    """

    alpha = _validate_alpha(alpha)
    if max_batches is not None:
        if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0:
            raise ValueError("max_batches must be a positive integer or None")
    if method not in {"batchnorm", "sequential"}:
        raise ValueError("method must be 'batchnorm' or 'sequential'")
    _validate_architecture(model_a, model_b)
    names = _validate_layer_names(model_a, model_b, layer_names)
    _validate_calibration_data(calibration_data)

    endpoint_a = copy.deepcopy(model_a).eval()
    endpoint_b = copy.deepcopy(model_b).eval()
    merged = interpolate(model_a, model_b, alpha)
    if method == "batchnorm":
        _repair_batchnorm(
            endpoint_a,
            endpoint_b,
            merged,
            calibration_data,
            alpha,
            names,
            max_batches,
        )
    else:
        _repair_sequential(
            endpoint_a,
            endpoint_b,
            merged,
            calibration_data,
            alpha,
            names,
            max_batches,
        )
    return fuse_repair(merged) if fuse else merged.eval()


__all__ = ["fuse_repair", "interpolate", "repair"]
