"""Sequential REPAIR adapted from sibling gcn_mc/repair.py and repair/core.py."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .adapters.base import ArchitectureAdapter, ObservationSelector, RepairSite, State
from .paths import _fraction


@dataclass(frozen=True)
class ChannelMoments:
    mean: Tensor
    std: Tensor


Moments = Mapping[str, ChannelMoments]


@torch.no_grad()
def collect_moments(
    model: nn.Module, adapter: ArchitectureAdapter, state: State,
    calibration_data: Iterable[object], *,
    selector: ObservationSelector | None = None,
    sites: tuple[RepairSite, ...] | None = None,
) -> dict[str, ChannelMoments]:
    """Pool selected preactivations without changing the model's modes/hooks."""
    if calibration_data is None:
        raise ValueError("calibration data is required")
    if iter(calibration_data) is calibration_data:
        raise TypeError("calibration data must be re-iterable")
    sites = adapter.repair_sites(model) if sites is None else sites
    if not sites or len({site.name for site in sites}) != len(sites):
        raise ValueError("repair sites must be nonempty with unique names")
    totals = {}
    modes = [(module, module.training) for module in model.modules()]
    handles = []
    current_batch = None

    def record(site, output):
        if not isinstance(output, Tensor) or not output.is_floating_point():
            raise ValueError(f"repair site must output a floating tensor: {site.name}")
        selected = selector(site, output, current_batch) if selector else output
        if not isinstance(selected, Tensor) or selected.ndim != output.ndim:
            raise ValueError("observation selection must preserve tensor rank")
        if selected.shape[site.channel_axis] != output.shape[site.channel_axis]:
            raise ValueError("observation selection must preserve channels")
        channels = selected.shape[site.channel_axis]
        if channels == 0:
            raise ValueError("repair sites must have channels")
        values = selected.movedim(site.channel_axis, -1).reshape(-1, channels)
        if not torch.isfinite(values).all():
            raise ValueError(f"nonfinite calibration activation: {site.name}")
        values = values.to(dtype=torch.float64)
        batch_sum, batch_square = values.sum(0), values.square().sum(0)
        if site.name not in totals:
            totals[site.name] = [batch_sum, batch_square, values.shape[0], output.dtype]
        else:
            accumulator = totals[site.name]
            accumulator[0].add_(batch_sum)
            accumulator[1].add_(batch_square)
            accumulator[2] += values.shape[0]

    try:
        model.eval()
        for site in sites:
            module = model.get_submodule(site.module_path)
            handles.append(module.register_forward_hook(
                lambda _module, _args, output, site=site: record(site, output)
            ))
        for current_batch in calibration_data:
            adapter.forward(model, state, current_batch)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    result = {}
    for site in sites:
        if site.name not in totals or totals[site.name][2] < 2:
            raise ValueError(f"calibration needs at least two selected observations: {site.name}")
        total, square_total, count, dtype = totals[site.name]
        mean = total / count
        variance = (square_total / count - mean.square()).clamp_min(0)
        result[site.name] = ChannelMoments(mean.to(dtype=dtype), variance.sqrt().to(dtype=dtype))
    return result


@torch.no_grad()
def repair_state(
    model: nn.Module, adapter: ArchitectureAdapter, state: State, t: float,
    endpoint_moments: tuple[Moments, Moments], calibration_data: Iterable[object] | None,
    *, selector: ObservationSelector | None = None, eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Correct each hidden site after earlier corrections; endpoints bypass REPAIR."""
    t = _fraction(t)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("repair epsilon must be finite and positive")
    repaired = {name: value.detach().clone() for name, value in state.items()}
    if t == 0 or t == 1:
        return repaired
    a, b = endpoint_moments
    for site in adapter.repair_sites(model):
        current = collect_moments(model, adapter, repaired, calibration_data,
                                  selector=selector, sites=(site,))[site.name]
        target_mean = ((1 - t) * a[site.name].mean + t * b[site.name].mean).to(current.mean)
        target_std = ((1 - t) * a[site.name].std + t * b[site.name].std).to(current.std)
        scale = target_std / current.std.clamp_min(eps)
        shift = target_mean - scale * current.mean
        repaired = adapter.apply_correction(repaired, site, scale, shift)
    return repaired
