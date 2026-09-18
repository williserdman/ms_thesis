"""Graph-specific channel alignment and REPAIR for the project GCN."""

from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
import sys

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .model import GCN


EPSILON = 1e-5
CORRELATION_EPSILON = 1e-4

_REPAIR_SRC = Path(__file__).resolve().parents[2] / "repair" / "src"
_REPAIR_CORE = _REPAIR_SRC / "repair" / "core.py"
if str(_REPAIR_SRC) not in sys.path:
    sys.path.insert(0, str(_REPAIR_SRC))
repair_core = importlib.import_module("repair.core")
if Path(repair_core.__file__).resolve() != _REPAIR_CORE:
    raise RuntimeError(f"Expected sibling REPAIR source at {_REPAIR_CORE}")
interpolate_models = repair_core.interpolate


def _validate_model(model: nn.Module) -> GCN:
    if not isinstance(model, GCN):
        raise TypeError("graph REPAIR supports only gcn_mc.model.GCN")
    if len(model.convs) < 2:
        raise ValueError("GCN must contain at least one hidden convolution")
    return model


def _train_mask(graph) -> torch.Tensor:
    mask = getattr(graph, "train_mask", None)
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 1
        or mask.numel() != graph.x.shape[0]
        or not bool(mask.any())
    ):
        raise ValueError("graph.train_mask must be a nonempty boolean node mask")
    return mask


def _mode_flags(model: nn.Module) -> list[bool]:
    return [module.training for module in model.modules()]


def _restore_modes(model: nn.Module, flags: list[bool]) -> None:
    for module, training in zip(model.modules(), flags):
        module.training = training


def _capture_hidden(model: GCN, graph, *, post_relu: bool) -> dict[str, torch.Tensor]:
    """Capture full-graph hidden convolution outputs with dropout disabled."""
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for index, conv in enumerate(model.convs[:-1]):
        name = f"convs.{index}"

        def capture(_module, _args, output, *, layer_name=name):
            value = F.relu(output) if post_relu else output
            captured[layer_name] = value.detach().clone()

        handles.append(conv.register_forward_hook(capture))

    flags = _mode_flags(model)
    try:
        model.eval()
        with torch.no_grad():
            model(graph.x, graph.edge_index)
    finally:
        for handle in handles:
            handle.remove()
        _restore_modes(model, flags)

    if len(captured) != len(model.convs) - 1:
        raise RuntimeError("failed to capture every hidden GCN convolution")
    return captured


def hidden_statistics(model: GCN, graph) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return training-node population moments of hidden GCN preactivations."""
    model = _validate_model(model)
    mask = _train_mask(graph)
    outputs = _capture_hidden(model, graph, post_relu=False)
    statistics = {}
    for name, output in outputs.items():
        values = output[mask.to(output.device)].to(dtype=torch.float64)
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        statistics[name] = (
            mean.to(dtype=output.dtype),
            std.to(dtype=output.dtype),
        )
    return statistics


def _correlation(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    left = reference.to(dtype=torch.float64)
    right = candidate.to(device=left.device, dtype=torch.float64)
    left = left - left.mean(dim=0)
    right = right - right.mean(dim=0)
    covariance = left.T @ right / left.shape[0]
    left_std = torch.sqrt(left.square().mean(dim=0))
    right_std = torch.sqrt(right.square().mean(dim=0))
    return covariance / (torch.outer(left_std, right_std) + CORRELATION_EPSILON)


def _permute_hidden(model: GCN, index: int, permutation: torch.Tensor) -> None:
    current = model.convs[index]
    following = model.convs[index + 1]
    output_order = permutation.to(current.lin.weight.device)
    input_order = permutation.to(following.lin.weight.device)
    with torch.no_grad():
        current.lin.weight.copy_(current.lin.weight.index_select(0, output_order))
        if current.bias is not None:
            current.bias.copy_(current.bias.index_select(0, output_order))
        following.lin.weight.copy_(following.lin.weight.index_select(1, input_order))


def align_gcn(reference: GCN, candidate: GCN, graph) -> tuple[GCN, dict]:
    """Return an eval-mode candidate copy aligned to the reference hidden channels."""
    _validate_model(reference)
    _validate_model(candidate)
    _train_mask(graph)
    if len(reference.convs) != len(candidate.convs):
        raise ValueError("GCN endpoints must have the same depth")

    aligned_reference = copy.deepcopy(reference).eval()
    original_candidate = copy.deepcopy(candidate).eval()
    aligned_candidate = copy.deepcopy(candidate).eval()
    mask = graph.train_mask
    permutations: dict[str, list[int]] = {}

    for index in range(len(aligned_reference.convs) - 1):
        name = f"convs.{index}"
        reference_output = _capture_hidden(
            aligned_reference, graph, post_relu=True
        )[name][mask]
        candidate_output = _capture_hidden(
            aligned_candidate, graph, post_relu=True
        )[name][mask]
        correlation = _correlation(reference_output, candidate_output)
        rows, columns = linear_sum_assignment(
            correlation.detach().cpu().numpy(), maximize=True
        )
        if not torch.equal(torch.from_numpy(rows), torch.arange(correlation.shape[0])):
            raise RuntimeError("Hungarian assignment did not cover every hidden channel")
        permutation = torch.from_numpy(columns).to(dtype=torch.long)
        _permute_hidden(aligned_candidate, index, permutation)
        permutations[name] = permutation.tolist()

    with torch.no_grad():
        expected = original_candidate(graph.x, graph.edge_index)
        actual = aligned_candidate(graph.x, graph.edge_index)
    max_error = float((actual - expected).abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return aligned_candidate.eval(), {
        "permutations": permutations,
        "max_logit_error": max_error,
    }


def _mean_variance(std: torch.Tensor) -> float:
    return float(std.to(dtype=torch.float64).square().mean().item())


def _layer_diagnostics(
    source: tuple[torch.Tensor, torch.Tensor],
    target: tuple[torch.Tensor, torch.Tensor],
    post: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    source_mean, source_std = source
    target_mean, target_std = target
    post_mean, post_std = post
    return {
        "pre_mean_variance": _mean_variance(source_std),
        "target_mean_variance": _mean_variance(target_std),
        "post_mean_variance": _mean_variance(post_std),
        "max_mean_target_residual": float(
            (post_mean - target_mean).abs().max().item()
        ),
        "max_std_target_residual": float(
            (post_std - target_std).abs().max().item()
        ),
    }


def _fuse_correction(
    model: GCN,
    index: int,
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> None:
    conv = model.convs[index]
    scale = target_std / torch.sqrt(source_std.square() + EPSILON)
    shift = target_mean - scale * source_mean
    with torch.no_grad():
        conv.lin.weight.mul_(scale.reshape(-1, 1))
        if conv.bias is None:
            conv.bias = nn.Parameter(
                shift.detach().clone(), requires_grad=conv.lin.weight.requires_grad
            )
        else:
            conv.bias.copy_(scale * conv.bias + shift)


def repair_gcn(
    reference: GCN,
    aligned: GCN,
    graph,
    alpha: float,
) -> tuple[GCN, dict]:
    """Interpolate aligned endpoints and sequentially repair hidden moments."""
    _validate_model(reference)
    _validate_model(aligned)
    _train_mask(graph)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError("alpha must be a finite number in [0, 1]") from error
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")

    endpoint_a = hidden_statistics(reference, graph)
    endpoint_b = hidden_statistics(aligned, graph)
    if alpha == 0.0:
        repaired = copy.deepcopy(reference).eval()
    elif alpha == 1.0:
        repaired = copy.deepcopy(aligned).eval()
    else:
        repaired = interpolate_models(reference, aligned, alpha)

    layers: dict[str, dict[str, float]] = {}
    for index in range(len(repaired.convs) - 1):
        name = f"convs.{index}"
        source = hidden_statistics(repaired, graph)[name]
        target = (
            endpoint_a[name][0] * (1.0 - alpha) + endpoint_b[name][0] * alpha,
            endpoint_a[name][1] * (1.0 - alpha) + endpoint_b[name][1] * alpha,
        )
        if 0.0 < alpha < 1.0:
            _fuse_correction(repaired, index, *source, *target)
        post = hidden_statistics(repaired, graph)[name]
        layers[name] = _layer_diagnostics(source, target, post)

    return repaired.eval(), {
        "alpha": alpha,
        "epsilon": EPSILON,
        "layers": layers,
    }


__all__ = ["align_gcn", "hidden_statistics", "repair_gcn"]
