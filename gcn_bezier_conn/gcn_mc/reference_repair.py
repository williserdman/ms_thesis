"""Channel alignment and affine REPAIR for the tunedGNN reference models."""

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
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from .model import build_model
from .paths import calibrate_batchnorm
from .reference_models import ReferenceModel


EPSILON = 1e-5
CORRELATION_EPSILON = 1e-4
_REPAIR_SRC = Path(__file__).resolve().parents[2] / "repair" / "src"
if str(_REPAIR_SRC) not in sys.path:
    sys.path.insert(0, str(_REPAIR_SRC))
interpolate_models = importlib.import_module("repair.core").interpolate


class _Affine(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("shift", torch.zeros(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale + self.shift


def _stage_names(model: ReferenceModel) -> list[str]:
    return (["lin_in"] if model.pre_linear else []) + [
        f"local_convs.{index}" for index in range(len(model.local_convs))
    ]


class ReferenceRepairModel(nn.Module):
    """A reference model with corrections at complete hidden-stage outputs."""

    def __init__(self, base: ReferenceModel) -> None:
        super().__init__()
        if not isinstance(base, ReferenceModel):
            raise TypeError("base must be a ReferenceModel")
        self.base = copy.deepcopy(base).eval()
        self.stage_names = _stage_names(base)
        weight = base.pred_local.weight
        self.corrections = nn.ModuleList([
            _Affine(base.pred_local.in_features).to(device=weight.device, dtype=weight.dtype)
            for _ in self.stage_names
        ])
        self.eval()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        correction = 0
        if self.base.pre_linear:
            x = self.corrections[correction](self.base.lin_in(x))
            correction += 1
            x = F.dropout(x, p=self.base.dropout, training=self.training)
        for index, block in enumerate(self.base.local_convs):
            transformed = block(x) if self.base.architecture == "mlp" else block(x, edge_index)
            x = transformed + self.base.lins[index](x) if self.base.res else transformed
            if self.base.ln:
                x = self.base.lns[index](x)
            elif self.base.bn:
                x = self.base.bns[index](x)
            x = self.corrections[correction](x)
            correction += 1
            x = F.relu(x)
            x = F.dropout(x, p=self.base.dropout, training=self.training)
        return self.base.pred_local(x)


def _validate(model: nn.Module) -> ReferenceModel:
    base = model.base if isinstance(model, ReferenceRepairModel) else model
    if not isinstance(base, ReferenceModel):
        raise TypeError("reference REPAIR requires ReferenceModel endpoints")
    return base


def _forward_stages(model: nn.Module, graph) -> dict[str, torch.Tensor]:
    base = _validate(model)
    wrapper = model if isinstance(model, ReferenceRepairModel) else None
    corrections = iter(wrapper.corrections) if wrapper is not None else None
    stages: dict[str, torch.Tensor] = {}
    x = graph.x
    if base.pre_linear:
        x = base.lin_in(x)
        if corrections is not None:
            x = next(corrections)(x)
        stages["lin_in"] = x
        x = F.dropout(x, p=base.dropout, training=model.training)
    for index, block in enumerate(base.local_convs):
        transformed = block(x) if base.architecture == "mlp" else block(x, graph.edge_index)
        x = transformed + base.lins[index](x) if base.res else transformed
        if base.ln:
            x = base.lns[index](x)
        elif base.bn:
            x = base.bns[index](x)
        if corrections is not None:
            x = next(corrections)(x)
        stages[f"local_convs.{index}"] = x
        x = F.relu(x)
        x = F.dropout(x, p=base.dropout, training=model.training)
    return stages


def _train_mask(graph) -> torch.Tensor:
    mask = getattr(graph, "train_mask", None)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1 or not bool(mask.any()):
        raise ValueError("graph.train_mask must be a nonempty boolean node mask")
    return mask


def reference_statistics(model: nn.Module, graph) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return train-node moments immediately before each stage's ReLU/dropout."""
    mask = _train_mask(graph)
    modes = [module.training for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            stages = _forward_stages(model, graph)
    finally:
        for module, training in zip(model.modules(), modes):
            module.training = training
    result = {}
    for name, value in stages.items():
        selected = value[mask.to(value.device)].to(torch.float64)
        result[name] = (selected.mean(0).to(value.dtype), selected.std(0, unbiased=False).to(value.dtype))
    return result


def _correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left, right = left.to(torch.float64), right.to(device=left.device, dtype=torch.float64)
    left, right = left - left.mean(0), right - right.mean(0)
    return (left.T @ right / left.shape[0]) / (
        torch.outer(left.square().mean(0).sqrt(), right.square().mean(0).sqrt()) + CORRELATION_EPSILON
    )


def _rows(linear: nn.Linear, order: torch.Tensor) -> None:
    order = order.to(linear.weight.device)
    linear.weight.copy_(linear.weight.index_select(0, order))
    if linear.bias is not None:
        linear.bias.copy_(linear.bias.index_select(0, order))


def _columns(linear: nn.Linear, order: torch.Tensor) -> None:
    linear.weight.copy_(linear.weight.index_select(1, order.to(linear.weight.device)))


def _block_output(block: nn.Module, order: torch.Tensor) -> None:
    if isinstance(block, nn.Linear):
        _rows(block, order)
    elif isinstance(block, GCNConv):
        _rows(block.lin, order)
        if block.bias is not None:
            block.bias.copy_(block.bias.index_select(0, order.to(block.bias.device)))
    elif isinstance(block, SAGEConv):
        _rows(block.lin_l, order)
        _rows(block.lin_r, order)
    elif isinstance(block, GATConv):
        projection = block.lin if block.lin is not None else block.lin_src
        _rows(projection, order)
        block.att_src.copy_(block.att_src.index_select(-1, order.to(block.att_src.device)))
        block.att_dst.copy_(block.att_dst.index_select(-1, order.to(block.att_dst.device)))
        if block.bias is not None:
            block.bias.copy_(block.bias.index_select(0, order.to(block.bias.device)))
    else:  # pragma: no cover
        raise TypeError(type(block).__name__)


def _block_input(block: nn.Module, order: torch.Tensor) -> None:
    if isinstance(block, nn.Linear):
        _columns(block, order)
    elif isinstance(block, GCNConv):
        _columns(block.lin, order)
    elif isinstance(block, SAGEConv):
        _columns(block.lin_l, order)
        _columns(block.lin_r, order)
    elif isinstance(block, GATConv):
        projection = block.lin if block.lin is not None else block.lin_src
        _columns(projection, order)
    else:  # pragma: no cover
        raise TypeError(type(block).__name__)


def _permute_stage(model: ReferenceModel, stage: int, order: torch.Tensor) -> None:
    names = _stage_names(model)
    name = names[stage]
    with torch.no_grad():
        if name == "lin_in":
            _rows(model.lin_in, order)
            block_index = 0
        else:
            block_index = int(name.rsplit(".", 1)[1])
            _block_output(model.local_convs[block_index], order)
            if model.res:
                _rows(model.lins[block_index], order)
            if model.ln:
                _rows_norm(model.lns[block_index], order)
            elif model.bn:
                _rows_norm(model.bns[block_index], order)
            block_index += 1
        if block_index < len(model.local_convs):
            _block_input(model.local_convs[block_index], order)
            if model.res:
                _columns(model.lins[block_index], order)
        else:
            _columns(model.pred_local, order)


def _rows_norm(norm: nn.Module, order: torch.Tensor) -> None:
    for name in ("weight", "bias", "running_mean", "running_var"):
        value = getattr(norm, name, None)
        if value is not None:
            value.copy_(value.index_select(0, order.to(value.device)))


def align_reference(reference: ReferenceModel, candidate: ReferenceModel, graph) -> tuple[ReferenceModel, dict]:
    """Align candidate hidden channels while preserving its graph function."""
    left, right = _validate(reference), _validate(candidate)
    if left.state_dict().keys() != right.state_dict().keys():
        raise ValueError("reference endpoints must have matching architectures")
    mask = _train_mask(graph)
    aligned_reference = copy.deepcopy(left).eval()
    aligned = copy.deepcopy(right).eval()
    original = copy.deepcopy(right).eval()
    permutations = {}
    for stage, name in enumerate(_stage_names(aligned)):
        with torch.no_grad():
            a = _forward_stages(aligned_reference, graph)[name]
            b = _forward_stages(aligned, graph)[name]
        selected_a = a[mask.to(a.device)]
        selected_b = b[mask.to(b.device)]
        if name != "lin_in":
            selected_a, selected_b = F.relu(selected_a), F.relu(selected_b)
        _, columns = linear_sum_assignment(_correlation(selected_a, selected_b).cpu().numpy(), maximize=True)
        order = torch.as_tensor(columns, dtype=torch.long)
        _permute_stage(aligned, stage, order)
        permutations[name] = order.tolist()
    with torch.no_grad():
        expected = original(graph.x, graph.edge_index)
        actual = aligned(graph.x, graph.edge_index)
    error = float((actual - expected).abs().max())
    prediction_disagreements = int((actual.argmax(dim=-1) != expected.argmax(dim=-1)).sum())
    verification_dtype = str(actual.dtype)
    high_precision_error = None
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    except AssertionError:
        candidate64 = original.double()
        aligned64 = copy.deepcopy(aligned).double()
        with torch.no_grad():
            expected64 = candidate64(graph.x.double(), graph.edge_index)
            actual64 = aligned64(graph.x.double(), graph.edge_index)
        high_precision_error = float((actual64 - expected64).abs().max())
        torch.testing.assert_close(actual64, expected64, rtol=1e-9, atol=1e-10)
        verification_dtype = str(actual64.dtype)
    return aligned, {"permutations": permutations, "max_logit_error": error,
                     "prediction_disagreements": prediction_disagreements,
                     "verification_dtype": verification_dtype,
                     "high_precision_max_logit_error": high_precision_error}


def repair_reference(reference: ReferenceModel, aligned: ReferenceModel, graph, alpha: float) -> tuple[nn.Module, dict]:
    """Interpolate calibrated endpoints and sequentially correct stage moments."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    _validate(reference); _validate(aligned); _train_mask(graph)
    if alpha in (0., 1.):
        return copy.deepcopy(reference if alpha == 0. else aligned).eval(), {"alpha": alpha, "epsilon": EPSILON, "layers": {}}
    endpoint_a, endpoint_b = reference_statistics(reference, graph), reference_statistics(aligned, graph)
    merged = interpolate_models(reference, aligned, alpha)
    calibrate_batchnorm(merged, graph)
    repaired = ReferenceRepairModel(merged).eval()
    diagnostics = {}
    for index, name in enumerate(repaired.stage_names):
        source_mean, source_std = reference_statistics(repaired, graph)[name]
        target_mean = endpoint_a[name][0] * (1 - alpha) + endpoint_b[name][0] * alpha
        target_std = endpoint_a[name][1] * (1 - alpha) + endpoint_b[name][1] * alpha
        scale = target_std / torch.sqrt(source_std.square() + EPSILON)
        shift = target_mean - scale * source_mean
        correction = repaired.corrections[index]
        correction.scale.copy_(scale)
        correction.shift.copy_(shift)
        post_mean, post_std = reference_statistics(repaired, graph)[name]
        diagnostics[name] = {"pre_mean_variance": float(source_std.to(torch.float64).square().mean()),
                             "target_mean_variance": float(target_std.to(torch.float64).square().mean()),
                             "post_mean_variance": float(post_std.to(torch.float64).square().mean()),
                             "max_mean_target_residual": float((post_mean-target_mean).abs().max()),
                             "max_std_target_residual": float((post_std-target_std).abs().max())}
    return repaired, {"alpha": alpha, "epsilon": EPSILON, "layers": diagnostics}


def load_repaired_model(checkpoint, device: str | torch.device = "cpu") -> nn.Module:
    """Load either a normal endpoint checkpoint or reference-affine-v1 wrapper."""
    record = torch.load(checkpoint, map_location=device, weights_only=True) if not isinstance(checkpoint, dict) else checkpoint
    base = build_model(record["model_config"]).to(device)
    if record.get("repair_format") == "reference-affine-v1":
        model: nn.Module = ReferenceRepairModel(base).to(device)
    else:
        model = base
    model.load_state_dict(record["state_dict"])
    return model.eval()


__all__ = ["ReferenceRepairModel", "align_reference", "load_repaired_model", "reference_statistics", "repair_reference"]
