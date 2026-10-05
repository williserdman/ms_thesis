"""Channel alignment adapted from the official REPAIR implementation.

Source: https://github.com/KellerJordan/REPAIR at commit
e90263d7a4d48376091327274ae541d8d6d34743.
"""

from collections.abc import Iterable, Iterator
import copy
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .models import VGG


@dataclass(frozen=True)
class _HiddenLayer:
    name: str
    affine: nn.Conv2d | nn.Linear
    activation: nn.ReLU
    following: nn.Conv2d | nn.Linear


def _vgg_hidden_layers(model: VGG) -> list[_HiddenLayer]:
    convolution_indices = [
        index for index, module in enumerate(model.features) if isinstance(module, nn.Conv2d)
    ]
    if len(convolution_indices) != 8:
        raise TypeError("unsupported VGG architecture: expected the VGG11 convolution layout")

    result = []
    for position, index in enumerate(convolution_indices):
        affine = model.features[index]
        if index + 1 >= len(model.features) or not isinstance(model.features[index + 1], nn.ReLU):
            raise TypeError("unsupported VGG architecture: every convolution must precede ReLU")
        following = (
            model.features[convolution_indices[position + 1]]
            if position + 1 < len(convolution_indices)
            else model.classifier
        )
        if not isinstance(following, (nn.Conv2d, nn.Linear)):
            raise TypeError("unsupported VGG architecture: invalid classifier")
        result.append(
            _HiddenLayer(
                name=f"features.{index}",
                affine=affine,
                activation=model.features[index + 1],
                following=following,
            )
        )
    return result


def _mlp_hidden_layers(model: nn.Sequential) -> list[_HiddenLayer]:
    modules = list(model)
    if len(modules) < 3 or len(modules) % 2 == 0:
        raise TypeError("unsupported sequential architecture: expected Linear/ReLU pairs and a final Linear")
    for index, module in enumerate(modules):
        expected = nn.Linear if index % 2 == 0 else nn.ReLU
        if not isinstance(module, expected):
            raise TypeError("unsupported sequential architecture: expected Linear/ReLU pairs and a final Linear")

    return [
        _HiddenLayer(
            name=str(index),
            affine=modules[index],
            activation=modules[index + 1],
            following=modules[index + 2],
        )
        for index in range(0, len(modules) - 1, 2)
    ]


def _hidden_layers(model: nn.Module) -> list[_HiddenLayer]:
    if isinstance(model, VGG):
        return _vgg_hidden_layers(model)
    if isinstance(model, nn.Sequential):
        return _mlp_hidden_layers(model)
    raise TypeError(
        "unsupported architecture: expected repair.models.VGG or an alternating Linear/ReLU nn.Sequential"
    )


def hidden_layer_names(model: nn.Module) -> list[str]:
    """Return the affine module names whose outputs can be channel-aligned."""
    return [layer.name for layer in _hidden_layers(model)]


def _check_compatible(reference: nn.Module, candidate: nn.Module) -> None:
    if isinstance(reference, VGG) != isinstance(candidate, VGG):
        raise ValueError("models must have compatible supported architectures")
    if isinstance(reference, nn.Sequential) != isinstance(candidate, nn.Sequential):
        raise ValueError("models must have compatible supported architectures")

    reference_modules = list(reference.modules())
    candidate_modules = list(candidate.modules())
    if len(reference_modules) != len(candidate_modules):
        raise ValueError("models must have compatible module layouts")
    for left, right in zip(reference_modules, candidate_modules):
        if type(left) is not type(right):
            raise ValueError("models must have compatible module layouts")
        if isinstance(left, (nn.Conv2d, nn.Linear)):
            if left.weight.shape != right.weight.shape or (left.bias is None) != (right.bias is None):
                raise ValueError("models must have compatible affine layer shapes")

    reference_layers = _hidden_layers(reference)
    candidate_layers = _hidden_layers(candidate)
    if [layer.name for layer in reference_layers] != [layer.name for layer in candidate_layers]:
        raise ValueError("models must have compatible hidden layers")


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("supported models must contain parameters")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("all model parameters must be on one device")
    return parameters[0].device, parameters[0].dtype


def _inputs_from_batch(batch: object) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise TypeError("calibration batches must be tensors or (input, label) pairs")


def _move_inputs(inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
    device, dtype = _model_device_and_dtype(model)
    if inputs.is_floating_point() or inputs.is_complex():
        return inputs.to(device=device, dtype=dtype)
    return inputs.to(device=device)


def _flatten_activations(activations: torch.Tensor) -> torch.Tensor:
    if activations.ndim < 2:
        raise ValueError("hidden activations must have a channel dimension")
    return (
        activations.detach()
        .movedim(1, -1)
        .reshape(-1, activations.shape[1])
        .to(dtype=torch.float64)
    )


def _activation_correlation(
    reference: nn.Module,
    candidate: nn.Module,
    reference_activation: nn.Module,
    candidate_activation: nn.Module,
    calibration_data: Iterable[object],
    max_batches: int | None,
) -> torch.Tensor:
    captured_reference: list[torch.Tensor] = []
    captured_candidate: list[torch.Tensor] = []

    def capture_reference(_module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        captured_reference[:] = [output]

    def capture_candidate(_module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        captured_candidate[:] = [output]

    reference_hook = reference_activation.register_forward_hook(capture_reference)
    candidate_hook = candidate_activation.register_forward_hook(capture_candidate)
    observation_count = 0
    sum_reference = sum_candidate = None
    square_reference = square_candidate = cross = None
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(calibration_data):
                if max_batches is not None and batch_index >= max_batches:
                    break
                inputs = _inputs_from_batch(batch)
                captured_reference.clear()
                captured_candidate.clear()
                reference(_move_inputs(inputs, reference))
                candidate(_move_inputs(inputs, candidate))
                if not captured_reference or not captured_candidate:
                    raise RuntimeError("failed to capture hidden activations")
                left = _flatten_activations(captured_reference[0])
                right = _flatten_activations(captured_candidate[0]).to(left.device)
                if left.shape[0] != right.shape[0]:
                    raise ValueError("models produced different calibration observation counts")
                if left.shape[1] != right.shape[1]:
                    raise ValueError("models must have equal hidden channel counts")
                if sum_reference is None:
                    sum_reference = torch.zeros(
                        left.shape[1], dtype=torch.float64, device=left.device
                    )
                    sum_candidate = torch.zeros_like(sum_reference)
                    square_reference = torch.zeros_like(sum_reference)
                    square_candidate = torch.zeros_like(sum_candidate)
                    cross = torch.zeros(
                        (left.shape[1], right.shape[1]),
                        dtype=torch.float64,
                        device=left.device,
                    )
                sum_reference += left.sum(dim=0)
                sum_candidate += right.sum(dim=0)
                square_reference += left.square().sum(dim=0)
                square_candidate += right.square().sum(dim=0)
                cross += left.T @ right
                observation_count += left.shape[0]
    finally:
        reference_hook.remove()
        candidate_hook.remove()

    if observation_count == 0:
        raise ValueError("calibration_data is empty")
    assert sum_reference is not None and sum_candidate is not None
    assert square_reference is not None and square_candidate is not None and cross is not None
    mean_reference = sum_reference / observation_count
    mean_candidate = sum_candidate / observation_count
    covariance = cross / observation_count - torch.outer(mean_reference, mean_candidate)
    variance_reference = (square_reference / observation_count - mean_reference.square()).clamp_min(0)
    variance_candidate = (square_candidate / observation_count - mean_candidate.square()).clamp_min(0)
    denominator = torch.outer(variance_reference.sqrt(), variance_candidate.sqrt())
    return (covariance / (denominator + 1e-4)).cpu()


def _permute_layer(layer: _HiddenLayer, permutation: torch.Tensor) -> None:
    output_permutation = permutation.to(layer.affine.weight.device)
    input_permutation = permutation.to(layer.following.weight.device)
    with torch.no_grad():
        layer.affine.weight.copy_(layer.affine.weight.index_select(0, output_permutation))
        if layer.affine.bias is not None:
            layer.affine.bias.copy_(layer.affine.bias.index_select(0, output_permutation))
        layer.following.weight.copy_(layer.following.weight.index_select(1, input_permutation))


def align_models(
    reference: nn.Module,
    candidate: nn.Module,
    calibration_data: Iterable[object],
    *,
    max_batches: int | None = None,
) -> nn.Module:
    """Return an eval-mode copy of ``candidate`` aligned to ``reference``."""
    if max_batches is not None and (
        isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0
    ):
        raise ValueError("max_batches must be a positive integer or None")
    if not isinstance(calibration_data, Iterable):
        raise TypeError("calibration_data must be a re-iterable collection of batches")
    if isinstance(calibration_data, Iterator):
        raise TypeError("calibration_data must be re-iterable; one-shot iterators are unsupported")

    _check_compatible(reference, candidate)
    aligned_reference = copy.deepcopy(reference).eval()
    aligned_candidate = copy.deepcopy(candidate).eval()
    reference_layers = _hidden_layers(aligned_reference)
    candidate_layers = _hidden_layers(aligned_candidate)

    for reference_layer, candidate_layer in zip(reference_layers, candidate_layers):
        correlation = _activation_correlation(
            aligned_reference,
            aligned_candidate,
            reference_layer.activation,
            candidate_layer.activation,
            calibration_data,
            max_batches,
        )
        rows, columns = linear_sum_assignment(correlation.numpy(), maximize=True)
        if not torch.equal(torch.from_numpy(rows), torch.arange(correlation.shape[0])):
            raise RuntimeError("Hungarian assignment did not cover every reference channel")
        _permute_layer(candidate_layer, torch.from_numpy(columns).to(dtype=torch.long))

    return aligned_candidate
