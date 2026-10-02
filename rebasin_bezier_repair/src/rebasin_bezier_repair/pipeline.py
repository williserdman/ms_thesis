"""Compose alignment, fixed-endpoint curve fitting, and per-sample REPAIR."""

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from git_re_basin import apply_permutation, validate_spec, validate_state_pair, weight_matching
from torch import Tensor, nn

from .adapters.base import ArchitectureAdapter, ObservationSelector, State
from .paths import _fixed_state, _fraction, bezier_state, fit_curve
from .repair import Moments, collect_moments, repair_state


def _check_source(data, name):
    iterator = iter(data)
    if iterator is data:
        raise TypeError(f"{name} data must be re-iterable")
    try:
        next(iterator)
    except StopIteration as error:
        raise ValueError(f"{name} data must be nonempty") from error


@dataclass
class ConnectivityPath:
    """Frozen-endpoint path; callers provide data when repairing interior points."""

    endpoint_a: State
    endpoint_b: State
    aligned_b: State
    control: State
    permutation: dict[str, Tensor]
    endpoint_moments: tuple[Moments, Moments]
    _template: nn.Module = field(repr=False)
    _adapter: ArchitectureAdapter = field(repr=False)
    _selector: ObservationSelector | None = field(default=None, repr=False)
    _repair_eps: float = field(default=1e-8, repr=False)

    @torch.no_grad()
    def model_at(
        self, t: float, repair: bool = False,
        calibration_data: Iterable[object] | None = None,
    ) -> nn.Module:
        """Return an independent eval model; REPAIR is bypassed at exact endpoints."""
        t = _fraction(t)
        names = tuple(dict(self._template.named_parameters()))
        state = bezier_state(self.endpoint_a, self.aligned_b, self.control, t,
                             parameter_names=names)
        if repair:
            state = repair_state(self._template, self._adapter, state, t,
                                 self.endpoint_moments, calibration_data,
                                 selector=self._selector, eps=self._repair_eps)
        model = copy.deepcopy(self._template)
        model.load_state_dict(state)
        return model.eval()


def connect(
    model_a: nn.Module, model_b: nn.Module, *, adapter: ArchitectureAdapter,
    train_data: Iterable[object], loss_fn: Callable[[Tensor, object], Tensor],
    calibration_data: Iterable[object],
    select_observations: ObservationSelector | None = None,
    curve_steps: int = 100, curve_lr: float = 0.01,
    seed: int = 0, matching_max_iter: int = 100,
    repair_eps: float = 1e-8, invariance_rtol: float = 1e-4,
    invariance_atol: float = 1e-5,
) -> ConnectivityPath:
    """Align B, fit a quadratic control, and cache aligned endpoint moments.

    Loss and observation selection own the task's training/calibration masks.
    Models and data must already share device, dtype, preprocessing, and class order.
    """
    if not math.isfinite(repair_eps) or repair_eps <= 0:
        raise ValueError("repair epsilon must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in (invariance_rtol, invariance_atol)):
        raise ValueError("invariance tolerances must be finite and nonnegative")
    _check_source(train_data, "training")
    _check_source(calibration_data, "calibration")
    for model in (model_a, model_b):
        if any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                   nn.SyncBatchNorm)) for module in model.modules()):
            raise ValueError("native BatchNorm models are not supported")
    structure_a = [(name, type(module)) for name, module in model_a.named_modules()]
    structure_b = [(name, type(module)) for name, module in model_b.named_modules()]
    if structure_a != structure_b:
        raise ValueError("endpoint module architectures differ")
    a = {name: value.detach().clone() for name, value in model_a.state_dict().items()}
    b = {name: value.detach().clone() for name, value in model_b.state_dict().items()}
    validate_state_pair(a, b)
    spec = adapter.permutation_spec(model_a, a)
    if spec != adapter.permutation_spec(model_b, b):
        raise ValueError("endpoint permutation specifications differ")
    validate_spec(spec, a)
    if adapter.repair_sites(model_a) != adapter.repair_sites(model_b):
        raise ValueError("endpoint repair sites differ")
    permutation = weight_matching(spec, a, b, seed=seed, max_iter=matching_max_iter)
    aligned = apply_permutation(spec, permutation, b)
    names = tuple(dict(model_a.named_parameters()))
    _fixed_state(a, aligned, names)
    template = copy.deepcopy(model_a).eval()
    original_b = copy.deepcopy(model_b).eval()
    for parameter in template.parameters():
        parameter.grad = None
    with torch.no_grad():
        for batch in calibration_data:
            reference = adapter.forward(original_b, b, batch)
            actual = adapter.forward(template, aligned, batch)
            try:
                torch.testing.assert_close(actual, reference, rtol=invariance_rtol,
                                           atol=invariance_atol)
            except AssertionError as error:
                raise ValueError("alignment does not preserve endpoint B logits") from error
    control = fit_curve(template, adapter, a, aligned, train_data, loss_fn,
                        steps=curve_steps, lr=curve_lr, seed=seed)
    moments = tuple(collect_moments(template, adapter, state, calibration_data,
                                    selector=select_observations) for state in (a, aligned))
    return ConnectivityPath(a, b, aligned, control, permutation, moments,
                            template, adapter, select_observations, repair_eps)
