"""The architecture-specific seam used by matching, fitting, and REPAIR."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

State = Mapping[str, Tensor]
PermutationSpec = Mapping[str, tuple[str | None, ...]]


@dataclass(frozen=True)
class RepairSite:
    """A preactivation output; correction keys are owned by the adapter."""

    name: str
    module_path: str
    channel_axis: int


ObservationSelector = Callable[[RepairSite, Tensor, object], Tensor]


class ArchitectureAdapter(Protocol):
    def permutation_spec(self, model: nn.Module, state: State) -> PermutationSpec: ...

    def forward(self, model: nn.Module, state: State, batch: object) -> Tensor: ...

    def repair_sites(self, model: nn.Module) -> tuple[RepairSite, ...]: ...

    def apply_correction(
        self, state: State, site: RepairSite, scale: Tensor, shift: Tensor,
    ) -> dict[str, Tensor]: ...


@torch.no_grad()
def correct_affine_rows(
    state: State, weight_name: str, bias_name: str, scale: Tensor, shift: Tensor,
) -> dict[str, Tensor]:
    """Fold a channelwise output correction into copied affine parameters."""
    weight, bias = state[weight_name], state[bias_name]
    if scale.shape != bias.shape or shift.shape != bias.shape or bias.ndim != 1:
        raise ValueError("correction vectors must match the output bias")
    if weight.shape[0] != bias.numel():
        raise ValueError("weight output rows must match the bias")
    result = {key: value.detach().clone() for key, value in state.items()}
    scale, shift = scale.to(weight), shift.to(bias)
    shape = (scale.numel(),) + (1,) * (weight.ndim - 1)
    result[weight_name].mul_(scale.reshape(shape))
    result[bias_name].mul_(scale).add_(shift)
    return result
