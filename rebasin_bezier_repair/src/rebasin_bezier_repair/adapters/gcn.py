"""Plain GCN adapter; correction rules follow sibling gcn_mc/repair.py."""

from git_re_basin import validate_spec
from torch import Tensor, nn
from torch.func import functional_call
from torch_geometric.nn import GCNConv

from .base import RepairSite, State, correct_affine_rows


class GCNAdapter:
    """Biased, uncached GCNConv stacks with fixed input and output order."""

    @staticmethod
    def _convolutions(model: nn.Module) -> nn.ModuleList:
        convs = getattr(model, "convs", None)
        if type(convs) is not nn.ModuleList or len(convs) < 2:
            raise ValueError("GCNAdapter requires a convs ModuleList with at least two layers")
        for conv in convs:
            if type(conv) is not GCNConv:
                raise ValueError("only plain GCNConv layers are supported")
            if conv.cached:
                raise ValueError("cached GCNConv layers are unsupported")
            if conv.bias is None:
                raise ValueError("GCN repair requires biased GCNConv layers")
        expected = {
            name
            for index in range(len(convs))
            for name in (f"convs.{index}.lin.weight", f"convs.{index}.bias")
        }
        if set(model.state_dict()) != expected:
            raise ValueError("GCNAdapter does not support additional parameters or buffers")
        return convs

    def permutation_spec(self, model: nn.Module, state: State):
        convs = self._convolutions(model)
        expected = set(model.state_dict())
        if set(state) != expected:
            raise ValueError("GCN state must match the supported convolution stack")
        spec = {}
        for index in range(len(convs)):
            incoming = None if index == 0 else f"hidden_{index - 1}"
            outgoing = None if index == len(convs) - 1 else f"hidden_{index}"
            spec[f"convs.{index}.lin.weight"] = (outgoing, incoming)
            spec[f"convs.{index}.bias"] = (outgoing,)
        validate_spec(spec, state)
        return spec

    def forward(self, model: nn.Module, state: State, batch: object) -> Tensor:
        self._convolutions(model)
        return functional_call(
            model, state, (batch.x, batch.edge_index), strict=True,
        )

    def repair_sites(self, model: nn.Module) -> tuple[RepairSite, ...]:
        convs = self._convolutions(model)
        return tuple(
            RepairSite(f"convs.{index}", f"convs.{index}", -1)
            for index in range(len(convs) - 1)
        )

    def apply_correction(self, state, site, scale, shift):
        path = site.module_path
        if not path.startswith("convs.") or not path.removeprefix("convs.").isdigit():
            raise ValueError("invalid GCN repair site")
        return correct_affine_rows(
            state, f"{path}.lin.weight", f"{path}.bias", scale, shift,
        )
