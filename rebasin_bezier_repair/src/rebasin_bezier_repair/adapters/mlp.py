"""Sequential MLP adapter; axis rules follow git_re_basin/adapters.py."""

from git_re_basin import validate_spec
from torch import Tensor, nn
from torch.func import functional_call

from .base import RepairSite, State, correct_affine_rows


class MLPAdapter:
    """Biased Linear/ReLU stacks with a fixed-order Linear classifier."""

    @staticmethod
    def _linear_layers(model):
        if type(model) is not nn.Sequential:
            raise ValueError("MLPAdapter requires a plain nn.Sequential")
        children = list(model.named_children())
        if len(children) < 3 or len(children) % 2 != 1:
            raise ValueError("expected alternating Linear/ReLU ending in Linear")
        for index, (_, layer) in enumerate(children):
            expected = nn.Linear if index % 2 == 0 else nn.ReLU
            if type(layer) is not expected:
                raise ValueError("only plain Linear/ReLU stacks are supported")
            if expected is nn.Linear and layer.bias is None:
                raise ValueError("MLP repair requires biased Linear layers")
        return children[::2]

    def permutation_spec(self, model: nn.Module, state: State):
        layers = self._linear_layers(model)
        spec = {name: (None,) * value.ndim for name, value in state.items()}
        for index, (name, _) in enumerate(layers):
            incoming = None if index == 0 else f"hidden_{index - 1}"
            outgoing = None if index == len(layers) - 1 else f"hidden_{index}"
            spec[f"{name}.weight"] = (outgoing, incoming)
            spec[f"{name}.bias"] = (outgoing,)
        validate_spec(spec, state)
        return spec

    def forward(self, model: nn.Module, state: State, batch: object) -> Tensor:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        return functional_call(model, state, (inputs,), strict=True)

    def repair_sites(self, model: nn.Module) -> tuple[RepairSite, ...]:
        return tuple(RepairSite(name, name, -1)
                     for name, _ in self._linear_layers(model)[:-1])

    def apply_correction(self, state, site, scale, shift):
        return correct_affine_rows(
            state, f"{site.module_path}.weight", f"{site.module_path}.bias", scale, shift,
        )
