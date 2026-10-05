"""Behavior tests for channel alignment and the supported model recipes."""

import copy
import unittest

import torch
from torch import nn

from repair.alignment import align_models, hidden_layer_names
from repair.models import tiny_mlp, vgg11


def _permute_hidden(model, layer_index, next_layer_index, permutation):
    """Apply a function-preserving hidden-unit permutation to an MLP."""
    current = model[layer_index]
    following = model[next_layer_index]
    with torch.no_grad():
        current.weight.copy_(current.weight[permutation])
        if current.bias is not None:
            current.bias.copy_(current.bias[permutation])
        following.weight.copy_(following.weight[:, permutation])


class ModelRecipeTest(unittest.TestCase):
    def test_tiny_mlp_has_named_hidden_affine_layers(self):
        model = tiny_mlp(input_dim=5, hidden_dims=(7, 6), num_classes=3)

        self.assertEqual(hidden_layer_names(model), ["0", "2"])
        self.assertEqual(tuple(model(torch.zeros(4, 5)).shape), (4, 3))

    def test_vgg11_uses_upstream_parameter_names_and_scaled_channels(self):
        model = vgg11(width=1 / 64, num_classes=7)

        self.assertEqual(
            hidden_layer_names(model),
            [
                "features.0", "features.3", "features.6", "features.8",
                "features.11", "features.13", "features.16", "features.18",
            ],
        )
        self.assertEqual(tuple(model.features[0].weight.shape), (1, 3, 3, 3))
        self.assertEqual(tuple(model.features[3].weight.shape), (2, 1, 3, 3))
        self.assertEqual(tuple(model.classifier.weight.shape), (7, 8))
        self.assertEqual(tuple(model(torch.zeros(2, 3, 32, 32)).shape), (2, 7))
        self.assertEqual(next(model.parameters()).device.type, "cpu")


class AlignmentTest(unittest.TestCase):
    def test_known_mlp_permutations_are_recovered_without_changing_predictions(self):
        torch.manual_seed(4)
        reference = tiny_mlp(input_dim=4, hidden_dims=(5, 4), num_classes=3)
        with torch.no_grad():
            reference[0].bias.fill_(1.5)
            reference[2].bias.fill_(1.5)
        candidate = copy.deepcopy(reference)
        _permute_hidden(candidate, 0, 2, torch.tensor([2, 4, 0, 1, 3]))
        _permute_hidden(candidate, 2, 4, torch.tensor([3, 1, 0, 2]))
        calibration = [torch.randn(17, 4), torch.randn(31, 4)]
        probe = torch.randn(13, 4)
        candidate_before = copy.deepcopy(candidate.state_dict())
        reference.train()
        candidate.train()

        torch.testing.assert_close(reference(probe), candidate(probe), rtol=1e-6, atol=1e-7)
        aligned = align_models(reference, candidate, calibration)

        self.assertFalse(aligned.training)
        self.assertTrue(reference.training)
        self.assertTrue(candidate.training)
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, aligned.state_dict()[name]), name)
        for name, value in candidate_before.items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        torch.testing.assert_close(reference(probe), aligned(probe), rtol=1e-6, atol=1e-7)

    def test_alignment_supports_hidden_layers_without_bias(self):
        torch.manual_seed(8)
        reference = nn.Sequential(
            nn.Linear(3, 4, bias=False), nn.ReLU(), nn.Linear(4, 2, bias=False)
        )
        candidate = copy.deepcopy(reference)
        _permute_hidden(candidate, 0, 2, torch.tensor([2, 0, 3, 1]))

        aligned = align_models(reference, candidate, [torch.randn(40, 3)])

        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, aligned.state_dict()[name]), name)

    def test_tiny_vgg_alignment_preserves_candidate_logits(self):
        torch.manual_seed(12)
        reference = vgg11(width=1 / 64, num_classes=3)
        candidate = copy.deepcopy(reference)
        permutation = torch.tensor([1, 0])
        with torch.no_grad():
            candidate.features[3].weight.copy_(candidate.features[3].weight[permutation])
            candidate.features[3].bias.copy_(candidate.features[3].bias[permutation])
            candidate.features[6].weight.copy_(candidate.features[6].weight[:, permutation])
        calibration = [torch.randn(2, 3, 32, 32)]
        probe = torch.randn(2, 3, 32, 32)
        expected = candidate(probe)

        aligned = align_models(reference, candidate, calibration, max_batches=1)

        torch.testing.assert_close(aligned(probe), expected, rtol=1e-5, atol=1e-6)

    def test_incompatible_and_invalid_inputs_are_rejected(self):
        model = tiny_mlp(input_dim=4, hidden_dims=(5,), num_classes=2)
        mismatched = tiny_mlp(input_dim=4, hidden_dims=(6,), num_classes=2)

        with self.assertRaisesRegex(ValueError, "compatible"):
            align_models(model, mismatched, [torch.randn(3, 4)])
        with self.assertRaisesRegex(ValueError, "max_batches"):
            align_models(model, model, [torch.randn(3, 4)], max_batches=0)
        with self.assertRaisesRegex(ValueError, "empty"):
            align_models(model, model, [])
        with self.assertRaisesRegex(TypeError, "re-iterable"):
            align_models(model, model, (x for x in [torch.randn(3, 4)]))
        with self.assertRaisesRegex(TypeError, "unsupported"):
            hidden_layer_names(nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2)))


if __name__ == "__main__":
    unittest.main()
