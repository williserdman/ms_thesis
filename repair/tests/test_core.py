import copy
import unittest

import torch
from torch import nn

from repair.core import interpolate, repair


class TwoLayerNet(nn.Module):
    def __init__(self, *, bias=True):
        super().__init__()
        self.first = nn.Linear(1, 1, bias=bias)
        self.relu = nn.ReLU()
        self.second = nn.Linear(1, 1, bias=bias)

    def forward(self, inputs):
        return self.second(self.relu(self.first(inputs)))


def make_pair(*, bias=True):
    model_a = TwoLayerNet(bias=bias)
    model_b = TwoLayerNet(bias=bias)
    with torch.no_grad():
        model_a.first.weight.fill_(1.0)
        model_b.first.weight.fill_(-3.0)
        model_a.second.weight.fill_(1.0)
        model_b.second.weight.fill_(1.0)
        if bias:
            model_a.first.bias.zero_()
            model_b.first.bias.zero_()
            model_a.second.bias.zero_()
            model_b.second.bias.zero_()
    return model_a, model_b


def activations(model, layer_name, inputs):
    values = []
    handle = model.get_submodule(layer_name).register_forward_hook(
        lambda _module, _args, output: values.append(output.detach())
    )
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()
    return torch.cat(values)


class InterpolateTests(unittest.TestCase):
    def test_endpoints_are_copies_and_inputs_are_unchanged(self):
        model_a, model_b = make_pair()
        model_a.train()
        model_b.train()
        before_a = copy.deepcopy(model_a.state_dict())
        before_b = copy.deepcopy(model_b.state_dict())

        at_a = interpolate(model_a, model_b, alpha=0.0)
        at_b = interpolate(model_a, model_b, alpha=1.0)

        self.assertIsNot(at_a, model_a)
        self.assertIsNot(at_b, model_b)
        self.assertFalse(at_a.training)
        self.assertFalse(at_b.training)
        self.assertTrue(model_a.training)
        self.assertTrue(model_b.training)
        for name, value in before_a.items():
            torch.testing.assert_close(at_a.state_dict()[name], value)
            torch.testing.assert_close(model_a.state_dict()[name], value)
        for name, value in before_b.items():
            torch.testing.assert_close(at_b.state_dict()[name], value)
            torch.testing.assert_close(model_b.state_dict()[name], value)

    def test_midpoint_interpolates_parameters(self):
        model_a, model_b = make_pair()
        merged = interpolate(model_a, model_b, alpha=0.25)
        torch.testing.assert_close(
            merged.first.weight, torch.tensor([[0.0]])
        )


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        self.calibration = [(self.inputs, torch.zeros(4, dtype=torch.long))]

    def test_batchnorm_repair_matches_weighted_endpoint_moments(self):
        model_a, model_b = make_pair()

        repaired = repair(
            model_a,
            model_b,
            self.calibration,
            layer_names=["first"],
            method="batchnorm",
            fuse=True,
        )

        actual = activations(repaired, "first", self.inputs)
        endpoint_a = activations(model_a.eval(), "first", self.inputs)
        endpoint_b = activations(model_b.eval(), "first", self.inputs)
        expected_mean = 0.5 * endpoint_a.mean(0) + 0.5 * endpoint_b.mean(0)
        expected_std = 0.5 * endpoint_a.std(0, unbiased=False) + 0.5 * endpoint_b.std(0, unbiased=False)
        torch.testing.assert_close(actual.mean(0), expected_mean, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual.std(0, unbiased=False), expected_std, atol=2e-5, rtol=2e-5)

    def test_sequential_repair_uses_repaired_prefix(self):
        model_a, model_b = make_pair()

        repaired = repair(
            model_a,
            model_b,
            [self.inputs],
            layer_names=["first", "second"],
            method="sequential",
            fuse=True,
        )

        # Repairing ``first`` makes the merged prefix's ReLU moments equal the
        # target, so a sequential measurement leaves ``second`` unchanged.
        torch.testing.assert_close(repaired.second.weight, torch.tensor([[1.0]]))

        for layer_name in ("first", "second"):
            actual = activations(repaired, layer_name, self.inputs)
            endpoint_a = activations(model_a.eval(), layer_name, self.inputs)
            endpoint_b = activations(model_b.eval(), layer_name, self.inputs)
            expected_mean = 0.5 * endpoint_a.mean(0) + 0.5 * endpoint_b.mean(0)
            expected_std = 0.5 * endpoint_a.std(0, unbiased=False) + 0.5 * endpoint_b.std(0, unbiased=False)
            torch.testing.assert_close(actual.mean(0), expected_mean, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(actual.std(0, unbiased=False), expected_std, atol=3e-5, rtol=3e-5)

    def test_fused_and_unfused_outputs_match(self):
        model_a, model_b = make_pair()
        wrapped = repair(
            model_a,
            model_b,
            self.calibration,
            layer_names=["first", "second"],
            method="batchnorm",
            fuse=False,
        )
        fused = repair(
            model_a,
            model_b,
            self.calibration,
            layer_names=["first", "second"],
            method="batchnorm",
            fuse=True,
        )

        with torch.no_grad():
            torch.testing.assert_close(wrapped(self.inputs), fused(self.inputs))

    def test_fusion_adds_bias_to_biasless_layer(self):
        model_a, model_b = make_pair(bias=False)
        repaired = repair(
            model_a,
            model_b,
            self.calibration,
            layer_names=["first"],
            method="sequential",
            fuse=True,
        )

        self.assertIsNotNone(repaired.first.bias)
        self.assertEqual(repaired.first.bias.shape, torch.Size([1]))

    def test_conv2d_fusion_matches_unfused_correction(self):
        model_a = nn.Sequential(nn.Conv2d(1, 2, kernel_size=1, bias=False))
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_a[0].weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]]]))
            model_b[0].weight.copy_(torch.tensor([[[[-3.0]]], [[[4.0]]]]))
        images = torch.tensor(
            [
                [[[-2.0, -1.0], [1.0, 2.0]]],
                [[[-1.0, 0.0], [2.0, 3.0]]],
            ]
        )
        wrapped = repair(
            model_a,
            model_b,
            [images],
            layer_names=["0"],
            method="sequential",
            fuse=False,
        )
        fused = repair(
            model_a,
            model_b,
            [images],
            layer_names=["0"],
            method="sequential",
            fuse=True,
        )

        self.assertIsInstance(fused[0], nn.Conv2d)
        self.assertIsNotNone(fused[0].bias)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(images), fused(images))

    def test_empty_and_one_shot_calibration_are_rejected(self):
        model_a, model_b = make_pair()
        with self.assertRaisesRegex(ValueError, "empty"):
            repair(model_a, model_b, [], layer_names=["first"])
        with self.assertRaisesRegex(TypeError, "re-iterable"):
            repair(
                model_a,
                model_b,
                iter([self.inputs]),
                layer_names=["first"],
            )


if __name__ == "__main__":
    unittest.main()
