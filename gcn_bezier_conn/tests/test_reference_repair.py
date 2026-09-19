from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from gcn_mc.reference_models import ReferenceModel
from gcn_mc.reference_repair import (
    ReferenceRepairModel,
    align_reference,
    load_repaired_model,
    reference_statistics,
    repair_reference,
)
from gcn_mc.reference_repair import _permute_stage
from gcn_mc.reference_repair import EPSILON, interpolate_models
from gcn_mc.paths import calibrate_batchnorm


def graph():
    return SimpleNamespace(
        x=torch.tensor([[1., 0., -1.], [0., 2., 1.], [1., -1., .5], [-2., 1., 0.], [.5, .2, 2.]]),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 0, 2, 4], [1, 2, 3, 4, 0, 3, 0, 1]]),
        train_mask=torch.tensor([True, True, True, False, False]),
        y=torch.tensor([0, 1, 0, 1, 1]),
    )


def model(architecture, *, normalization="batch", pre_linear=True):
    return ReferenceModel(3, 4, 2, depth=2, dropout=0., normalization=normalization,
                          residual=True, pre_linear=pre_linear, heads=1,
                          architecture=architecture).eval()


class ReferenceRepairTests(unittest.TestCase):
    def test_high_precision_fallback_rejects_a_changed_function(self):
        data = graph()
        torch.manual_seed(31)
        first = model("mlp", normalization="none", pre_linear=False)
        torch.manual_seed(37)
        second = model("mlp", normalization="none", pre_linear=False)

        def corrupt(endpoint, stage, order):
            _permute_stage(endpoint, stage, order)
            with torch.no_grad():
                endpoint.pred_local.bias[0].add_(.1)

        with mock.patch("gcn_mc.reference_repair._permute_stage", side_effect=corrupt):
            with self.assertRaises(AssertionError):
                align_reference(first, second, data)

    def test_known_nonidentity_permutations_preserve_logits_including_gat_attention(self):
        data = graph()
        order = torch.tensor([2, 0, 3, 1])
        for architecture in ("gcn", "mlp", "graphsage", "gat"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(1)
                endpoint = model(architecture, normalization="layer")
                expected = endpoint(data.x, data.edge_index)
                for stage in range(3):
                    _permute_stage(endpoint, stage, order)
                torch.testing.assert_close(endpoint(data.x, data.edge_index), expected,
                                           rtol=1e-5, atol=1e-6)

    def test_alignment_preserves_logits_for_every_reference_architecture(self):
        data = graph()
        for architecture in ("gcn", "mlp", "graphsage", "gat"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(2)
                first = model(architecture)
                torch.manual_seed(7)
                second = model(architecture)
                first.train()
                first.local_convs[0].eval()
                modes = [module.training for module in first.modules()]
                expected = second(data.x, data.edge_index)
                aligned, diagnostics = align_reference(first, second, data)
                torch.testing.assert_close(aligned(data.x, data.edge_index), expected,
                                           rtol=1e-5, atol=1e-6)
                self.assertEqual(set(diagnostics["permutations"]),
                                 {"lin_in", "local_convs.0", "local_convs.1"})
                self.assertEqual([module.training for module in first.modules()], modes)

    def test_repair_matches_weighted_training_moments_and_ignores_labels(self):
        data = graph()
        torch.manual_seed(11)
        first = model("gat", normalization="layer")
        torch.manual_seed(13)
        second = model("gat", normalization="layer")
        aligned, _ = align_reference(first, second, data)
        repaired, diagnostics = repair_reference(first, aligned, data, .4)
        self.assertIsInstance(repaired, ReferenceRepairModel)
        self.assertEqual(set(next(iter(diagnostics["layers"].values()))),
                         {"pre_mean_variance", "target_mean_variance", "post_mean_variance",
                          "max_mean_target_residual", "max_std_target_residual"})
        frozen_buffers = {name: value.clone() for name, value in repaired.named_buffers()}
        target_a = reference_statistics(first, data)
        target_b = reference_statistics(aligned, data)
        actual = reference_statistics(repaired, data)
        for name in actual:
            torch.testing.assert_close(actual[name][0], .6 * target_a[name][0] + .4 * target_b[name][0], atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(actual[name][1], .6 * target_a[name][1] + .4 * target_b[name][1], atol=2e-4, rtol=2e-4)
        relabeled = SimpleNamespace(**vars(data))
        relabeled.y = 9 - data.y
        repeated, _ = repair_reference(first, aligned, relabeled, .4)
        for left, right in zip(repaired.state_dict().values(), repeated.state_dict().values()):
            torch.testing.assert_close(left, right)
        for name, value in repaired.named_buffers():
            torch.testing.assert_close(value, frozen_buffers[name])

    def test_endpoints_identity_wrapper_and_checkpoint_replay(self):
        data = graph()
        torch.manual_seed(17)
        first = model("graphsage", normalization="none", pre_linear=False)
        torch.manual_seed(19)
        second = model("graphsage", normalization="none", pre_linear=False)
        aligned, _ = align_reference(first, second, data)
        left, _ = repair_reference(first, aligned, data, 0.)
        right, _ = repair_reference(first, aligned, data, 1.)
        self.assertIs(type(left), ReferenceModel)
        torch.testing.assert_close(left(data.x, data.edge_index), first(data.x, data.edge_index))
        torch.testing.assert_close(right(data.x, data.edge_index), aligned(data.x, data.edge_index))
        wrapped = ReferenceRepairModel(first)
        torch.testing.assert_close(wrapped(data.x, data.edge_index), first(data.x, data.edge_index))

        config = {"family": "tunedgnn", "architecture": "graphsage", "in_channels": 3,
                  "hidden_channels": 4, "out_channels": 2, "depth": 2, "dropout": 0.,
                  "normalization": "none", "residual": True, "pre_linear": False, "heads": 1}
        with tempfile.NamedTemporaryFile(suffix=".pt") as target:
            torch.save({"repair_format": "reference-affine-v1", "model_config": config,
                        "state_dict": wrapped.state_dict()}, target.name)
            loaded = load_repaired_model(target.name)
        torch.testing.assert_close(loaded(data.x, data.edge_index), wrapped(data.x, data.edge_index))

    def test_batchnorm_base_is_calibrated_once_and_frozen_during_repair(self):
        data = graph()
        torch.manual_seed(23)
        first = model("gcn", normalization="batch")
        torch.manual_seed(29)
        second = model("gcn", normalization="batch")
        aligned, _ = align_reference(first, second, data)
        alpha = .35
        expected_base = interpolate_models(first, aligned, alpha)
        calibrate_batchnorm(expected_base, data)
        source_mean, source_std = reference_statistics(expected_base, data)["lin_in"]
        stats_a = reference_statistics(first, data)["lin_in"]
        stats_b = reference_statistics(aligned, data)["lin_in"]
        target_mean = (1 - alpha) * stats_a[0] + alpha * stats_b[0]
        target_std = (1 - alpha) * stats_a[1] + alpha * stats_b[1]

        repaired, _ = repair_reference(first, aligned, data, alpha)
        expected_scale = target_std / torch.sqrt(source_std.square() + EPSILON)
        torch.testing.assert_close(repaired.corrections[0].scale, expected_scale)
        torch.testing.assert_close(repaired.corrections[0].shift,
                                   target_mean - expected_scale * source_mean)
        expected_buffers = {name: value.clone() for name, value in expected_base.named_buffers()}
        before = {name: value.clone() for name, value in repaired.base.named_buffers()}
        reference_statistics(repaired, data)
        repaired(data.x, data.edge_index)
        for name, value in repaired.base.named_buffers():
            torch.testing.assert_close(value, expected_buffers[name])
            torch.testing.assert_close(value, before[name])


if __name__ == "__main__":
    unittest.main()
