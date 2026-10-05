"""Behavior checks for the pinned tunedGNN model adaptation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from gcn_mc.model import GCN, build_model
from gcn_mc.presets import reference_profile


def _load_upstream_model_class():
    path = (
        Path(__file__).resolve().parents[1]
        / "upstream"
        / "tunedGNN"
        / "medium_graph"
        / "model.py"
    )
    spec = importlib.util.spec_from_file_location("pinned_tunedgnn_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MPNNs


class ModelFactoryTests(unittest.TestCase):
    def test_missing_family_replays_legacy_gcn_predictions(self):
        config = {
            "in_channels": 3,
            "hidden_channels": 5,
            "out_channels": 2,
            "depth": 3,
            "dropout": 0.0,
        }
        torch.manual_seed(4)
        expected = GCN(**config).eval()
        actual = build_model(config).eval()
        actual.load_state_dict(expected.state_dict())
        x = torch.randn(6, 3)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 2]],
            dtype=torch.long,
        )

        self.assertIs(type(actual), GCN)
        torch.testing.assert_close(actual(x, edge_index), expected(x, edge_index))

    def test_reference_graph_models_match_pinned_upstream_predictions(self):
        upstream_class = _load_upstream_model_class()
        x = torch.randn(7, 4)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 0, 3]],
            dtype=torch.long,
        )
        cases = (
            ("gcn", "gcn", "batch", True, True, 4),
            ("graphsage", "sage", "layer", False, False, 3),
            ("gat", "gat", "none", True, False, 3),
        )
        for architecture, upstream_name, normalization, residual, pre_linear, depth in cases:
            with self.subTest(architecture=architecture):
                config = {
                    "family": "tunedgnn",
                    "architecture": architecture,
                    "in_channels": 4,
                    "hidden_channels": 5,
                    "out_channels": 3,
                    "depth": depth,
                    "dropout": 0.0,
                    "normalization": normalization,
                    "residual": residual,
                    "pre_linear": pre_linear,
                    "heads": 1,
                }
                actual = build_model(config).eval()
                expected = upstream_class(
                    4,
                    5,
                    3,
                    local_layers=depth,
                    dropout=0.0,
                    heads=1,
                    pre_linear=pre_linear,
                    res=residual,
                    ln=normalization == "layer",
                    bn=normalization == "batch",
                    gnn=upstream_name,
                ).eval()
                actual.load_state_dict(expected.state_dict(), strict=True)

                torch.testing.assert_close(
                    actual(x, edge_index), expected(x, edge_index), rtol=0, atol=0
                )

    def test_reference_mlp_is_independent_of_edges(self):
        model = build_model(
            {
                "family": "tunedgnn",
                "architecture": "mlp",
                "in_channels": 3,
                "hidden_channels": 6,
                "out_channels": 2,
                "depth": 3,
                "dropout": 0.0,
                "normalization": "layer",
                "residual": True,
                "pre_linear": False,
                "heads": 1,
            }
        ).eval()
        x = torch.randn(5, 3)
        sparse_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        dense_edges = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 0, 3, 1, 4, 4, 0]],
            dtype=torch.long,
        )

        torch.testing.assert_close(model(x, sparse_edges), model(x, dense_edges))

    def test_reference_gat_rejects_unsupported_multihead_width(self):
        with self.assertRaisesRegex(ValueError, "heads=1"):
            build_model(
                {
                    "family": "tunedgnn",
                    "architecture": "gat",
                    "in_channels": 3,
                    "hidden_channels": 6,
                    "out_channels": 2,
                    "depth": 2,
                    "dropout": 0.0,
                    "normalization": "none",
                    "residual": False,
                    "pre_linear": False,
                    "heads": 2,
                }
            )


class ReferenceProfileTests(unittest.TestCase):
    def test_profiles_match_pinned_commands_and_mlp_uses_gcn_recipe(self):
        expected = {
            ("Cora", "gcn"): (512, 3, 0.7, "none", False, False, 500, 0.001, 5e-4),
            ("Squirrel", "graphsage"): (256, 3, 0.7, "batch", True, False, 500, 0.01, 5e-4),
            ("Roman-Empire", "gat"): (512, 10, 0.3, "batch", True, True, 2500, 0.001, 0.0),
            ("Chameleon", "graphsage"): (256, 4, 0.7, "batch", True, False, 200, 0.01, 0.001),
        }
        for (dataset, architecture), values in expected.items():
            with self.subTest(dataset=dataset, architecture=architecture):
                profile = reference_profile(dataset, architecture)
                model = profile["model"]
                training = profile["training"]
                self.assertEqual(
                    (
                        model["hidden_channels"],
                        model["depth"],
                        model["dropout"],
                        model["normalization"],
                        model["residual"],
                        model["pre_linear"],
                        training["epochs"],
                        training["lr"],
                        training["weight_decay"],
                    ),
                    values,
                )
                self.assertEqual(model["heads"], 1)
                self.assertEqual(training["selection"], "val_accuracy")
                self.assertEqual(
                    profile["source"]["commit"],
                    "23f9604e8b13a9a6d3faa2f691cd844006979153",
                )

        self.assertEqual(
            reference_profile("cora", "mlp")["model"],
            reference_profile("cora", "gcn")["model"],
        )

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown reference dataset"):
            reference_profile("not-a-dataset", "gcn")


if __name__ == "__main__":
    unittest.main()
