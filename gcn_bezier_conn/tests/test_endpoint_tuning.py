"""Endpoint tuning must persist, isolate held-out labels, and preserve curve settings."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from gcn_mc.experiment import Config, resolved_config, run, train_endpoint


def fixture():
    graph = Data(
        x=torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., .5], [.2, -.3], [-.5, 1.]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
    )
    metadata = {"name": "Cora", "num_features": 2, "num_classes": 2,
                "split_sha256": "fixture", "loader_sha256": "fixture"}
    return graph, metadata


class EndpointTuningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_validation_only_training_never_reads_test_targets(self):
        graph, _ = fixture()
        graph.y[graph.test_mask] = -999
        steps = []
        _, result = train_endpoint(
            graph, dict(in_channels=2, hidden_channels=4, out_channels=2, depth=2, dropout=0.),
            Config(epochs=2), 42, evaluate_test=False,
            epoch_callback=lambda epoch, loss, accuracy: steps.append((epoch, loss, accuracy)),
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual(set(result["metrics"]), {"train", "val"})
        self.assertEqual(set(result["metrics_before_calibration"]), {"train", "val"})

    def test_persistent_reuse_resume_and_identity(self):
        from gcn_mc.tuning import cache_context, tune_endpoints
        graph, metadata = fixture()
        with tempfile.TemporaryDirectory() as folder:
            config, _ = resolved_config(Config(
                preset="reference", tune_endpoints=True, tuning_trials=2, tuning_cache=folder,
                overrides={"hidden_channels": 4, "depth": 2, "epochs": 2, "dropout": 0.},
            ), "Cora")
            selected, first = tune_endpoints(graph, metadata, config)
            self.assertFalse(first["cache_hit"])
            self.assertEqual(first["new_trials"], 2)
            self.assertEqual(selected.hidden_channels, 4)
            self.assertEqual(selected.depth, 2)
            self.assertEqual(selected.dropout, 0.)
            self.assertTrue(Path(first["best_parameters_file"]).is_file())
            changed_test = graph.clone()
            changed_test.y[graph.test_mask] = -999
            second_config = replace(config, curve_lr=.123, curve_epochs=7, pairs=[(4, 5)], points=9)
            with patch("gcn_mc.experiment.train_endpoint", side_effect=AssertionError("Cached tuning must not train")):
                again, second = tune_endpoints(changed_test, metadata, second_config)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["new_trials"], 0)
            self.assertEqual(first["key"], second["key"])
            self.assertEqual(first["best_params"], second["best_params"])
            self.assertEqual(again.curve_lr, .123)
            _, resumed = tune_endpoints(graph, metadata, replace(config, tuning_trials=3))
            self.assertEqual(resumed["new_trials"], 1)
            self.assertEqual(resumed["finished_trials"], 3)
            original_context = cache_context(graph, metadata, config)
            self.assertNotEqual(original_context, cache_context(graph, metadata, replace(config, epochs=3)))
            changed_val = graph.clone()
            changed_val.y[graph.val_mask] = 1 - changed_val.y[graph.val_mask]
            self.assertNotEqual(original_context, cache_context(changed_val, metadata, config))

    def test_tuned_runner_records_parameters_without_changing_curve_settings(self):
        graph, metadata = fixture()
        with tempfile.TemporaryDirectory() as folder:
            config = Config(
                preset="reference", architecture="mlp", tune_endpoints=True, tuning_trials=2,
                tuning_cache=str(Path(folder) / "cache"), pairs=[(0, 1)], curve_epochs=2,
                points=3, curve_lr=.013,
                overrides={"hidden_channels": 4, "depth": 2, "epochs": 2, "dropout": 0.},
            )
            with patch("gcn_mc.experiment.load_graph", return_value=(graph, metadata)):
                report_path = run(config, Path(folder) / "run")
            dataset = json.loads(report_path.read_text())["datasets"][0]
            self.assertEqual(dataset["tuning"]["finished_trials"], 2)
            self.assertEqual(dataset["config"]["lr"], dataset["tuning"]["best_params"]["lr"])
            self.assertEqual(dataset["model"]["hidden_channels"], 4)
            self.assertEqual(dataset["config"]["curve_lr"], .013)
            self.assertEqual(len(dataset["pairs"][0]["history"]), 2)


if __name__ == "__main__":
    unittest.main()
