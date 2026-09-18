"""Focused behavior checks for the reference architecture workflow."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from gcn_mc.experiment import Config, metrics, run
from gcn_mc.paths import clone_parameters, path_logits
from gcn_mc.repair_experiment import run_repair


class RepairScopeTests(unittest.TestCase):
    def test_reference_models_are_rejected_before_output_or_checkpoint_io(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            source.write_text(json.dumps({
                "schema_version": 1,
                "datasets": [{"model": {
                    "family": "tunedgnn", "architecture": "gat",
                }}],
            }))
            output = root / "repair"
            try:
                run_repair(source, output)
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertIn("legacy GCN", str(error))
            else:
                self.fail("Reference GAT must not enter the GCN REPAIR runner")
            self.assertFalse(output.exists())


class ReferenceRunnerTests(unittest.TestCase):
    def test_saved_endpoints_and_paths_replay_for_all_reference_architectures(self):
        from gcn_mc.model import build_model

        graph = Data(
            x=torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., .5], [.2, -.3], [-.5, 1.]]),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]),
            y=torch.tensor([0, 1, 0, 1, 0, 1]),
            train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
            val_mask=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
            test_mask=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
        )
        metadata = {"name": "Cora", "num_features": 2, "num_classes": 2, "split_sha256": "fixture"}
        for architecture in ("gcn", "mlp", "graphsage", "gat"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as folder:
                config = Config(
                    preset="reference", architecture=architecture, pairs=[(0, 1)],
                    curve_epochs=2, points=3,
                    overrides={"hidden_channels": 4, "depth": 2, "epochs": 2,
                               "normalization": "batch", "residual": True, "pre_linear": True},
                )
                with patch("gcn_mc.experiment.load_graph", return_value=(graph, metadata)):
                    report_path = run(config, folder)
                report = json.loads(report_path.read_text())
                dataset = report["datasets"][0]
                self.assertEqual(dataset["model"]["architecture"], architecture)
                self.assertEqual(dataset["config"]["selection"], "val_accuracy")
                self.assertEqual(dataset["config"]["epochs"], 2)
                self.assertEqual(dataset["model"]["hidden_channels"], 4)
                self.assertIn("source", dataset)
                endpoints = []
                for index, endpoint in enumerate(dataset["endpoints"]):
                    checkpoint = torch.load(Path(folder) / endpoint["checkpoint"], weights_only=True)
                    model = build_model(checkpoint["model_config"]).eval()
                    model.load_state_dict(checkpoint["state_dict"])
                    endpoints.append(clone_parameters(model))
                    expected_epoch = max(endpoint["history"], key=lambda step: step["val_accuracy"])["epoch"]
                    self.assertEqual(endpoint["selected_epoch"], expected_epoch)
                    with torch.no_grad():
                        replay = metrics(model(graph.x, graph.edge_index), graph)
                    for split in ("train", "val", "test"):
                        for metric in ("loss", "accuracy"):
                            self.assertAlmostEqual(replay[split][metric], endpoint["metrics"][split][metric], places=6)
                            for method in ("linear", "bezier"):
                                value = dataset["pairs"][0][method]["splits"][split][metric][0 if index == 0 else -1]
                                self.assertAlmostEqual(replay[split][metric], value, places=6)
                curve = torch.load(Path(folder) / dataset["pairs"][0]["checkpoint"], weights_only=True)
                for method, control in (("linear", None), ("bezier", curve["control"])):
                    with torch.no_grad():
                        replay = metrics(path_logits(model, graph, *endpoints, 0.5, control), graph)
                    for split in ("train", "val", "test"):
                        for metric in ("loss", "accuracy"):
                            recorded = dataset["pairs"][0][method]["splits"][split][metric][1]
                            self.assertAlmostEqual(replay[split][metric], recorded, places=6)


if __name__ == "__main__":
    unittest.main()
