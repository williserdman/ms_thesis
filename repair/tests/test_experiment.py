"""The demo must exercise training through saved, reloadable repaired predictions."""

import json
from pathlib import Path
import tempfile
import unittest

import torch


class ExperimentTest(unittest.TestCase):
    def test_cpu_demo_writes_metrics_and_reloadable_checkpoint(self):
        from repair.experiment import main
        from repair.models import tiny_mlp

        with tempfile.TemporaryDirectory() as temp:
            main([
                "demo", "--epochs", "2", "--samples", "64", "--batch-size", "32",
                "--alphas", "0", "0.5", "1", "--device", "cpu", "--output", temp,
            ])
            report = json.loads((Path(temp) / "report.json").read_text())
            self.assertEqual(len(report["curve"]), 3)
            self.assertEqual(report["dataset"], "synthetic")
            for row in report["curve"]:
                for variant in ["unaligned", "aligned", "repaired"]:
                    self.assertTrue(0 <= row[variant]["accuracy"] <= 1)
                    self.assertGreaterEqual(row[variant]["loss"], 0)
            model = tiny_mlp()
            model.load_state_dict(torch.load(Path(temp) / "repaired_midpoint.pt", weights_only=True))
            self.assertEqual(tuple(model(torch.zeros(3, 16)).shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
