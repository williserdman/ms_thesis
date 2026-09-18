"""Exercise tuning, saved configuration, and development without Optuna."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SpectralTrainingTest(unittest.TestCase):
    def test_tuned_config_runs_without_importing_optuna(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "configs"
            output_dir = Path(directory) / "runs"
            args = [
                "--datasets", "Cora", "--epochs", "2", "--patience", "1",
                "--accelerator", "cpu", "--threads", "2",
                "--config-dir", str(config_dir), "--output-dir", str(output_dir),
            ]
            tuning = subprocess.run(
                [sys.executable, "src/train_spectral.py", *args, "--optuna", "--trials", "1"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(tuning.returncode, 0, tuning.stdout + tuning.stderr)
            config = json.loads((config_dir / "Cora.json").read_text())
            self.assertEqual(config["dataset"], "Cora")
            self.assertEqual(config["selection"]["n_trials"], 1)
            self.assertGreater(config["model"]["learning_rate"], 0)
            self.assertIn("test_accuracy", config["test_metrics"])
            original = (config_dir / "Cora.json").read_bytes()

            development = subprocess.run(
                [sys.executable, "-c",
                 "import runpy,sys; sys.modules['optuna']=None; "
                 "sys.path.insert(0,'src'); "
                 "sys.argv=['src/train_spectral.py']+sys.argv[1:]; "
                 "runpy.run_path('src/train_spectral.py',run_name='__main__')", *args],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(development.returncode, 0, development.stdout + development.stderr)
            self.assertIn("test_accuracy", development.stdout)
            self.assertEqual((config_dir / "Cora.json").read_bytes(), original)

    def test_single_band_pass_filter_uses_scoped_config_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "configs"
            output_dir = Path(directory) / "runs"
            args = [
                "--datasets", "Cora", "--filter", "band-pass",
                "--epochs", "2", "--patience", "1",
                "--accelerator", "cpu", "--threads", "2",
                "--config-dir", str(config_dir), "--output-dir", str(output_dir),
            ]
            tuning = subprocess.run(
                [sys.executable, "src/train_spectral.py", *args, "--optuna", "--trials", "1"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(tuning.returncode, 0, tuning.stdout + tuning.stderr)

            config_path = config_dir / "band-pass" / "Cora.json"
            config = json.loads(config_path.read_text())
            self.assertEqual(config["model"]["filters"], ["g_band_pass"])
            checkpoint = Path(config["selection"]["checkpoint"])
            self.assertTrue(checkpoint.is_file(), checkpoint)
            self.assertTrue(checkpoint.is_relative_to(output_dir / "band-pass" / "Cora"))
            original = config_path.read_bytes()

            development = subprocess.run(
                [sys.executable, "-c",
                 "import runpy,sys; sys.modules['optuna']=None; "
                 "sys.path.insert(0,'src'); "
                 "sys.argv=['src/train_spectral.py']+sys.argv[1:]; "
                 "runpy.run_path('src/train_spectral.py',run_name='__main__')", *args],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(development.returncode, 0, development.stdout + development.stderr)
            self.assertIn("test_accuracy", development.stdout)
            self.assertEqual(config_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
