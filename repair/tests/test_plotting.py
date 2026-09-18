"""Tests for report plotting."""

import json
from pathlib import Path
import tempfile
import unittest


class PlottingTest(unittest.TestCase):
    def test_plot_report_writes_png_and_pdf_without_modifying_report(self):
        from repair.plotting import plot_report

        report = {
            "schema_version": 1,
            "dataset": "cifar10",
            "architecture": {"name": "vgg11"},
            "endpoints": [
                {"loss": 1.2, "accuracy": 0.70, "samples": 20},
                {"loss": 0.8, "accuracy": 0.80, "samples": 20},
            ],
            "curve": [
                {
                    "alpha": 0,
                    "unaligned": {"loss": 1.2, "accuracy": 0.70, "samples": 20},
                    "aligned": {"loss": 1.2, "accuracy": 0.70, "samples": 20},
                    "repaired": {"loss": 1.2, "accuracy": 0.70, "samples": 20},
                },
                {
                    "alpha": 0.5,
                    "unaligned": {"loss": 1.6, "accuracy": 0.60, "samples": 20},
                    "aligned": {"loss": 0.9, "accuracy": 0.75, "samples": 20},
                    "repaired": {"loss": 0.85, "accuracy": 0.76, "samples": 20},
                },
                {
                    "alpha": 1,
                    "unaligned": {"loss": 0.8, "accuracy": 0.80, "samples": 20},
                    "aligned": {"loss": 0.8, "accuracy": 0.80, "samples": 20},
                    "repaired": {"loss": 0.8, "accuracy": 0.80, "samples": 20},
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            output_dir = Path(temp) / "plots"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            before = report_path.read_bytes()

            outputs = plot_report(report_path, output_dir)

            self.assertEqual(outputs, [output_dir / "interpolation.png", output_dir / "interpolation.pdf"])
            for path, header in (
                (output_dir / "interpolation.png", b"\x89PNG\r\n\x1a\n"),
                (output_dir / "interpolation.pdf", b"%PDF"),
            ):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, len(header))
                self.assertTrue(path.read_bytes().startswith(header))
            self.assertEqual(report_path.read_bytes(), before)

    def test_plot_report_rejects_unknown_schema_version(self):
        from repair.plotting import plot_report

        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            report_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            with self.assertRaises(ValueError):
                plot_report(report_path)


if __name__ == "__main__":
    unittest.main()
