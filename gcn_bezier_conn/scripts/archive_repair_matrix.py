"""Audit and archive the saved-endpoint alignment/REPAIR matrix."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("Cora", "Roman-empire", "squirrel", "chameleon")
MODELS = {"gcn": "GCN", "mlp": "MLP", "graphsage": "GraphSAGE", "gat": "GAT"}
METHODS = {"linear": ("Linear", "#b84a39"), "aligned": ("Aligned linear", "#ba801f"),
           "repaired": ("Aligned + REPAIR", "#248454"), "bezier": ("Bézier", "#285eaa")}


def archive(run_root, output):
    output.mkdir(parents=True, exist_ok=True)
    figures = {metric: plt.subplots(4, 4, figsize=(15, 12), sharex=True) for metric in ("loss", "accuracy")}
    summary = {"configurations": 0, "pairs": 0, "runs": {}, "max_source_loss_error": 0.,
               "max_source_accuracy_error": 0., "max_midpoint_replay_loss_error": 0.,
               "max_midpoint_replay_accuracy_error": 0., "float64_alignment_checks": 0, "alignment_prediction_checks": 0,
               "max_recorded_alignment_prediction_disagreements": 0,
               "max_recorded_source_correct_count_difference": 0}
    lines = ["# Alignment and REPAIR on tuned endpoints", "",
             "All four architectures and datasets, three saved endpoint pairs per configuration. "
             "Each pair compares the same two endpoints and its original Bézier curve. "
             "No endpoint training, Optuna search or curve fitting occurred in this analysis.", "",
             "![Test loss paths](test_loss_grid.png)", "",
             "![Test accuracy paths](test_accuracy_grid.png)", "",
             "Mean sampled test cross-entropy barriers across three pairs:", "",
             "| Dataset | Model | Linear | Aligned | Aligned + REPAIR | Bézier |",
             "|---|---|---:|---:|---:|---:|"]
    for row, name in enumerate(DATASETS):
        summary["runs"][name] = {}
        for column, (architecture, display) in enumerate(MODELS.items()):
            report_path = run_root / name / architecture / "report.json"
            report = json.loads(report_path.read_text())
            source_path = Path(report["source_report"])
            source_bytes = source_path.read_bytes()
            source = json.loads(source_bytes)["datasets"][0]
            assert hashlib.sha256(source_bytes).hexdigest() == report["source_report_sha256"]
            assert report["experiment"] == "gnn_repair" and report["methods"] == list(METHODS)
            data = report["datasets"][0]
            assert data["model"] == source["model"] and data["model"]["architecture"] == architecture
            assert data["data"] == source["data"]
            assert len(data["endpoints"]) == 6 and len(data["pairs"]) == 3
            assert data["tuning"]["key"] == source["tuning"]["key"]
            for pair, original in zip(data["pairs"], source["pairs"]):
                assert [pair["seed_a"], pair["seed_b"]] == [original["seed_a"], original["seed_b"]]
                assert pair["alignment"]["permutations"]
                if "prediction_disagreements" in pair["alignment"]:
                    summary["alignment_prediction_checks"] += 1
                if pair["alignment"].get("verification_dtype") == "torch.float64":
                    summary["float64_alignment_checks"] += 1
                summary["max_recorded_alignment_prediction_disagreements"] = max(
                    summary["max_recorded_alignment_prediction_disagreements"],
                    pair["alignment"].get("prediction_disagreements", 0))
                assert len(pair["calibration"]) == 21
                for method in METHODS:
                    assert pair[method]["t"] == original["linear"]["t"]
                    for split in ("train", "val", "test"):
                        for metric in ("loss", "accuracy"):
                            values = np.asarray(pair[method]["splits"][split][metric])
                            assert len(values) == 21 and np.isfinite(values).all()
                            endpoints = {endpoint["seed"]: endpoint for endpoint in data["endpoints"]}
                            for index, seed in ((0, pair["seed_a"]), (-1, pair["seed_b"])):
                                endpoint_error = abs(values[index] - endpoints[seed]["metrics"][split][metric])
                                assert endpoint_error <= (2e-5 if metric == "loss" else 1e-6), (name, architecture, method, split, metric, endpoint_error)
                            if method in ("linear", "bezier"):
                                error = float(np.max(np.abs(values - original[method]["splits"][split][metric])))
                                summary[f"max_source_{metric}_error"] = max(summary[f"max_source_{metric}_error"], error)
                                assert error <= (2e-5 if metric == "loss" else 1e-6), (name, architecture, method, split, metric, error)
                for method, splits in pair.get("source_replay", {}).items():
                    for split, replay in splits.items():
                        assert replay["max_loss_error"] <= 2e-5
                        assert replay["max_correct_count_difference"] <= 1
                        summary["max_recorded_source_correct_count_difference"] = max(
                            summary["max_recorded_source_correct_count_difference"], replay["max_correct_count_difference"])
                for metric in ("loss", "accuracy"):
                    error = pair[f"midpoint_replay_max_{metric}_error"]
                    summary[f"max_midpoint_replay_{metric}_error"] = max(summary[f"max_midpoint_replay_{metric}_error"], error)
                    assert error <= (2e-5 if metric == "loss" else 1e-6)
                summary["pairs"] += 1
            barriers = {method: data["summary"][method]["test"]["barrier_mean"] for method in METHODS}
            summary["runs"][name][architecture] = {
                "model": data["model"], "tuning_cache_key": data["tuning"]["key"],
                "source_report_sha256": report["source_report_sha256"],
                "test_barrier_means": barriers, "metrics": data["summary"],
                "max_alignment_logit_error": max(p["alignment"]["max_logit_error"] for p in data["pairs"]),
            }
            lines.append(f"| {name} | {display} | " + " | ".join(f"{barriers[m]:.4f}" for m in METHODS) + " |")
            destination = output / name / architecture
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_path, destination / "report.json")
            for suffix in ("png", "pdf"):
                shutil.copy2(report_path.parent / name / f"connectivity.{suffix}", destination / f"connectivity.{suffix}")
            for metric, (_, axes) in figures.items():
                ax = axes[row, column]
                for method, (label, color) in METHODS.items():
                    ts = data["pairs"][0][method]["t"]
                    curves = np.asarray([p[method]["splits"]["test"][metric] for p in data["pairs"]])
                    mean, std = curves.mean(0), curves.std(0)
                    ax.plot(ts, mean, color=color, label=label, linewidth=1.6)
                    ax.fill_between(ts, mean - std, mean + std, color=color, alpha=.12)
                ax.set_title(f"{name} / {display}", fontsize=11)
                ax.set_xlim(0, 1)
                ax.grid(alpha=.2)
                if metric == "accuracy":
                    ax.set_ylim(0, 1)
                if column == 0:
                    ax.set_ylabel("Test cross entropy" if metric == "loss" else "Test accuracy")
                if row == 3:
                    ax.set_xlabel("Path position t")
            summary["configurations"] += 1
    for metric, (figure, axes) in figures.items():
        figure.suptitle("Saved Optuna endpoints: alignment and REPAIR on linear paths\nThree pairs per panel; shading is population standard deviation", fontsize=15)
        figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
        figure.tight_layout(rect=(0, .035, 1, .95))
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"test_{metric}_grid.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)
    lines.extend(["", "Per-model folders contain labeled train/validation/test plots and full reports. "
                  "The original models, tuning provenance and source report hashes identify the reused endpoints.", "",
                  "Alignment uses training-node correlations and checks endpoint logit invariance. "
                  "REPAIR uses training-node moments after complete blocks, with inherited full-graph BatchNorm "
                  "calibration frozen before correction. It matches weighted endpoint means and standard deviations "
                  "sequentially; Bézier remains unchanged. The repaired path is not a straight parameter line.", "",
                  "All 48 repaired midpoint checkpoints were reloaded on the GPU that ran their analysis. "
                  "The audit verifies source curves, finite metrics, pair/grid identity and midpoint replay. "
                  "Weights remain in the ignored local run directory; reports and graphics are tracked."])
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    archive(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
