"""Generate PNG and PDF figures from a saved report without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_report(report_path):
    report_path = Path(report_path).expanduser().resolve()
    report = json.loads(report_path.read_text())
    paths = []
    styles = {
        "linear": ("Linear", "#b84a39"),
        "aligned": ("Aligned linear", "#ba801f"),
        "repaired": ("Aligned + REPAIR", "#248454"),
        "bezier": ("Bézier", "#285eaa"),
    }
    for dataset in report["datasets"]:
        figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
        for column, split in enumerate(("train", "val", "test")):
            for row, metric in enumerate(("loss", "accuracy")):
                ax = axes[row, column]
                for method in report.get("methods", ["linear", "bezier"]):
                    label, color = styles[method]
                    ts = dataset["pairs"][0][method]["t"]
                    curves = np.asarray([pair[method]["splits"][split][metric] for pair in dataset["pairs"]])
                    mean, std = curves.mean(axis=0), curves.std(axis=0)
                    ax.plot(ts, mean, label=label, color=color, linewidth=2)
                    if len(curves) > 1:
                        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)
                ax.grid(alpha=0.2)
                ax.set_xlim(0, 1)
                if row == 0:
                    ax.set_title(split.capitalize())
                else:
                    ax.set_xlabel("Path position t")
                    ax.set_ylim(0, 1)
                if column == 0:
                    ax.set_ylabel("Cross entropy" if metric == "loss" else "Accuracy")
        note = "smoke test" if report["config"]["smoke"] else "linear/Bézier baseline"
        if dataset.get("tuning") and not report["config"]["smoke"]:
            note = "Optuna endpoints; fixed Bézier settings"
        if report.get("experiment") == "gnn_repair":
            note = "Alignment + REPAIR comparison" + (", smoke endpoints" if report["config"]["smoke"] else "")
        n_pairs = len(dataset["pairs"])
        model = dataset.get("model", report["config"])
        model_name = {"gcn": "GCN", "mlp": "MLP", "graphsage": "GraphSAGE", "gat": "GAT"}[model.get("architecture", "gcn")]
        architecture = (
            f"{model_name}, depth {model['depth']}, hidden width {model['hidden_channels']}, "
            f"dropout {model['dropout']:.3g}"
        )
        if model.get("family") == "tunedgnn":
            architecture += f", norm {model['normalization']}, residual {model['residual']}"
        figure.suptitle(
            f"{dataset['data']['name']}: {architecture}\n{note}, {n_pairs} endpoint pair(s)"
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
        figure.tight_layout(rect=(0, 0.05, 1, 0.94))
        directory = report_path.parent / dataset["data"]["name"]
        directory.mkdir(exist_ok=True)
        for extension in ("png", "pdf"):
            path = directory / f"connectivity.{extension}"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            paths.append(path)
        plt.close(figure)
    return paths
