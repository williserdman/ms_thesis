"""Regenerate the four-model Cora overview from the adjacent saved reports."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    names = {"gcn": "GCN", "mlp": "MLP", "graphsage": "GraphSAGE", "gat": "GAT"}
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey="row")
    for column, (architecture, title) in enumerate(names.items()):
        report = json.loads((root / "Cora" / architecture / "report.json").read_text())
        dataset = report["datasets"][0]
        model = dataset["model"]
        pair = dataset["pairs"][0]
        axes[0, column].set_title(
            f"{title}: depth {model['depth']}, width {model['hidden_channels']}\n"
            f"dropout {model['dropout']}, residual {model['residual']}", fontsize=10)
        for row, metric in enumerate(("loss", "accuracy")):
            ax = axes[row, column]
            for method, label, color in (("linear", "Linear", "#b84a39"), ("bezier", "Bézier", "#285eaa")):
                path = pair[method]
                ax.plot(path["t"], path["splits"]["test"][metric], label=label, color=color, linewidth=2)
            ax.set_xlim(0, 1)
            ax.grid(alpha=.2)
            if row == 1:
                ax.set_ylim(0, 1)
                ax.set_xlabel("Path position t")
            if column == 0:
                ax.set_ylabel("Test cross entropy" if row == 0 else "Test accuracy")
    figure.suptitle("Cora reference architecture pilots — one seed pair (0:1)\n"
                    "500 endpoint epochs · 200 Bézier steps · thesis public masks", fontsize=12)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, .06, 1, .9))
    for extension in ("png", "pdf"):
        figure.savefig(root / f"cora_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
