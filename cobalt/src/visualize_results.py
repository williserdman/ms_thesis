import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


PAPER_BASELINES = {
    "Cora": {
        "GCN": 75.01,
        "GAT": 77.22,
        "APPNP": 79.93,
        "ChebNet": 69.58,
        "JKNet": 71.31,
        "GPR-GNN": 79.65,
        "BernNet": 73.39,
        "JacobiConv": 80.02,
        "Arnoldi-GCN": 80.25,
        "G-Arnoldi-GCN": 82.33,
    },
    "Citeseer": {
        "GCN": 67.57,
        "GAT": 66.42,
        "APPNP": 68.27,
        "ChebNet": 65.36,
        "JKNet": 61.36,
        "GPR-GNN": 66.92,
        "BernNet": 65.84,
        "JacobiConv": 68.23,
        "Arnoldi-GCN": 67.81,
        "G-Arnoldi-GCN": 69.88,
    },
    "Pubmed": {
        "GCN": 84.17,
        "GAT": 83.32,
        "APPNP": 84.22,
        "ChebNet": 83.88,
        "JKNet": 82.92,
        "GPR-GNN": 84.21,
        "BernNet": 84.20,
        "JacobiConv": 84.32,
        "Arnoldi-GCN": 84.02,
        "G-Arnoldi-GCN": 85.23,
    },
    "Photo": {
        "GCN": 81.81,
        "GAT": 86.66,
        "APPNP": 83.24,
        "ChebNet": 88.00,
        "JKNet": 78.25,
        "GPR-GNN": 88.55,
        "BernNet": 86.33,
        "JacobiConv": 86.41,
        "Arnoldi-GCN": 88.25,
        "G-Arnoldi-GCN": 92.46,
    },
    "Computers": {
        "GCN": 68.58,
        "GAT": 72.38,
        "APPNP": 67.46,
        "ChebNet": 79.25,
        "JKNet": 66.43,
        "GPR-GNN": 80.73,
        "BernNet": 79.25,
        "JacobiConv": 81.54,
        "Arnoldi-GCN": 78.81,
        "G-Arnoldi-GCN": 83.81,
    },
    "Texas": {
        "GCN": 32.13,
        "GAT": 34.27,
        "APPNP": 34.67,
        "ChebNet": 32.13,
        "JKNet": 30.75,
        "GPR-GNN": 33.56,
        "BernNet": 40.69,
        "JacobiConv": 41.23,
        "Arnoldi-GCN": 63.20,
        "G-Arnoldi-GCN": 66.20,
    },
    "Cornell": {
        "GCN": 22.08,
        "GAT": 24.39,
        "APPNP": 34.98,
        "ChebNet": 27.57,
        "JKNet": 25.20,
        "GPR-GNN": 38.84,
        "BernNet": 39.32,
        "JacobiConv": 39.23,
        "Arnoldi-GCN": 51.24,
        "G-Arnoldi-GCN": 55.87,
    },
    "Actor": {
        "GCN": 22.45,
        "GAT": 24.31,
        "APPNP": 28.41,
        "ChebNet": 22.00,
        "JKNet": 21.02,
        "GPR-GNN": 27.70,
        "BernNet": 28.85,
        "JacobiConv": 26.37,
        "Arnoldi-GCN": 26.63,
        "G-Arnoldi-GCN": 27.73,
    },
    "Chameleon": {
        "GCN": 39.89,
        "GAT": 37.86,
        "APPNP": 29.38,
        "ChebNet": 36.41,
        "JKNet": 32.66,
        "GPR-GNN": 33.23,
        "BernNet": 34.73,
        "JacobiConv": 41.12,
        "Arnoldi-GCN": 40.25,
        "G-Arnoldi-GCN": 43.35,
    },
    "Squirrel": {
        "GCN": 29.66,
        "GAT": 24.56,
        "APPNP": 21.11,
        "ChebNet": 26.43,
        "JKNet": 24.20,
        "GPR-GNN": 23.43,
        "BernNet": 22.38,
        "JacobiConv": 32.23,
        "Arnoldi-GCN": 24.12,
        "G-Arnoldi-GCN": 26.64,
    },
    "Roman-empire": {
        "GCN": 29.02,
        "GAT": 37.26,
        "APPNP": 35.30,
        "ChebNet": 35.97,
        "JKNet": 35.97,
        "GPR-GNN": 36.13,
        "BernNet": 39.63,
        "JacobiConv": 41.02,
        "Arnoldi-GCN": 53.04,
        "G-Arnoldi-GCN": 69.63,
    },
    "Amazon-ratings": {
        "GCN": 28.97,
        "GAT": 29.97,
        "APPNP": 29.88,
        "ChebNet": 28.81,
        "JKNet": 28.81,
        "GPR-GNN": 30.03,
        "BernNet": 29.32,
        "JacobiConv": 30.24,
        "Arnoldi-GCN": 43.71,
        "G-Arnoldi-GCN": 48.68,
    },
    "Minesweeper": {
        "GCN": 72.60,
        "GAT": 73.51,
        "APPNP": 67.75,
        "ChebNet": 73.18,
        "JKNet": 73.42,
        "GPR-GNN": 76.87,
        "BernNet": 75.49,
        "JacobiConv": 74.13,
        "Arnoldi-GCN": 69.73,
        "G-Arnoldi-GCN": 88.52,
    },
    "Tolokers": {
        "GCN": 73.11,
        "GAT": 71.11,
        "APPNP": 68.99,
        "ChebNet": 72.48,
        "JKNet": 70.53,
        "GPR-GNN": 68.64,
        "BernNet": 69.08,
        "JacobiConv": 65.15,
        "Arnoldi-GCN": 70.25,
        "G-Arnoldi-GCN": 73.01,
    },
    "Questions": {
        "GCN": 59.86,
        "GAT": 64.39,
        "APPNP": 46.80,
        "ChebNet": 64.51,
        "JKNet": 56.55,
        "GPR-GNN": 54.13,
        "BernNet": 56.27,
        "JacobiConv": 56.21,
        "Arnoldi-GCN": 56.12,
        "G-Arnoldi-GCN": 66.18,
    },
}

DATASET_ALIASES = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "Pubmed",
    "photo": "Photo",
    "computers": "Computers",
    "texas": "Texas",
    "cornell": "Cornell",
    "actor": "Actor",
    "chameleon": "Chameleon",
    "squirrel": "Squirrel",
    "roman-empire": "roman-empire",
    "amazon-ratings": "amazon-ratings",
    "minesweeper": "Minesweeper",
    "tolokers": "Tolokers",
    "questions": "Questions",
}


def canonical_dataset_name(name):
    key = str(name).strip()
    return DATASET_ALIASES.get(key.lower(), key)


def _extract_metric_list(test_field):
    if isinstance(test_field, dict):
        metrics = test_field.get("test_metrics", [])
        if isinstance(metrics, dict):
            return [metrics]
        if isinstance(metrics, list):
            return metrics
        return []
    if isinstance(test_field, list):
        return test_field
    if isinstance(test_field, dict):
        return [test_field]
    return []


def _extract_test_accuracy(entry):
    metrics_list = _extract_metric_list(entry.get("test"))
    if not metrics_list:
        return None
    m0 = metrics_list[0]
    if not isinstance(m0, dict):
        return None
    for key in ("test_accuracy", "test_acc", "accuracy"):
        if key in m0:
            try:
                return float(m0[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_test_loss(entry):
    metrics_list = _extract_metric_list(entry.get("test"))
    if not metrics_list:
        return None
    m0 = metrics_list[0]
    if not isinstance(m0, dict):
        return None
    for key in ("test_loss", "loss"):
        if key in m0:
            try:
                return float(m0[key])
            except (TypeError, ValueError):
                return None
    return None


def load_entries(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    entries = []
    for path in files:
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
        except Exception:
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item = dict(item)
                    item["_source_file"] = path
                    entries.append(item)
        elif isinstance(data, dict):
            payload = dict(data)
            payload["_source_file"] = path
            entries.append(payload)

    return entries, files


def build_frame(entries):
    rows = []
    for e in entries:
        dataset = canonical_dataset_name(e.get("dataset", "unknown"))
        pipeline = e.get("pipeline", "single_stage")
        acc = _extract_test_accuracy(e)
        test_loss = _extract_test_loss(e)

        if "best_stage2_val_loss" in e:
            val_loss = e.get("best_stage2_val_loss")
        else:
            val_loss = e.get("best_val_loss")

        rows.append(
            {
                    "dataset": dataset,
                "pipeline": pipeline,
                "test_accuracy": acc,
                "test_loss": test_loss,
                "best_val_loss": val_loss,
                "filter_profile": (
                    (e.get("best_stage2_params") or {}).get("s2_filter_profile")
                    if isinstance(e.get("best_stage2_params"), dict)
                    else None
                ),
                "source_file": e.get("_source_file", "unknown"),
            }
        )

    return pd.DataFrame(rows)


def plot_performance(df, out_dir):
    if df.empty:
        return

    perf = df.dropna(subset=["test_accuracy"]).copy()
    if perf.empty:
        return

    perf = perf.sort_values(["dataset", "pipeline"])

    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = perf["dataset"].tolist()
    x = np.arange(len(x_labels))
    colors = ["#1f77b4" if p == "two_stage" else "#ff7f0e" for p in perf["pipeline"]]

    ax.bar(x, perf["test_accuracy"].values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy by Dataset")

    handles = [
        Rectangle((0, 0), 1, 1, color="#1f77b4", label="two_stage"),
        Rectangle((0, 0), 1, 1, color="#ff7f0e", label="single_stage"),
    ]
    ax.legend(handles=handles)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "performance_test_accuracy.png"), dpi=200)
    plt.close(fig)


def plot_val_loss(df, out_dir):
    view = df.dropna(subset=["best_val_loss"]).copy()
    if view.empty:
        return

    view = view.sort_values("best_val_loss", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{d} ({p})" for d, p in zip(view["dataset"], view["pipeline"])]
    x = np.arange(len(labels))
    ax.bar(x, view["best_val_loss"].astype(float).values, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Best Validation Loss by Dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "performance_best_val_loss.png"), dpi=200)
    plt.close(fig)


def plot_filter_coefficients(entries, out_dir):
    produced = 0
    for e in entries:
        dataset = canonical_dataset_name(e.get("dataset", "unknown"))
        test_blob = e.get("test", {})
        if not isinstance(test_blob, dict):
            continue

        coeffs = test_blob.get("filter_coefficients")
        if not isinstance(coeffs, dict) or not coeffs:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, vals in coeffs.items():
            if not isinstance(vals, list) or len(vals) == 0:
                continue
            xs = np.arange(len(vals))
            ys = [float(v) for v in vals]
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=name)

        ax.set_xlabel("Coefficient Index")
        ax.set_ylabel("Coefficient Value")
        ax.set_title(f"Learned Filter Coefficients ({dataset})")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out_name = f"filters_{dataset.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(os.path.join(out_dir, out_name), dpi=220)
        plt.close(fig)
        produced += 1

    return produced


def plot_cluster_stats(entries, out_dir):
    produced = 0
    for e in entries:
        dataset = canonical_dataset_name(e.get("dataset", "unknown"))
        test_blob = e.get("test", {})
        if not isinstance(test_blob, dict):
            continue

        stats = test_blob.get("cluster_stats")
        if not isinstance(stats, dict):
            continue

        hard_counts = stats.get("hard_counts")
        hard_fracs = stats.get("hard_fractions")
        if not isinstance(hard_counts, list) or not hard_counts:
            continue

        k = len(hard_counts)
        x = np.arange(k)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].bar(x, [int(v) for v in hard_counts], color="#9467bd")
        axes[0].set_title(f"Cluster Counts ({dataset})")
        axes[0].set_xlabel("Cluster Index")
        axes[0].set_ylabel("Assigned Nodes")

        if isinstance(hard_fracs, list) and len(hard_fracs) == k:
            axes[1].bar(x, [float(v) for v in hard_fracs], color="#8c564b")
            axes[1].set_ylabel("Fraction")
        else:
            total = max(sum(hard_counts), 1)
            frac = [float(v) / total for v in hard_counts]
            axes[1].bar(x, frac, color="#8c564b")
            axes[1].set_ylabel("Fraction")

        entropy = stats.get("mean_assignment_entropy")
        norm_entropy = stats.get("normalized_assignment_entropy")
        subtitle = f"entropy={entropy:.4f}, normalized={norm_entropy:.4f}" if isinstance(entropy, (int, float)) and isinstance(norm_entropy, (int, float)) else ""
        axes[1].set_title(f"Cluster Fractions ({dataset})\n{subtitle}")
        axes[1].set_xlabel("Cluster Index")

        fig.tight_layout()
        out_name = f"clusters_{dataset.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(os.path.join(out_dir, out_name), dpi=220)
        plt.close(fig)
        produced += 1

    return produced


def plot_paper_baseline_comparison(df, out_dir):
    if df.empty:
        return 0

    produced = 0
    for dataset in df["dataset"].dropna().unique():
        if dataset not in PAPER_BASELINES:
            continue

        ours = df[(df["dataset"] == dataset) & df["test_accuracy"].notna()].copy()
        if ours.empty:
            continue

        paper_methods = PAPER_BASELINES[dataset]
        comparison = []
        for method, score in paper_methods.items():
            comparison.append((method, score, "paper"))

        for _, row in ours.iterrows():
            label = row["pipeline"]
            if row.get("filter_profile"):
                label = f"{label}:{row['filter_profile']}"
            comparison.append((label, float(row["test_accuracy"]) * 100.0, "ours"))

        comparison = sorted(comparison, key=lambda item: item[1], reverse=True)

        labels = [item[0] for item in comparison]
        values = [item[1] for item in comparison]
        colors = ["#1f77b4" if item[2] == "paper" else "#d62728" for item in comparison]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"Paper Baselines vs. Ours on {dataset}")

        handles = [
            Rectangle((0, 0), 1, 1, color="#1f77b4", label="paper baselines"),
            Rectangle((0, 0), 1, 1, color="#d62728", label="ours"),
        ]
        ax.legend(handles=handles)
        fig.tight_layout()
        out_name = f"paper_comparison_{dataset.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(os.path.join(out_dir, out_name), dpi=220)
        plt.close(fig)
        produced += 1

    return produced


def save_summary_csv(df, out_dir):
    if df.empty:
        return
    out_csv = os.path.join(out_dir, "results_summary.csv")
    df.to_csv(out_csv, index=False)


def save_baseline_comparison_csv(df, out_dir):
    rows = []
    for dataset, methods in PAPER_BASELINES.items():
        paper_best = max(methods.items(), key=lambda item: item[1])
        ours = df[(df["dataset"] == dataset) & df["test_accuracy"].notna()]
        ours_best = None
        if not ours.empty:
            best_row = ours.sort_values("test_accuracy", ascending=False).iloc[0]
            ours_best = {
                "label": best_row.get("pipeline", "ours"),
                "score": float(best_row["test_accuracy"]) * 100.0,
            }

        rows.append(
            {
                "dataset": dataset,
                "paper_best_method": paper_best[0],
                "paper_best_score": paper_best[1],
                "ours_best_method": None if ours_best is None else ours_best["label"],
                "ours_best_score": None if ours_best is None else ours_best["score"],
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "paper_vs_ours_summary.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize experiment performance, learned filters, and cluster statistics."
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["smoke_results_*.json", "training_results_*.json"],
        help="Glob patterns for result JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default="figures",
        help="Directory to write plots and CSV summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    entries, files = load_entries(args.patterns)
    if not entries:
        print("No result entries found. Check --patterns.")
        return

    print(f"Loaded {len(entries)} entries from {len(files)} files")

    df = build_frame(entries)
    save_summary_csv(df, args.out_dir)
    save_baseline_comparison_csv(df, args.out_dir)

    plot_performance(df, args.out_dir)
    plot_val_loss(df, args.out_dir)
    n_filter = plot_filter_coefficients(entries, args.out_dir)
    n_cluster = plot_cluster_stats(entries, args.out_dir)
    n_compare = plot_paper_baseline_comparison(df, args.out_dir)

    print(f"Wrote summary CSV: {os.path.join(args.out_dir, 'results_summary.csv')}")
    print(f"Generated performance plots in: {args.out_dir}")
    print(f"Generated filter plots: {n_filter}")
    print(f"Generated cluster-stat plots: {n_cluster}")
    print(f"Generated paper comparison plots: {n_compare}")


if __name__ == "__main__":
    main()
