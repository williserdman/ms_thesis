#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _group_by_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        m = str(r.get("synthetic_type", "unknown"))
        out.setdefault(m, []).append(r)
    return out


def _bar_plot_means(rows: list[dict[str, Any]], dataset: str, out_dir: Path) -> None:
    grouped = _group_by_model(rows)
    models = sorted(grouped.keys())

    linear_means = []
    bezier_means = []
    for m in models:
        linear_vals = [_to_float(r.get("linear_barrier")) for r in grouped[m]]
        bezier_vals = [_to_float(r.get("bezier_barrier")) for r in grouped[m]]
        linear_means.append(sum(linear_vals) / max(1, len(linear_vals)))
        bezier_means.append(sum(bezier_vals) / max(1, len(bezier_vals)))

    x = list(range(len(models)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], linear_means, width=width, label="Linear Barrier")
    ax.bar([i + width / 2 for i in x], bezier_means, width=width, label="Bezier Barrier")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Barrier")
    ax.set_title(f"Barrier Means by Synthetic Model ({dataset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "barrier_means_by_model.png", dpi=180)
    plt.close(fig)


def _scatter_homophily(rows: list[dict[str, Any]], dataset: str, out_dir: Path) -> None:
    grouped = _group_by_model(rows)
    fig, ax = plt.subplots(figsize=(10, 5))

    for model, model_rows in grouped.items():
        x = [_to_float(r.get("edge_homophily")) for r in model_rows]
        y = [_to_float(r.get("linear_barrier")) for r in model_rows]
        ax.scatter(x, y, alpha=0.7, label=model)

    ax.set_xlabel("Realized Edge Homophily")
    ax.set_ylabel("Linear Barrier")
    ax.set_title(f"Linear Barrier vs Homophily ({dataset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "linear_barrier_vs_homophily.png", dpi=180)
    plt.close(fig)


def _scatter_density(rows: list[dict[str, Any]], dataset: str, out_dir: Path) -> None:
    grouped = _group_by_model(rows)
    fig, ax = plt.subplots(figsize=(10, 5))

    for model, model_rows in grouped.items():
        x = [_to_float(r.get("edge_density")) for r in model_rows]
        y = [_to_float(r.get("linear_barrier")) for r in model_rows]
        ax.scatter(x, y, alpha=0.7, label=model)

    ax.set_xlabel("Realized Edge Density")
    ax.set_ylabel("Linear Barrier")
    ax.set_title(f"Linear Barrier vs Density ({dataset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "linear_barrier_vs_density.png", dpi=180)
    plt.close(fig)


def _scatter_degree_std(rows: list[dict[str, Any]], dataset: str, out_dir: Path) -> None:
    grouped = _group_by_model(rows)
    fig, ax = plt.subplots(figsize=(10, 5))

    for model, model_rows in grouped.items():
        x = [_to_float(r.get("degree_std")) for r in model_rows]
        y = [_to_float(r.get("linear_barrier")) for r in model_rows]
        ax.scatter(x, y, alpha=0.7, label=model)

    ax.set_xlabel("Degree Std")
    ax.set_ylabel("Linear Barrier")
    ax.set_title(f"Linear Barrier vs Degree Std ({dataset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "linear_barrier_vs_degree_std.png", dpi=180)
    plt.close(fig)


def _write_topk(rows: list[dict[str, Any]], out_dir: Path, k: int = 10) -> None:
    sorted_rows = sorted(rows, key=lambda r: _to_float(r.get("linear_barrier")), reverse=True)
    top_rows = sorted_rows[:k]
    if not top_rows:
        return

    out_path = out_dir / "top_linear_barriers.csv"
    keys = list(top_rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(top_rows)


def _collect_dataset_rows(sweep_dir: Path, dataset: str) -> list[dict[str, Any]]:
    ds_dir = sweep_dir / dataset
    if not ds_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for csv_path in sorted(ds_dir.glob("barrier_results_*.csv")):
        rows.extend(_read_csv_rows(csv_path))
    return rows


def _collect_all_rows(sweep_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for r in _collect_dataset_rows(sweep_dir=sweep_dir, dataset=dataset):
            row = dict(r)
            row["dataset"] = dataset
            all_rows.append(row)
    return all_rows


def _bar_plot_means_all(rows: list[dict[str, Any]], out_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        dataset = str(r.get("dataset", "unknown"))
        model = str(r.get("synthetic_type", "unknown"))
        grouped.setdefault((dataset, model), []).append(r)

    keys = sorted(grouped.keys())
    labels = [f"{d}\n{m}" for d, m in keys]

    linear_means = []
    bezier_means = []
    for key in keys:
        rs = grouped[key]
        linear_vals = [_to_float(r.get("linear_barrier")) for r in rs]
        bezier_vals = [_to_float(r.get("bezier_barrier")) for r in rs]
        linear_means.append(sum(linear_vals) / max(1, len(linear_vals)))
        bezier_means.append(sum(bezier_vals) / max(1, len(bezier_vals)))

    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.1), 6))
    ax.bar([i - width / 2 for i in x], linear_means, width=width, label="Linear Barrier")
    ax.bar([i + width / 2 for i in x], bezier_means, width=width, label="Bezier Barrier")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Barrier")
    ax.set_title("Barrier Means by Dataset and Synthetic Model (All Datasets)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "barrier_means_all_datasets.png", dpi=180)
    plt.close(fig)


def _scatter_all(
    rows: list[dict[str, Any]],
    x_key: str,
    x_label: str,
    title: str,
    out_name: str,
    out_dir: Path,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        dataset = str(r.get("dataset", "unknown"))
        model = str(r.get("synthetic_type", "unknown"))
        grouped.setdefault((dataset, model), []).append(r)

    fig, ax = plt.subplots(figsize=(12, 6))
    for (dataset, model), model_rows in sorted(grouped.items()):
        x = [_to_float(r.get(x_key)) for r in model_rows]
        y = [_to_float(r.get("linear_barrier")) for r in model_rows]
        ax.scatter(x, y, alpha=0.75, label=f"{dataset}:{model}")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Linear Barrier")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark sweep results with graphics")
    parser.add_argument("--sweep_dir", default="gcn_modeconnect/sweeps")
    parser.add_argument("--datasets", default="cora,citeseer,squirrel_filtered,chameleon_filtered")
    parser.add_argument("--out_dir", default="gcn_modeconnect/benchmark_summary")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sweep_dir = repo_root / args.sweep_dir
    out_root = repo_root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_rows = _collect_all_rows(sweep_dir=sweep_dir, datasets=datasets)
    if all_rows:
        all_out = out_root / "all_datasets"
        all_out.mkdir(parents=True, exist_ok=True)
        _bar_plot_means_all(all_rows, all_out)
        _scatter_all(
            all_rows,
            x_key="edge_homophily",
            x_label="Realized Edge Homophily",
            title="Linear Barrier vs Homophily (All Datasets)",
            out_name="linear_barrier_vs_homophily_all_datasets.png",
            out_dir=all_out,
        )
        _scatter_all(
            all_rows,
            x_key="edge_density",
            x_label="Realized Edge Density",
            title="Linear Barrier vs Density (All Datasets)",
            out_name="linear_barrier_vs_density_all_datasets.png",
            out_dir=all_out,
        )
        _scatter_all(
            all_rows,
            x_key="degree_std",
            x_label="Degree Std",
            title="Linear Barrier vs Degree Std (All Datasets)",
            out_name="linear_barrier_vs_degree_std_all_datasets.png",
            out_dir=all_out,
        )
        _write_topk(all_rows, all_out, k=20)
        print(f"Saved combined summary graphics in {all_out}")
    else:
        print("No rows found for combined summary, skipping")

    for dataset in datasets:
        rows = _collect_dataset_rows(sweep_dir=sweep_dir, dataset=dataset)
        if not rows:
            print(f"No rows found for dataset={dataset}, skipping")
            continue

        ds_out = out_root / dataset
        ds_out.mkdir(parents=True, exist_ok=True)

        _bar_plot_means(rows, dataset, ds_out)
        _scatter_homophily(rows, dataset, ds_out)
        _scatter_density(rows, dataset, ds_out)
        _scatter_degree_std(rows, dataset, ds_out)
        _write_topk(rows, ds_out, k=10)

        print(f"Saved summary graphics for {dataset} in {ds_out}")


if __name__ == "__main__":
    main()
