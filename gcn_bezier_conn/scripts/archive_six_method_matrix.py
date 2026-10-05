"""Audit and archive the 15-dataset, six-method reference-model matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = (
    "Cora",
    "Roman-empire",
    "squirrel",
    "chameleon",
    "Questions",
    "computers",
    "photo",
    "Citeseer",
    "Pubmed",
    "actor",
    "texas",
    "cornell",
    "Amazon-ratings",
    "Minesweeper",
    "Tolokers",
)
ARCHITECTURES = {
    "gcn": "GCN",
    "mlp": "MLP",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
}
METHODS = {
    "linear": ("Raw linear", "#b84a39", "-"),
    "aligned": ("Aligned linear", "#ba801f", "-"),
    "repaired": ("Aligned linear + REPAIR", "#248454", "-"),
    "bezier": ("Raw Bézier", "#285eaa", "--"),
    "aligned_bezier": ("Aligned Bézier", "#7556a5", "--"),
    "repaired_bezier": ("Aligned Bézier + REPAIR", "#555b63", ":"),
}
EXPECTED_PAIRS = ((0, 1), (2, 3), (4, 5))
TUNED_DATASETS = frozenset(DATASETS[:4])
SOURCE_PROFILE_DATASETS = frozenset(
    ("Questions", "computers", "photo", "Citeseer", "Pubmed", "Amazon-ratings", "Minesweeper")
)
FALLBACK_DATASETS = frozenset(("actor", "texas", "cornell", "Tolokers"))
LOSS_TOLERANCE = 2e-5
ACCURACY_TOLERANCE = 1e-6


def _profile_group(dataset: str) -> str:
    if dataset in TUNED_DATASETS:
        return "reused_tuned"
    if dataset in SOURCE_PROFILE_DATASETS:
        return "pinned_source_profile"
    if dataset in FALLBACK_DATASETS:
        return "local_fallback"
    return "unknown"


def _matrix(run_root: Path) -> tuple[list[str], list[str], dict, list[str]]:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        return list(DATASETS), list(ARCHITECTURES), {}, []
    try:
        manifest = json.loads(manifest_path.read_text())
        datasets = manifest["datasets"]
        architectures = manifest["architectures"]
        if not isinstance(datasets, list) or not all(isinstance(item, str) for item in datasets):
            raise ValueError("manifest datasets must be a list of strings")
        if not isinstance(architectures, list) or not all(
            isinstance(item, str) for item in architectures
        ):
            raise ValueError("manifest architectures must be a list of strings")
        errors = []
        if len(datasets) != len(set(datasets)):
            errors.append("manifest datasets contain duplicates")
        if len(architectures) != len(set(architectures)):
            errors.append("manifest architectures contain duplicates")
        provenance = {
            key: value
            for key, value in manifest.items()
            if key not in {"datasets", "architectures"}
        }
        return datasets, architectures, provenance, errors
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return list(DATASETS), list(ARCHITECTURES), {}, [f"invalid manifest.json: {error}"]


def _source_dataset(source: dict, name: str) -> dict:
    matches = [item for item in source.get("datasets", []) if item.get("data", {}).get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"source report must contain exactly one {name!r} dataset")
    return matches[0]


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _checkpoint_path(report_path: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("checkpoint path must be a nonempty string")
    return report_path.parent / relative


def _audit_configuration(report_path: Path, dataset_name: str, architecture: str):
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    report = json.loads(report_path.read_text())
    check(report.get("experiment") == "gnn_repair", "experiment is not gnn_repair")
    check(report.get("methods") == list(METHODS), "report method list is not the required six-method order")
    datasets = report.get("datasets", [])
    check(len(datasets) == 1, f"report has {len(datasets)} datasets instead of one")
    if not datasets:
        raise ValueError("report contains no dataset result")
    data = datasets[0]
    check(data.get("data", {}).get("name") == dataset_name, "dataset name does not match matrix location")
    check(data.get("model", {}).get("architecture") == architecture, "architecture does not match matrix location")

    source_path = Path(report.get("source_report", ""))
    check(source_path.is_file(), f"source report is missing: {source_path}")
    source_data = None
    if source_path.is_file():
        source_bytes = source_path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        check(source_hash == report.get("source_report_sha256"), "source report SHA-256 differs")
        source = json.loads(source_bytes)
        source_data = _source_dataset(source, dataset_name)
        check(data.get("model") == source_data.get("model"), "model differs from source report")
        check(data.get("data") == source_data.get("data"), "data metadata differs from source report")
    else:
        source_hash = report.get("source_report_sha256")

    config = data.get("config", {})
    check(config.get("curve_epochs") == 200, "effective curve_epochs is not 200")
    endpoints = data.get("endpoints", [])
    check(len(endpoints) == 6, f"found {len(endpoints)} endpoints instead of six")
    endpoint_by_seed = {endpoint.get("seed"): endpoint for endpoint in endpoints}
    check(set(endpoint_by_seed) == set(range(6)), "endpoint seeds are not 0 through 5")
    for endpoint in endpoints:
        try:
            path = _checkpoint_path(report_path, endpoint.get("checkpoint"))
            check(path.is_file(), f"endpoint checkpoint is missing: {path}")
        except ValueError as error:
            errors.append(f"endpoint checkpoint: {error}")
        for split in ("train", "val", "test"):
            for metric in ("loss", "accuracy"):
                check(
                    _finite(endpoint.get("metrics", {}).get(split, {}).get(metric)),
                    f"endpoint {endpoint.get('seed')} {split} {metric} is not finite",
                )

    split_path = report_path.parent / dataset_name / "split.pt"
    check(split_path.is_file(), f"split checkpoint is missing: {split_path}")
    pairs = data.get("pairs", [])
    actual_pairs = [(pair.get("seed_a"), pair.get("seed_b")) for pair in pairs]
    check(len(pairs) == 3, f"found {len(pairs)} pairs instead of three")
    check(actual_pairs == list(EXPECTED_PAIRS), f"pair order is {actual_pairs}, expected {list(EXPECTED_PAIRS)}")
    source_pairs = {}
    if source_data is not None:
        source_pairs = {
            (pair.get("seed_a"), pair.get("seed_b")): pair
            for pair in source_data.get("pairs", [])
        }

    replay_max = {
        "linear_loss": 0.0,
        "linear_accuracy": 0.0,
        "bezier_loss": 0.0,
        "bezier_accuracy": 0.0,
    }
    for pair in pairs:
        pair_id = (pair.get("seed_a"), pair.get("seed_b"))
        prefix = f"pair {pair_id[0]}:{pair_id[1]}"
        original = source_pairs.get(pair_id)
        check(original is not None, f"{prefix} is missing from source report")
        check(len(pair.get("aligned_history", [])) == 200, f"{prefix} aligned_history length is not 200")

        for field in (
            "checkpoint",
            "aligned_checkpoint",
            "repaired_midpoint_checkpoint",
            "repaired_bezier_midpoint_checkpoint",
        ):
            try:
                path = _checkpoint_path(report_path, pair.get(field))
                check(path.is_file(), f"{prefix} {field} is missing: {path}")
            except ValueError as error:
                errors.append(f"{prefix} {field}: {error}")

        try:
            aligned_path = _checkpoint_path(report_path, pair.get("aligned_checkpoint"))
            if aligned_path.is_file():
                import torch

                aligned_checkpoint = torch.load(aligned_path, map_location="cpu", weights_only=True)
                control = aligned_checkpoint.get("control")
                check(isinstance(control, dict) and bool(control), f"{prefix} aligned checkpoint has no control")
                check(
                    aligned_checkpoint.get("endpoint_seeds") == list(pair_id),
                    f"{prefix} aligned checkpoint endpoint seeds differ",
                )
                check(
                    aligned_checkpoint.get("model_config") == data.get("model"),
                    f"{prefix} aligned checkpoint model differs",
                )
                check(
                    aligned_checkpoint.get("split_sha256") == data.get("data", {}).get("split_sha256"),
                    f"{prefix} aligned checkpoint split differs",
                )
        except Exception as error:
            errors.append(f"{prefix} aligned checkpoint could not be read: {error}")

        grid = None
        for method in METHODS:
            path = pair.get(method)
            if not isinstance(path, dict):
                errors.append(f"{prefix} is missing method {method}")
                continue
            ts = path.get("t")
            check(isinstance(ts, list) and len(ts) == 21, f"{prefix} {method} does not have 21 points")
            if isinstance(ts, list) and len(ts) == 21:
                check(ts[0] == 0 and ts[-1] == 1, f"{prefix} {method} grid is not endpoint-inclusive")
                if grid is None:
                    grid = ts
                else:
                    check(ts == grid, f"{prefix} {method} grid differs from other methods")
            for split in ("train", "val", "test"):
                split_values = path.get("splits", {}).get(split, {})
                for metric in ("loss", "accuracy"):
                    values = split_values.get(metric)
                    valid_values = isinstance(values, list) and len(values) == 21
                    check(valid_values, f"{prefix} {method} {split} {metric} does not have 21 values")
                    if not valid_values:
                        continue
                    check(all(_finite(value) for value in values), f"{prefix} {method} {split} {metric} is not finite")
                    tolerance = LOSS_TOLERANCE if metric == "loss" else ACCURACY_TOLERANCE
                    for index, seed in ((0, pair_id[0]), (-1, pair_id[1])):
                        endpoint = endpoint_by_seed.get(seed, {})
                        expected = endpoint.get("metrics", {}).get(split, {}).get(metric)
                        if _finite(expected) and _finite(values[index]):
                            check(
                                abs(float(values[index]) - float(expected)) <= tolerance,
                                f"{prefix} {method} {split} {metric} endpoint differs",
                            )
                for summary_metric in ("barrier", "argmax_t", "max_loss", "min_accuracy"):
                    check(
                        _finite(split_values.get(summary_metric)),
                        f"{prefix} {method} {split} {summary_metric} is not finite",
                    )

        if original is not None:
            for method in ("linear", "bezier"):
                check(pair.get(method) == original.get(method), f"{prefix} raw {method} metrics were not retained")

        replay_fields = {
            "linear_loss": "midpoint_replay_max_loss_error",
            "linear_accuracy": "midpoint_replay_max_accuracy_error",
            "bezier_loss": "repaired_bezier_midpoint_replay_max_loss_error",
            "bezier_accuracy": "repaired_bezier_midpoint_replay_max_accuracy_error",
        }
        for key, field in replay_fields.items():
            value = pair.get(field)
            check(_finite(value), f"{prefix} {field} is not finite")
            if _finite(value):
                replay_max[key] = max(replay_max[key], float(value))
                tolerance = LOSS_TOLERANCE if key.endswith("loss") else ACCURACY_TOLERANCE
                check(float(value) <= tolerance, f"{prefix} {field} exceeds {tolerance:g}")

        for method, splits in pair.get("source_replay", {}).items():
            for split, replay in splits.items():
                check(
                    _finite(replay.get("max_loss_error"))
                    and replay["max_loss_error"] <= LOSS_TOLERANCE,
                    f"{prefix} {method} {split} source loss replay failed",
                )
                check(
                    replay.get("max_correct_count_difference", 2) <= 1,
                    f"{prefix} {method} {split} source accuracy replay failed",
                )

    barriers = {}
    for method in METHODS:
        value = data.get("summary", {}).get(method, {}).get("test", {}).get("barrier_mean")
        check(_finite(value), f"{method} test barrier mean is not finite")
        if _finite(value):
            barriers[method] = float(value)
        for split in ("train", "val", "test"):
            values = data.get("summary", {}).get(method, {}).get(split, {})
            check(values.get("n_pairs") == 3, f"{method} {split} summary does not cover three pairs")
            for field in ("barrier_mean", "barrier_std"):
                check(_finite(values.get(field)), f"{method} {split} {field} is not finite")

    record = {
        "dataset": dataset_name,
        "architecture": architecture,
        "profile_group": _profile_group(dataset_name),
        "report_path": str(report_path),
        "source_report": str(source_path),
        "source_report_sha256": source_hash,
        "endpoint_count": len(endpoints),
        "pair_count": len(pairs),
        "test_barrier_means": barriers,
        "midpoint_replay_max": replay_max,
        "model": data.get("model"),
        "errors": errors,
    }
    return report, data, record


def _plot_configuration(data: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for column, split in enumerate(("train", "val", "test")):
        for row, metric in enumerate(("loss", "accuracy")):
            ax = axes[row, column]
            for method, (label, color, style) in METHODS.items():
                ts = data["pairs"][0][method]["t"]
                curves = np.asarray(
                    [pair[method]["splits"][split][metric] for pair in data["pairs"]],
                    dtype=float,
                )
                mean, std = curves.mean(0), curves.std(0)
                ax.plot(ts, mean, color=color, linestyle=style, label=label, linewidth=1.7)
                ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.10)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 1)
            if row == 0:
                ax.set_title(split.capitalize())
            else:
                ax.set_xlabel("Path position t")
                ax.set_ylim(0, 1)
            if column == 0:
                ax.set_ylabel("Cross entropy" if metric == "loss" else "Accuracy")
    model = data["model"]
    display = ARCHITECTURES.get(model["architecture"], model["architecture"])
    figure.suptitle(
        f"{data['data']['name']} / {display}: six connectivity paths\n"
        "Three endpoint pairs; shading is population standard deviation"
    )
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.09, 1, 0.93))
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"connectivity.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_overviews(
    output: Path,
    datasets: list[str],
    architectures: list[str],
    available: dict[tuple[str, str], dict],
) -> list[str]:
    errors = []
    panels = datasets[:15]
    if len(datasets) != 15:
        errors.append(f"overview requires 15 datasets, manifest has {len(datasets)}")
        panels = (datasets + [f"missing-{index}" for index in range(15)])[:15]
    for architecture in architectures:
        for metric in ("loss", "accuracy"):
            figure, axes = plt.subplots(5, 3, figsize=(14, 17), sharex=True)
            for index, dataset in enumerate(panels):
                ax = axes.flat[index]
                data = available.get((dataset, architecture))
                if data is None:
                    ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
                else:
                    try:
                        for method, (label, color, style) in METHODS.items():
                            ts = data["pairs"][0][method]["t"]
                            curves = np.asarray(
                                [pair[method]["splits"]["test"][metric] for pair in data["pairs"]],
                                dtype=float,
                            )
                            mean, std = curves.mean(0), curves.std(0)
                            ax.plot(ts, mean, color=color, linestyle=style, label=label, linewidth=1.4)
                            ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.09)
                    except (KeyError, TypeError, ValueError) as error:
                        errors.append(f"{dataset}/{architecture} overview {metric}: {error}")
                        ax.text(0.5, 0.5, "Invalid report", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(dataset, fontsize=10)
                ax.set_xlim(0, 1)
                ax.grid(alpha=0.2)
                if metric == "accuracy":
                    ax.set_ylim(0, 1)
                if index % 3 == 0:
                    ax.set_ylabel("Test cross entropy" if metric == "loss" else "Test accuracy")
                if index // 3 == 4:
                    ax.set_xlabel("Path position t")
            display = ARCHITECTURES.get(architecture, architecture)
            figure.suptitle(
                f"{display}: test {metric} across 15 datasets\n"
                "Three endpoint pairs per available panel; shading is population standard deviation",
                fontsize=15,
            )
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if not handles:
                for ax in axes.flat[1:]:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        break
            if handles:
                figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
            figure.tight_layout(rect=(0, 0.045, 1, 0.955))
            for suffix in ("png", "pdf"):
                figure.savefig(
                    output / f"{architecture}_test_{metric}_overview.{suffix}",
                    dpi=180,
                    bbox_inches="tight",
                )
            plt.close(figure)
    return errors


def _write_csv(output: Path, records: list[dict]) -> None:
    fields = [
        "dataset",
        "architecture",
        "profile_group",
        "status",
        "report_path",
        "error_count",
        "errors",
        "endpoint_count",
        "pair_count",
        "source_report_sha256",
        "linear_midpoint_max_loss_error",
        "linear_midpoint_max_accuracy_error",
        "bezier_midpoint_max_loss_error",
        "bezier_midpoint_max_accuracy_error",
    ] + [f"{method}_test_barrier_mean" for method in METHODS]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            replay = record.get("midpoint_replay_max", {})
            row = {
                "dataset": record["dataset"],
                "architecture": record["architecture"],
                "profile_group": record["profile_group"],
                "status": record["status"],
                "report_path": record.get("report_path", ""),
                "error_count": len(record.get("errors", [])),
                "errors": " | ".join(record.get("errors", [])),
                "endpoint_count": record.get("endpoint_count", 0),
                "pair_count": record.get("pair_count", 0),
                "source_report_sha256": record.get("source_report_sha256", ""),
                "linear_midpoint_max_loss_error": replay.get("linear_loss", ""),
                "linear_midpoint_max_accuracy_error": replay.get("linear_accuracy", ""),
                "bezier_midpoint_max_loss_error": replay.get("bezier_loss", ""),
                "bezier_midpoint_max_accuracy_error": replay.get("bezier_accuracy", ""),
            }
            for method in METHODS:
                row[f"{method}_test_barrier_mean"] = record.get("test_barrier_means", {}).get(method, "")
            writer.writerow(row)


def _write_readme(output: Path, summary: dict, records: list[dict]) -> None:
    complete = summary["complete"]
    lines = [
        "# Six-method GNN connectivity matrix",
        "",
        f"Audit status: **{'complete' if complete else 'incomplete or failed'}**. "
        f"{summary['audited_configurations']} of {summary['expected_configurations']} configurations passed; "
        f"{summary['available_configurations']} reports were available.",
        "",
        "The intended matrix mixes 16 reused tuned configurations, 28 pinned source-profile "
        "configurations, and 16 explicit local-fallback configurations. These groups must be "
        "reported separately because their endpoint budgets and provenance differ.",
        "",
        "Methods: raw linear, aligned linear, aligned linear + REPAIR, raw Bézier, "
        "aligned Bézier, and aligned Bézier + REPAIR. Repaired paths are post-hoc "
        "activation corrections and are not straight or quadratic parameter-space paths.",
        "",
        "## Overview figures",
        "",
    ]
    for architecture in summary["architectures"]:
        display = ARCHITECTURES.get(architecture, architecture)
        lines.append(
            f"- {display}: [test loss]({architecture}_test_loss_overview.png), "
            f"[test accuracy]({architecture}_test_accuracy_overview.png)"
        )
    lines.extend(
        [
            "",
            "## Provenance groups",
            "",
            "| Group | Intended | Available | Passed |",
            "|---|---:|---:|---:|",
        ]
    )
    labels = {
        "reused_tuned": "Reused tuned endpoints",
        "pinned_source_profile": "Pinned source profiles",
        "local_fallback": "Local fallback profiles",
        "unknown": "Unknown",
    }
    for key, values in summary["profile_groups"].items():
        lines.append(
            f"| {labels.get(key, key)} | {values['expected']} | {values['available']} | {values['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Configuration audit",
            "",
            "| Dataset | Model | Group | Status | Errors |",
            "|---|---|---|---|---|",
        ]
    )
    for record in records:
        errors = "; ".join(record.get("errors", [])) or ""
        lines.append(
            f"| {record['dataset']} | {ARCHITECTURES.get(record['architecture'], record['architecture'])} "
            f"| {labels.get(record['profile_group'], record['profile_group'])} | {record['status']} | {errors} |"
        )
    if summary["missing_configurations"]:
        lines.extend(["", "## Missing configurations", ""])
        lines.extend(
            f"- {item['dataset']} / {item['architecture']}" for item in summary["missing_configurations"]
        )
    if summary["global_errors"]:
        lines.extend(["", "## Archive errors", ""])
        lines.extend(f"- {error}" for error in summary["global_errors"])
    lines.extend(
        [
            "",
            "Each available configuration folder contains a copied report plus PNG and PDF "
            "plots. Checkpoint weights remain under the run root. The audit checks their "
            "existence, source identity, all path metrics, and both repaired midpoint replays.",
        ]
    )
    if not complete:
        lines.extend(
            [
                "",
                "This archive does not claim matrix completion. Resolve every missing or failed "
                "configuration and rerun the audit before using aggregate conclusions.",
            ]
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def archive(run_root: Path, output: Path) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    datasets, architectures, manifest_provenance, global_errors = _matrix(run_root)
    matrix = [(dataset, architecture) for dataset in datasets for architecture in architectures]
    if len(matrix) != 60:
        global_errors.append(f"matrix has {len(matrix)} configurations instead of 60")

    records = []
    available_data: dict[tuple[str, str], dict] = {}
    missing = []
    for dataset, architecture in matrix:
        report_path = run_root / "analysis" / dataset / architecture / "report.json"
        group = _profile_group(dataset)
        if not report_path.is_file():
            missing.append({"dataset": dataset, "architecture": architecture, "report_path": str(report_path)})
            records.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "profile_group": group,
                    "status": "missing",
                    "report_path": str(report_path),
                    "errors": ["report.json is missing"],
                }
            )
            continue

        destination = output / dataset / architecture
        destination.mkdir(parents=True, exist_ok=True)
        try:
            report, data, record = _audit_configuration(report_path, dataset, architecture)
            shutil.copy2(report_path, destination / "report.json")
            available_data[(dataset, architecture)] = data
            try:
                _plot_configuration(data, destination)
            except Exception as error:
                record["errors"].append(f"per-configuration plot failed: {error}")
            record["status"] = "passed" if not record["errors"] else "failed"
        except Exception as error:
            try:
                shutil.copy2(report_path, destination / "report.json")
            except OSError:
                pass
            record = {
                "dataset": dataset,
                "architecture": architecture,
                "profile_group": group,
                "status": "failed",
                "report_path": str(report_path),
                "errors": [f"report audit crashed: {error}"],
            }
        records.append(record)

    global_errors.extend(_plot_overviews(output, datasets, architectures, available_data))
    passed = sum(record["status"] == "passed" for record in records)
    available_count = sum(record["status"] != "missing" for record in records)
    groups = {}
    for group in ("reused_tuned", "pinned_source_profile", "local_fallback", "unknown"):
        group_records = [record for record in records if record["profile_group"] == group]
        if group_records:
            groups[group] = {
                "expected": len(group_records),
                "available": sum(record["status"] != "missing" for record in group_records),
                "passed": sum(record["status"] == "passed" for record in group_records),
            }
    complete = len(matrix) == 60 and available_count == 60 and passed == 60 and not global_errors
    summary = {
        "schema_version": 1,
        "complete": complete,
        "run_root": str(run_root),
        "manifest_provenance": manifest_provenance,
        "datasets": datasets,
        "architectures": architectures,
        "methods": list(METHODS),
        "expected_configurations": len(matrix),
        "available_configurations": available_count,
        "audited_configurations": passed,
        "failed_configurations": [
            {
                "dataset": record["dataset"],
                "architecture": record["architecture"],
                "errors": record["errors"],
            }
            for record in records
            if record["status"] == "failed"
        ],
        "missing_configurations": missing,
        "profile_groups": groups,
        "global_errors": global_errors,
        "runs": {
            record["dataset"]: {
                candidate["architecture"]: candidate
                for candidate in records
                if candidate["dataset"] == record["dataset"]
            }
            for record in records
        },
    }
    _write_csv(output, records)
    _write_readme(output, summary, records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete": complete,
                "expected_configurations": len(matrix),
                "available_configurations": available_count,
                "audited_configurations": passed,
                "failed_configurations": len(summary["failed_configurations"]),
                "missing_configurations": len(missing),
                "global_errors": global_errors,
            },
            indent=2,
        )
    )
    return complete


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit and archive the 15-dataset by four-model six-method matrix."
    )
    parser.add_argument("run_root", type=Path, help="Matrix root containing analysis/Dataset/architecture/report.json")
    parser.add_argument("output", type=Path, help="Compact archive output directory")
    args = parser.parse_args(argv)
    return 0 if archive(args.run_root.expanduser().resolve(), args.output.expanduser().resolve()) else 1


if __name__ == "__main__":
    sys.exit(main())
