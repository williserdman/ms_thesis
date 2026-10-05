"""Publication-oriented plots for interpolation experiment reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


_VARIANTS = ("unaligned", "aligned", "repaired")
_COLORS = {
    "unaligned": "#0072B2",  # blue
    "aligned": "#D55E00",  # vermillion
    "repaired": "#009E73",  # green
}


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"report field {field!r} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"report field {field!r} must be finite")
    return number


def _read_report(report_path: str | Path) -> Mapping[str, Any]:
    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read report JSON: {path}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("report must use schema_version 1")

    endpoints = report.get("endpoints")
    curve = report.get("curve")
    if not isinstance(endpoints, list) or len(endpoints) < 2:
        raise ValueError("report endpoints must contain two metric objects")
    if not isinstance(curve, list) or not curve:
        raise ValueError("report curve must contain at least one row")
    for endpoint_index, endpoint in enumerate(endpoints[:2]):
        if not isinstance(endpoint, dict):
            raise ValueError(f"endpoint {endpoint_index} must be an object")
        _number(endpoint.get("loss"), f"endpoints[{endpoint_index}].loss")
        _number(endpoint.get("accuracy"), f"endpoints[{endpoint_index}].accuracy")
    for row_index, row in enumerate(curve):
        if not isinstance(row, dict):
            raise ValueError(f"curve row {row_index} must be an object")
        _number(row.get("alpha"), f"curve[{row_index}].alpha")
        for variant in _VARIANTS:
            metrics = row.get(variant)
            if not isinstance(metrics, dict):
                raise ValueError(f"curve[{row_index}] is missing {variant!r} metrics")
            _number(metrics.get("loss"), f"curve[{row_index}].{variant}.loss")
            _number(metrics.get("accuracy"), f"curve[{row_index}].{variant}.accuracy")
    return report


def _architecture_name(architecture: Any) -> str:
    if isinstance(architecture, dict):
        return str(architecture.get("name", "unknown"))
    return str(architecture or "unknown")


def plot_report(report_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Plot a schema-v1 report and return its PNG and PDF output paths."""

    report = _read_report(report_path)

    # Import plotting only when the API is used, and force a headless backend.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = sorted(report["curve"], key=lambda row: float(row["alpha"]))
    alphas = [_number(row["alpha"], "curve.alpha") for row in rows]
    endpoints = report["endpoints"]
    endpoint_a = endpoints[0]
    endpoint_b = endpoints[1]
    baseline_loss = [
        (1.0 - alpha) * float(endpoint_a["loss"]) + alpha * float(endpoint_b["loss"])
        for alpha in alphas
    ]
    baseline_accuracy = [
        100.0
        * ((1.0 - alpha) * float(endpoint_a["accuracy"]) + alpha * float(endpoint_b["accuracy"]))
        for alpha in alphas
    ]

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    try:
        for variant in _VARIANTS:
            axes[0].plot(
                alphas,
                [float(row[variant]["loss"]) for row in rows],
                marker="o",
                linewidth=2,
                color=_COLORS[variant],
                label=variant.capitalize(),
            )
            axes[1].plot(
                alphas,
                [100.0 * float(row[variant]["accuracy"]) for row in rows],
                marker="o",
                linewidth=2,
                color=_COLORS[variant],
                label=variant.capitalize(),
            )

        axes[0].plot(
            alphas,
            baseline_loss,
            linestyle="--",
            color="#333333",
            linewidth=1.5,
            label="Linear endpoint baseline",
        )
        axes[1].plot(
            alphas,
            baseline_accuracy,
            linestyle="--",
            color="#333333",
            linewidth=1.5,
            label="Linear endpoint baseline",
        )
        axes[0].set_ylabel("Test cross-entropy")
        axes[1].set_ylabel("Test accuracy (%)")
        for axis in axes:
            axis.set_xlabel("Alpha")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
            axis.grid(True, alpha=0.25)
        figure.legend(
            *axes[0].get_legend_handles_labels(),
            loc="outside lower center", ncol=2, frameon=False, fontsize=9,
        )
        dataset = str(report.get("dataset", "dataset"))
        architecture = _architecture_name(report.get("architecture"))
        figure.suptitle(f"{dataset} · {architecture}")

        destination = Path(output_dir) if output_dir is not None else Path(report_path).parent
        destination.mkdir(parents=True, exist_ok=True)
        png_path = destination / "interpolation.png"
        pdf_path = destination / "interpolation.pdf"
        figure.savefig(png_path, dpi=180)
        figure.savefig(pdf_path)
    finally:
        plt.close(figure)
    return [png_path, pdf_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot an interpolation report")
    parser.add_argument("report_json", type=Path, help="schema-v1 report JSON")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    for path in plot_report(args.report_json, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
