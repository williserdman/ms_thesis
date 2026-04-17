#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from gcn_modeconnect.run_barrier import compute_barrier_analysis
from gcn_modeconnect.train_gcn import set_all_seeds, train_one_run, DEFAULT_SEEDS


def _parse_float_list(raw: str) -> list[float]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError(f"Expected at least one float in '{raw}'")
    return out


def _parse_optional_float_list(raw: str) -> list[float | None]:
    out: list[float | None] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"none", "null", "na"}:
            out.append(None)
        else:
            out.append(float(item))
    if not out:
        raise ValueError(f"Expected at least one homophily value in '{raw}'")
    return out


def _format_num_for_id(x: float | None) -> str:
    if x is None:
        return "none"
    text = f"{x:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _condition_id(h: float | None, s: float, d: float, synthetic: bool, synthetic_type: str) -> str:
    syn = "syn1" if synthetic else "syn0"
    return f"h{_format_num_for_id(h)}_s{_format_num_for_id(s)}_d{_format_num_for_id(d)}_{syn}_{synthetic_type}"


def _build_train_args(args: argparse.Namespace, condition_id: str, h: float | None, s: float, d: float) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        split_index=args.split_index,
        split_seed=args.split_seed,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        condition_id=condition_id,
        homophily_target=h,
        sparsity_keep=s,
        degree_gamma=d,
        synthetic=args.synthetic,
        synthetic_type=args.synthetic_type,
        synthetic_p_in=args.synthetic_p_in,
        synthetic_p_out=args.synthetic_p_out,
        synthetic_target_edges=args.synthetic_target_edges,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Factorial GCN barrier sweep over graph properties")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--homophily_values", default="0.2,0.8")
    parser.add_argument("--sparsity_values", default="1.0,0.5")
    parser.add_argument("--degree_values", default="0.7,1.3")
    parser.add_argument("--limit_conditions", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--early_stopping", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--bezier_steps", type=int, default=11)
    parser.add_argument("--bezier_epochs", type=int, default=120)
    parser.add_argument("--bezier_lr", type=float, default=0.01)
    parser.add_argument("--bezier_weight_decay", type=float, default=0.0)

    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--synthetic_type",
        type=str,
        choices=["label_sbm", "dcsbm", "config_model"],
        default="label_sbm",
    )
    parser.add_argument("--synthetic_p_in", type=float, default=0.10)
    parser.add_argument("--synthetic_p_out", type=float, default=0.01)
    parser.add_argument("--synthetic_target_edges", type=int, default=None)

    parser.add_argument("--output_dir", default="gcn_modeconnect/sweeps")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    h_vals = _parse_optional_float_list(args.homophily_values)
    s_vals = _parse_float_list(args.sparsity_values)
    d_vals = _parse_float_list(args.degree_values)

    conditions: list[tuple[float | None, float, float]] = []
    for h in h_vals:
        for s in s_vals:
            for d in d_vals:
                conditions.append((h, s, d))

    if args.limit_conditions > 0:
        conditions = conditions[: args.limit_conditions]

    sweep_root = Path(__file__).resolve().parents[1] / args.output_dir / args.dataset.lower()
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(conditions)
    print(f"Running {total} conditions on dataset={args.dataset} device={device}")

    for idx, (h, s, d) in enumerate(conditions, start=1):
        cond_id = _condition_id(h, s, d, args.synthetic, args.synthetic_type)
        print(f"[{idx}/{total}] condition={cond_id}")
        train_args = _build_train_args(args=args, condition_id=cond_id, h=h, s=s, d=d)

        train_results: list[tuple[float, float, Path]] = []
        for run_idx in range(2):
            run_number = run_idx + 1
            model_seed = DEFAULT_SEEDS[run_idx % len(DEFAULT_SEEDS)]
            set_all_seeds(model_seed)
            best_val, best_test, ckpt = train_one_run(train_args, run_number, model_seed, device)
            train_results.append((best_val, best_test, ckpt))

        barrier_result = compute_barrier_analysis(
            dataset=args.dataset,
            condition_id=cond_id,
            run_a=1,
            run_b=2,
            steps=args.steps,
            bezier_steps=args.bezier_steps,
            bezier_epochs=args.bezier_epochs,
            bezier_lr=args.bezier_lr,
            bezier_weight_decay=args.bezier_weight_decay,
            device=str(device),
            output_dir="gcn_modeconnect/outputs",
            verbose=args.verbose,
        )

        ckpt_a_path = Path(barrier_result["checkpoint_a"])
        ckpt_b_path = Path(barrier_result["checkpoint_b"])
        ckpt_a = torch.load(ckpt_a_path, map_location="cpu", weights_only=False)
        stats = ckpt_a.get("transform_stats", {})

        row = {
            "dataset": args.dataset.lower(),
            "condition_id": cond_id,
            "synthetic": bool(args.synthetic),
            "synthetic_type": args.synthetic_type,
            "homophily_target": h,
            "sparsity_keep": s,
            "degree_gamma": d,
            "split_index": int(args.split_index),
            "split_seed": int(args.split_seed),
            "run1_best_val_acc": float(train_results[0][0]),
            "run1_test_at_best": float(train_results[0][1]),
            "run2_best_val_acc": float(train_results[1][0]),
            "run2_test_at_best": float(train_results[1][1]),
            "run1_checkpoint": str(train_results[0][2]),
            "run2_checkpoint": str(train_results[1][2]),
            "linear_barrier": float(barrier_result["linear_barrier"]),
            "bezier_barrier": float(barrier_result["bezier_barrier"]),
            "endpoint_a_val_loss": float(barrier_result["endpoint_metrics_a"]["val_loss"]),
            "endpoint_b_val_loss": float(barrier_result["endpoint_metrics_b"]["val_loss"]),
            "num_nodes": stats.get("num_nodes"),
            "num_directed_edges": stats.get("num_directed_edges"),
            "num_undirected_edges": stats.get("num_undirected_edges"),
            "edge_density": stats.get("edge_density"),
            "edge_homophily": stats.get("edge_homophily"),
            "degree_mean": stats.get("degree_mean"),
            "degree_std": stats.get("degree_std"),
            "output_dir": barrier_result["output_dir"],
        }
        rows.append(row)

    model_tag = args.synthetic_type if args.synthetic else "real"
    csv_path = sweep_root / f"barrier_results_{model_tag}.csv"
    json_path = sweep_root / f"barrier_results_{model_tag}.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    linear_vals = [float(r["linear_barrier"]) for r in rows]
    bezier_vals = [float(r["bezier_barrier"]) for r in rows]
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Linear barrier mean={np.mean(linear_vals):.6f}, std={np.std(linear_vals):.6f}")
    print(f"Bezier barrier mean={np.mean(bezier_vals):.6f}, std={np.std(bezier_vals):.6f}")


if __name__ == "__main__":
    main()
