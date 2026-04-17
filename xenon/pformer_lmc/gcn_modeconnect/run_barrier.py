#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from gcn_modeconnect.adapter import align_state_dict_to_model, build_gcn_manager
from gcn_modeconnect.analyzer import ModeConnectAnalyzer
from gcn_modeconnect.checkpoints import (
    load_checkpoint,
    resolve_checkpoint_pair,
    validate_checkpoint_pair,
)
from gcn_modeconnect.graph_transforms import GraphCondition
from gcn_modeconnect.plotting import plot_linear_vs_bezier, plot_single_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GCN mode-connectivity analysis")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition_id", default="baseline")
    parser.add_argument("--run_a", type=int, default=1)
    parser.add_argument("--run_b", type=int, default=2)
    parser.add_argument("--checkpoint_a", type=str, default=None)
    parser.add_argument("--checkpoint_b", type=str, default=None)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--bezier_steps", type=int, default=11)
    parser.add_argument("--bezier_epochs", type=int, default=120)
    parser.add_argument("--bezier_lr", type=float, default=0.01)
    parser.add_argument("--bezier_weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="gcn_modeconnect/outputs")
    parser.add_argument("--verbose", action="store_true")
    return parser


def compute_barrier_analysis(
    dataset: str,
    condition_id: str,
    run_a: int = 1,
    run_b: int = 2,
    checkpoint_a: str | None = None,
    checkpoint_b: str | None = None,
    steps: int = 21,
    bezier_steps: int = 11,
    bezier_epochs: int = 120,
    bezier_lr: float = 0.01,
    bezier_weight_decay: float = 0.0,
    device: str = "cuda:0",
    output_dir: str = "gcn_modeconnect/outputs",
    verbose: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, using CPU")
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)

    ckpt_path_a, ckpt_path_b = resolve_checkpoint_pair(
        repo_root=repo_root,
        dataset=dataset,
        condition_id=condition_id,
        run_a=run_a,
        run_b=run_b,
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
    )
    print(f"Checkpoint A: {ckpt_path_a}")
    print(f"Checkpoint B: {ckpt_path_b}")

    ckpt_a = load_checkpoint(ckpt_path_a, device=torch.device("cpu"))
    ckpt_b = load_checkpoint(ckpt_path_b, device=torch.device("cpu"))
    validate_checkpoint_pair(ckpt_a, ckpt_b, dataset=dataset, condition_id=condition_id)

    condition = GraphCondition(**ckpt_a["condition"])
    model, _, bundle, manager, _ = build_gcn_manager(
        dataset_name=dataset,
        condition=condition,
        split_index=int(ckpt_a.get("split_index", 0)),
        split_seed=int(ckpt_a.get("split_seed", 42)),
        hidden_channels=int(ckpt_a["hidden_channels"]),
        dropout=float(ckpt_a["dropout"]),
        device=torch_device,
    )

    theta_a = align_state_dict_to_model(ckpt_a["model_state_dict"], model)
    theta_b = align_state_dict_to_model(ckpt_b["model_state_dict"], model)

    manager.set_model_state(theta_a)
    metrics_a = manager.evaluate()
    manager.set_model_state(theta_b)
    metrics_b = manager.evaluate()

    print(
        "Endpoint validation losses: "
        f"A={metrics_a['val_loss']:.4f}, B={metrics_b['val_loss']:.4f}"
    )

    analyzer = ModeConnectAnalyzer(theta_a=theta_a, theta_b=theta_b, mm=manager, bezier_data=bundle)

    linear_metrics = analyzer.eval_linear_path(steps=steps)
    linear_val_losses = [linear_metrics[a]["val_loss"] for a in linear_metrics]
    linear_barrier = analyzer.barrier(linear_val_losses, L_a=metrics_a["val_loss"], L_b=metrics_b["val_loss"])

    analyzer.train_bezier(
        L_a=metrics_a["val_loss"],
        L_b=metrics_b["val_loss"],
        lr=bezier_lr,
        wd=bezier_weight_decay,
        a_steps=bezier_steps,
        epochs=bezier_epochs,
        verbose=verbose,
    )

    bezier_theta = analyzer.get_bezier_theta()
    if bezier_theta is None:
        raise RuntimeError("Bezier theta was not produced")

    bezier_metrics = analyzer.eval_bezier_path(theta=bezier_theta, steps=steps)
    bezier_val_losses = [bezier_metrics[a]["val_loss"] for a in bezier_metrics]
    bezier_barrier = analyzer.barrier(bezier_val_losses, L_a=metrics_a["val_loss"], L_b=metrics_b["val_loss"])

    out_dir = (repo_root / output_dir / dataset.lower() / condition_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_single_path(
        linear_metrics,
        title=f"GCN Linear path ({dataset}, {condition_id})",
        output_path=out_dir / "linear_path.png",
    )
    plot_single_path(
        bezier_metrics,
        title=f"GCN Bezier path ({dataset}, {condition_id})",
        output_path=out_dir / "bezier_path.png",
    )
    plot_linear_vs_bezier(
        linear_metrics,
        bezier_metrics,
        title=f"GCN Linear vs Bezier ({dataset}, {condition_id})",
        output_path=out_dir / "linear_vs_bezier.png",
    )

    print(f"Linear barrier (val loss): {linear_barrier:.6f}")
    print(f"Bezier barrier (val loss): {bezier_barrier:.6f}")
    print(f"Saved plots in: {out_dir}")

    return {
        "checkpoint_a": str(ckpt_path_a),
        "checkpoint_b": str(ckpt_path_b),
        "endpoint_metrics_a": metrics_a,
        "endpoint_metrics_b": metrics_b,
        "linear_barrier": float(linear_barrier),
        "bezier_barrier": float(bezier_barrier),
        "output_dir": str(out_dir),
    }


def main() -> None:
    args = _build_parser().parse_args()
    compute_barrier_analysis(
        dataset=args.dataset,
        condition_id=args.condition_id,
        run_a=args.run_a,
        run_b=args.run_b,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        steps=args.steps,
        bezier_steps=args.bezier_steps,
        bezier_epochs=args.bezier_epochs,
        bezier_lr=args.bezier_lr,
        bezier_weight_decay=args.bezier_weight_decay,
        device=args.device,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
