#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from mode_connectivity.analyzer import ModeConnectAnalyzer
from mode_connectivity.checkpoints import (
    canonicalize_dataset_name,
    resolve_checkpoint_pair,
    load_checkpoint,
    validate_checkpoint_pair,
)
from mode_connectivity.plotting import plot_linear_vs_bezier, plot_single_path
from mode_connectivity.polyformer_adapter import (
    align_state_dict_to_model,
    build_polyformer_manager,
)
from mode_connectivity.gcn_adapter import (
    align_state_dict_to_model as align_gcn_state_dict_to_model,
    build_gcn_manager,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyFormer mode-connectivity analysis between two trained runs")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "cora",
            "citeseer",
            "pubmed",
            "computers",
            "actor",
            "chameleon",
            "chameleon_filtered",
            "squirrel",
            "squirrel_filtered",
        ],
    )
    parser.add_argument("--model", type=str, choices=["polyformer", "gcn"], default="polyformer")
    parser.add_argument("--base", default="mono")
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
    parser.add_argument("--output_dir", type=str, default="mode_connectivity/outputs")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _infer_gcn_hidden_dim(ckpt: dict) -> int:
    state = ckpt["model_state_dict"]
    if "conv1.bias" in state:
        return int(state["conv1.bias"].shape[0])
    if "conv1.lin.weight" in state:
        return int(state["conv1.lin.weight"].shape[0])
    raise KeyError("Could not infer GCN hidden dimension from checkpoint state_dict")


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_name = canonicalize_dataset_name(args.dataset)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt_path_a, ckpt_path_b = resolve_checkpoint_pair(
        repo_root=repo_root,
        dataset=dataset_name,
        base=args.base,
        model=args.model,
        run_a=args.run_a,
        run_b=args.run_b,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
    )
    print(f"Checkpoint A: {ckpt_path_a}")
    print(f"Checkpoint B: {ckpt_path_b}")

    ckpt_a = load_checkpoint(ckpt_path_a, device=torch.device("cpu"))
    ckpt_b = load_checkpoint(ckpt_path_b, device=torch.device("cpu"))
    validate_checkpoint_pair(
        ckpt_a,
        ckpt_b,
        expected_dataset=dataset_name,
        expected_base=args.base,
        expected_model=args.model,
    )

    run_seed = int(ckpt_a.get("seed", 42))
    run_index = int(ckpt_a.get("run_index", args.run_a - 1))
    if args.model == "polyformer":
        _, _, _, bundle, manager = build_polyformer_manager(
            dataset_name=dataset_name,
            run_seed=run_seed,
            run_index=run_index,
            device=device,
            base=args.base,
        )
        theta_a = align_state_dict_to_model(ckpt_a["model_state_dict"], manager.model)
        theta_b = align_state_dict_to_model(ckpt_b["model_state_dict"], manager.model)
    else:
        hidden_dim = _infer_gcn_hidden_dim(ckpt_a)
        _, _, _, bundle, manager = build_gcn_manager(
            dataset_name=dataset_name,
            run_seed=run_seed,
            run_index=run_index,
            device=device,
            hidden_dim=hidden_dim,
        )
        theta_a = align_gcn_state_dict_to_model(ckpt_a["model_state_dict"], manager.model)
        theta_b = align_gcn_state_dict_to_model(ckpt_b["model_state_dict"], manager.model)

    manager.set_model_state(theta_a)
    metrics_a = manager.evaluate()
    manager.set_model_state(theta_b)
    metrics_b = manager.evaluate()

    print(
        "Endpoint validation losses: "
        f"A={metrics_a['val_loss']:.4f}, B={metrics_b['val_loss']:.4f}"
    )

    analyzer = ModeConnectAnalyzer(theta_a=theta_a, theta_b=theta_b, mm=manager, bezier_data=bundle)

    linear_metrics = analyzer.eval_linear_path(steps=args.steps)
    linear_val_losses = [linear_metrics[a]["val_loss"] for a in linear_metrics]
    linear_barrier = analyzer.barrier(linear_val_losses, L_a=metrics_a["val_loss"], L_b=metrics_b["val_loss"])

    analyzer.train_bezier(
        L_a=metrics_a["val_loss"],
        L_b=metrics_b["val_loss"],
        lr=args.bezier_lr,
        wd=args.bezier_weight_decay,
        a_steps=args.bezier_steps,
        epochs=args.bezier_epochs,
        verbose=args.verbose,
    )

    bezier_theta = analyzer.get_bezier_theta()
    if bezier_theta is None:
        raise RuntimeError("Bezier theta was not produced by train_bezier")

    bezier_metrics = analyzer.eval_bezier_path(theta=bezier_theta, steps=args.steps)
    bezier_val_losses = [bezier_metrics[a]["val_loss"] for a in bezier_metrics]
    bezier_barrier = analyzer.barrier(bezier_val_losses, L_a=metrics_a["val_loss"], L_b=metrics_b["val_loss"])

    output_dir = (repo_root / args.output_dir / dataset_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_single_path(
        linear_metrics,
        title=f"Linear interpolation path ({args.dataset})",
        output_path=output_dir / "linear_path.png",
    )
    plot_single_path(
        bezier_metrics,
        title=f"Bezier interpolation path ({args.dataset})",
        output_path=output_dir / "bezier_path.png",
    )
    plot_linear_vs_bezier(
        linear_metrics,
        bezier_metrics,
        title=f"Linear vs Bezier ({args.dataset})",
        output_path=output_dir / "linear_vs_bezier.png",
    )

    print(f"Linear barrier (val loss): {linear_barrier:.6f}")
    print(f"Bezier barrier (val loss): {bezier_barrier:.6f}")
    print(f"Saved plots in: {output_dir}")


if __name__ == "__main__":
    main()
