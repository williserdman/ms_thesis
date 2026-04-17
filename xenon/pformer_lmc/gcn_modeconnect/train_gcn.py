#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from gcn_modeconnect.adapter import build_gcn_manager
from gcn_modeconnect.graph_transforms import GraphCondition


DEFAULT_SEEDS = [1941488137, 4198936517, 983997847, 4023022221]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_condition_from_args(args: argparse.Namespace) -> GraphCondition:
    return GraphCondition(
        condition_id=args.condition_id,
        homophily_target=args.homophily_target,
        sparsity_keep=args.sparsity_keep,
        degree_gamma=args.degree_gamma,
        synthetic=args.synthetic,
        synthetic_type=args.synthetic_type,
        synthetic_p_in=args.synthetic_p_in,
        synthetic_p_out=args.synthetic_p_out,
        synthetic_target_edges=args.synthetic_target_edges,
    )


def train_one_run(args: argparse.Namespace, run_number: int, model_seed: int, device: torch.device) -> tuple[float, float, Path]:
    condition = build_condition_from_args(args)
    model, data, _, _, meta = build_gcn_manager(
        dataset_name=args.dataset,
        condition=condition,
        split_index=args.split_index,
        split_seed=args.split_seed,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    test_at_best = 0.0
    best_epoch = 0
    bad_counter = 0

    ckpt_dir = (
        Path(__file__).resolve().parents[1]
        / "gcn_modeconnect"
        / "checkpoints"
        / args.dataset.lower()
        / args.condition_id
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"gcn_run{run_number}_seed{model_seed}.pt"

    def eval_split(mask: torch.Tensor) -> tuple[float, float]:
        logits = model(data)
        split_logits = logits[mask]
        split_y = data.y[mask]
        loss = F.cross_entropy(split_logits, split_y)
        pred = split_logits.argmax(dim=1)
        acc = pred.eq(split_y).sum().item() / max(1, mask.sum().item())
        return float(loss.item()), float(acc)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        train_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, train_acc = eval_split(data.train_mask)
            _, val_acc = eval_split(data.val_mask)
            _, test_acc = eval_split(data.test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_at_best = test_acc
            best_epoch = epoch
            bad_counter = 0
            torch.save(
                {
                    "model": "gcn",
                    "dataset": args.dataset.lower(),
                    "condition_id": args.condition_id,
                    "condition": condition.to_dict(),
                    "run_index": run_number - 1,
                    "seed": model_seed,
                    "split_index": args.split_index,
                    "split_seed": args.split_seed,
                    "epoch": best_epoch,
                    "num_features": int(meta["num_features"]),
                    "num_classes": int(meta["num_classes"]),
                    "hidden_channels": args.hidden_channels,
                    "dropout": args.dropout,
                    "transform_stats": meta["transform_stats"],
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": float(best_val_acc),
                    "test_acc_at_best_val": float(test_at_best),
                },
                ckpt_path,
            )
        else:
            bad_counter += 1

        if bad_counter >= args.early_stopping:
            break

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(
                f"run={run_number} epoch={epoch} train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
            )

    return best_val_acc, test_at_best, ckpt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train independent GCN endpoints for barrier analysis")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--early_stopping", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--condition_id", type=str, default="baseline")
    parser.add_argument("--homophily_target", type=float, default=None)
    parser.add_argument("--sparsity_keep", type=float, default=1.0)
    parser.add_argument("--degree_gamma", type=float, default=1.0)
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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Training GCN on dataset={args.dataset} condition={args.condition_id} device={device}")
    all_results = []

    for run_idx in range(args.runs):
        run_number = run_idx + 1
        model_seed = DEFAULT_SEEDS[run_idx % len(DEFAULT_SEEDS)]
        set_all_seeds(model_seed)
        best_val, best_test, ckpt = train_one_run(args, run_number, model_seed, device)
        all_results.append((best_val, best_test, str(ckpt)))
        print(f"run={run_number} best_val={best_val:.4f} test_at_best={best_test:.4f} checkpoint={ckpt}")

    if all_results:
        mean_val = float(np.mean([x[0] for x in all_results]))
        mean_test = float(np.mean([x[1] for x in all_results]))
        print(f"summary mean_val={mean_val:.4f} mean_test={mean_test:.4f}")


if __name__ == "__main__":
    main()
