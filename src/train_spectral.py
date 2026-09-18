"""Tune fixed spectral features once, then train from a saved dataset config."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from loading.DatasetInfo import DatasetInfo
from loading.LightningGraphLoader import load_datasets
from models.fixed_spectral import FixedSpectralModel


DATASETS = ["Cora", "Roman-empire", "squirrel", "chameleon"]
FILTERS = ["g_low_pass", "g_high_pass"]
SINGLE_FILTERS = {
    "low-pass": "g_low_pass",
    "high-pass": "g_high_pass",
    "band-pass": "g_band_pass",
}


def fit(network, info, config, output_dir, accelerator):
    pl.seed_everything(config["seed"], workers=True)
    model = FixedSpectralModel(info, **config["model"])
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints", filename="best",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1,
        max_epochs=config["training"]["max_epochs"],
        callbacks=[checkpoint, EarlyStopping(
            monitor="val_loss", mode="min", patience=config["training"]["patience"],
        )],
        logger=TensorBoardLogger(str(output_dir), name="", version=""),
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, log_every_n_steps=1,
    )
    trainer.fit(
        model, train_dataloaders=network.data.train_dataloader(),
        val_dataloaders=network.data.val_dataloader(),
    )
    return trainer, model, float(checkpoint.best_model_score), checkpoint.best_model_path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_dataset(dataset, args):
    filters = [SINGLE_FILTERS[args.filter]] if args.filter else FILTERS
    config_path = args.config_dir / f"{dataset}.json"
    if args.filter:
        config_path = args.config_dir / args.filter / f"{dataset}.json"
    if args.optuna:
        config = {
            "dataset": dataset, "seed": args.seed,
            "model": {"K": 10, "filters": filters},
            "training": {
                "max_epochs": args.epochs or 300,
                "patience": args.patience or 50,
            },
        }
    else:
        if not config_path.exists():
            raise FileNotFoundError(f"No saved config at {config_path}; run with --optuna first.")
        config = json.loads(config_path.read_text())
        if config["dataset"] != dataset:
            raise ValueError(f"Config dataset does not match {dataset}")
        if args.filter and config["model"]["filters"] != filters:
            raise ValueError(f"Config filters do not match {args.filter}")
        if args.epochs is not None:
            config["training"]["max_epochs"] = args.epochs
        if args.patience is not None:
            config["training"]["patience"] = args.patience

    pl.seed_everything(config["seed"], workers=True)
    network = load_datasets([dataset])[dataset]
    info = DatasetInfo(
        network.num_classes, network.num_features, dataset,
        network.class_weights, network.data.data.num_nodes,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir / dataset / timestamp
    if args.filter:
        output_dir = args.output_dir / args.filter / dataset / timestamp
    output_dir.mkdir(parents=True)

    if args.optuna:
        # Development mode never imports or runs Optuna.
        import optuna

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.enqueue_trial({
            "hidden_dim": 64, "learning_rate": 0.01,
            "dropout_rate": 0.5, "weight_decay": 5e-4,
        })

        def objective(trial):
            params = {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.8),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            }
            trial_config = {**config, "model": {**config["model"], **params}}
            trainer, model, score, checkpoint = fit(
                network, info, trial_config, output_dir / f"trial_{trial.number:03d}",
                args.accelerator,
            )
            trial.set_user_attr("checkpoint", checkpoint)
            trial.set_user_attr("epochs", trainer.current_epoch)
            return score

        study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
        config["model"].update(study.best_params)
        config["selection"] = {
            "metric": "val_loss", "value": study.best_value,
            "n_trials": len(study.trials), "best_trial": study.best_trial.number,
            "seed": args.seed,
            "checkpoint": study.best_trial.user_attrs["checkpoint"],
        }
        write_json(output_dir / "trials.json", [
            {"number": trial.number, "value": trial.value, "params": trial.params,
             "state": trial.state.name, **trial.user_attrs}
            for trial in study.trials
        ])
        # Test the chosen validation checkpoint; never select using test metrics.
        model = FixedSpectralModel(info, **config["model"])
        trainer = pl.Trainer(
            accelerator=args.accelerator, devices=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
        )
        results = trainer.test(
            model, dataloaders=network.data.test_dataloader(),
            ckpt_path=study.best_trial.user_attrs["checkpoint"],
        )[0]
        config["test_metrics"] = results
        config["created_at"] = datetime.now(timezone.utc).isoformat()
        config["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
        write_json(config_path, config)
        print(f"Saved selected hyperparameters: {config_path}", flush=True)
    else:
        trainer, model, score, checkpoint = fit(network, info, config, output_dir, args.accelerator)
        results = trainer.test(
            model, dataloaders=network.data.test_dataloader(), ckpt_path=checkpoint,
        )[0]

    write_json(output_dir / "config.json", config)
    write_json(output_dir / "test_metrics.json", results)
    print(json.dumps({"dataset": dataset, "test": results, "output_dir": str(output_dir)}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--filter", choices=SINGLE_FILTERS)
    parser.add_argument("--optuna", action="store_true", help="Search and save configs; otherwise reuse them.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/fixed_spectral"))
    parser.add_argument("--output-dir", type=Path, default=Path("spectral_runs"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, help="Override the saved training epoch limit.")
    parser.add_argument("--patience", type=int, help="Override early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for tuning; development uses the saved seed.")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for dataset in args.datasets:
        run_dataset(dataset, args)


if __name__ == "__main__":
    main()
