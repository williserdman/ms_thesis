import datetime
import json
import argparse

import pytorch_lightning as pl
import torch

from src.main import _detect_accelerator
from src.optuna_trainer import OptunaTrainer
from src.two_stage_trainer import TwoStageTrainer

SEED = 42
SPLIT_INDEX = 1
SPLIT_SEED = 42
OPTUNA_SEED = 42

# Fast smoke baseline over a small representative subset.
SMOKE_DATASETS = ["Questions", "Cora", "texas"]

# Toggle pipeline for smoke baseline.
TWO_STAGE = True
CUT_TYPE = "maxcut"

# Tight trial budget for quick signal.
N_TRIALS_SINGLE_STAGE = 2
N_TRIALS_S1 = 2
N_TRIALS_S2 = 2

# Tight epoch budget for quick turnaround.
TRIAL_MAX_EPOCHS = 40
RETRAIN_MAX_EPOCHS = 80


def _write_results(path, results):
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run fast smoke baseline experiments.")
    parser.add_argument("--datasets", nargs="+", default=SMOKE_DATASETS)
    parser.add_argument("--single-stage", action="store_true")
    parser.add_argument("--cut-type", default=CUT_TYPE)
    parser.add_argument("--n-trials-single", type=int, default=N_TRIALS_SINGLE_STAGE)
    parser.add_argument("--n-trials-s1", type=int, default=N_TRIALS_S1)
    parser.add_argument("--n-trials-s2", type=int, default=N_TRIALS_S2)
    parser.add_argument("--trial-max-epochs", type=int, default=TRIAL_MAX_EPOCHS)
    parser.add_argument("--retrain-max-epochs", type=int, default=RETRAIN_MAX_EPOCHS)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _make_single_stage_trainer(accelerator: str):
    return OptunaTrainer(
        accelerator=accelerator,
        device="auto",
        split_index=SPLIT_INDEX,
        split_seed=SPLIT_SEED,
        optuna_seed=OPTUNA_SEED,
        trial_max_epochs=TRIAL_MAX_EPOCHS,
        best_max_epochs=RETRAIN_MAX_EPOCHS,
        trial_patience=10,
        best_patience=15,
    )


def _make_two_stage_trainer(accelerator: str):
    return TwoStageTrainer(
        accelerator=accelerator,
        device="auto",
        split_index=SPLIT_INDEX,
        split_seed=SPLIT_SEED,
        optuna_seed=OPTUNA_SEED,
        s1_trial_max_epochs=TRIAL_MAX_EPOCHS,
        s2_trial_max_epochs=TRIAL_MAX_EPOCHS,
        s1_retrain_max_epochs=RETRAIN_MAX_EPOCHS,
        s2_retrain_max_epochs=RETRAIN_MAX_EPOCHS,
        s1_trial_patience=10,
        s2_trial_patience=10,
        s1_retrain_patience=15,
        s2_retrain_patience=15,
    )


def run_smoke_test():
    args = _parse_args()

    pl.seed_everything(SEED, workers=True)

    accelerator, _ = _detect_accelerator()

    results = []

    two_stage = not args.single_stage
    datasets = args.datasets
    cut_type = args.cut_type

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        args.output
        if args.output is not None
        else f"smoke_results_{'twostage' if two_stage else 'singlestage'}_{timestamp}.json"
    )

    global TRIAL_MAX_EPOCHS, RETRAIN_MAX_EPOCHS
    TRIAL_MAX_EPOCHS = args.trial_max_epochs
    RETRAIN_MAX_EPOCHS = args.retrain_max_epochs

    for dataset_name in datasets:
        print(f"\n[SMOKE] Running {dataset_name} with accelerator={accelerator}")
        try:
            if two_stage:
                trainer = _make_two_stage_trainer(accelerator)
                study_s2, clusters, num_clusters = trainer.run_optimization(
                    dataset_name,
                    cut_type=cut_type,
                    n_trials_s1=args.n_trials_s1,
                    n_trials_s2=args.n_trials_s2,
                )
                test_metrics = trainer.test_best_model(
                    study_s2,
                    dataset_name,
                    clusters,
                    num_clusters,
                )

                if isinstance(test_metrics, dict):
                    test_payload = test_metrics
                else:
                    test_payload = {"test_metrics": test_metrics}

                results.append(
                    {
                        "dataset": dataset_name,
                        "status": "ok",
                        "pipeline": "two_stage",
                        "cut_type": cut_type,
                        "accelerator": accelerator,
                        "split_index": SPLIT_INDEX,
                        "split_seed": SPLIT_SEED,
                        "optuna_seed": OPTUNA_SEED,
                        "n_trials_s1": args.n_trials_s1,
                        "n_trials_s2": args.n_trials_s2,
                        "trial_max_epochs": TRIAL_MAX_EPOCHS,
                        "retrain_max_epochs": RETRAIN_MAX_EPOCHS,
                        "best_stage2_val_loss": study_s2.best_value,
                        "best_stage2_params": study_s2.best_trial.params,
                        "num_clusters": num_clusters,
                        "test": test_payload,
                    }
                )
            else:
                trainer = _make_single_stage_trainer(accelerator)
                study = trainer.run_optimization(
                    dataset_name,
                    n_trials=args.n_trials_single,
                )
                test_metrics = trainer.test_best_model(study, dataset_name)
                results.append(
                    {
                        "dataset": dataset_name,
                        "status": "ok",
                        "pipeline": "single_stage",
                        "accelerator": accelerator,
                        "split_index": SPLIT_INDEX,
                        "split_seed": SPLIT_SEED,
                        "optuna_seed": OPTUNA_SEED,
                        "n_trials": args.n_trials_single,
                        "trial_max_epochs": TRIAL_MAX_EPOCHS,
                        "retrain_max_epochs": RETRAIN_MAX_EPOCHS,
                        "best_val_loss": study.best_value,
                        "best_params": study.best_trial.params,
                        "test": {"test_metrics": test_metrics},
                    }
                )
        except Exception as exc:
            results.append(
                {
                    "dataset": dataset_name,
                    "status": "error",
                    "pipeline": "two_stage" if two_stage else "single_stage",
                    "accelerator": accelerator,
                    "error": str(exc),
                }
            )
            print(f"[SMOKE] {dataset_name} failed: {exc}")

        _write_results(out_path, results)

    print("\n[SMOKE] Completed.")
    print(f"[SMOKE] Wrote results to {out_path}")


if __name__ == "__main__":
    run_smoke_test()
