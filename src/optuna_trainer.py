import optuna
from optuna.integration import PyTorchLightningPruningCallback  # TODO
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import os
from loading.LightningGraphLoader import load_datasets
from models.MyModel import MyModel
from loading.DatasetInfo import DatasetInfo
import torch

ACCELERATOR = "cpu"
DEVICES = "auto"


# stack overflow suggestion to fix this callback (as was built with lightning.pytorch and we use pytorch_lightning)
class _OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):  # type: ignore
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _extract_network_info(network, network_name):
    return DatasetInfo(
        network.num_classes,
        network.num_features,
        network_name,
        network.class_weights,
        network.data.data.x.shape[0],
    )


def _objective(trial, network_name):
    # Set the hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.7)
    K = trial.suggest_categorical("K", [4, 8, 10])

    network = load_datasets([network_name])[network_name]
    network_info = _extract_network_info(network, network_name)

    model = MyModel(
        network_info,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        K=K,
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=100, verbose=False, mode="min"
    )

    # Optuna pruning callback
    pruning_callback = _OptunaPruning(trial, monitor="val_loss")

    # Logger
    logger = TensorBoardLogger(
        save_dir=os.getcwd(), name=f"optuna_logs/trial_{trial.number}"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[early_stop_callback, pruning_callback],
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=ACCELERATOR,
        devices=DEVICES,
    )

    # Training the model
    trainer.fit(model, network.data)

    # Final validation loss
    return trainer.callback_metrics["val_loss"].item()


def run_optimization(network_name, n_trials=20):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        lambda trial_num: _objective(trial_num, network_name), n_trials=n_trials
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


def test_best_model(study, network_name):
    # Getting the best hyperparameters
    best_params = study.best_trial.params

    print(best_params)

    network = load_datasets([network_name])[network_name]
    network_info = _extract_network_info(network, network_name)

    # Creating the model with the best hyperparameters
    model = MyModel(
        network_info,
        **best_params,
        # layer_1_size=best_params['layer_1_size'],
        # layer_2_size=best_params['layer_2_size'],
        # learning_rate=best_params["learning_rate"],
        # dropout_rate=best_params['dropout_rate']
    )

    # Creating trainer instance
    trainer = pl.Trainer(max_epochs=10, accelerator=ACCELERATOR, devices=DEVICES)

    # Training the model with the best hyperparameters
    trainer.fit(model, network.data)

    # Testing the model with the test data
    results = trainer.test(model, network.data)
    return results
