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


class OptunaTrainer:
    def __init__(self, accelerator: str, device: int | str | torch.device = "auto"):
        self.accelerator = accelerator

        if accelerator == "cpu":
            self.devices = "auto"
        else:
            self.devices = [device]

    def _objective(self, trial, network_name):
        # Set the hyperparameters to optimize
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.7)
        K = trial.suggest_categorical("K", [4, 8, 10])
        num_iters = trial.suggest_int("num_iters", 1, 3)
        num_clusters = trial.suggest_int("num_clusters", 2, 10)
        num_heads_main = trial.suggest_categorical("num_heads_main", [2, 4, 8, 16])
        loss_lambda = trial.suggest_float("loss_lambda", 0, 1)
        multi = trial.suggest_int("multi", 1, 4)

        network = load_datasets([network_name])[network_name]
        network_info = _extract_network_info(network, network_name)

        model = MyModel(
            network_info,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            K=K,
            num_iters=num_iters,
            num_clusters=num_clusters,
            num_heads_main=num_heads_main,
            loss_lambda=loss_lambda,
            multi=multi,
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
            accelerator=self.accelerator,
            devices=self.devices,  # type: ignore
        )

        # Training the model
        trainer.fit(
            model=model,
            train_dataloaders=network.data.train_dataloader(),
            val_dataloaders=network.data.val_dataloader(),
        )

        # Final validation loss
        return trainer.callback_metrics["val_loss"].item()

    def run_optimization(self, network_name, n_trials=20):
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(
            lambda trial_num: self._objective(trial_num, network_name),
            n_trials=n_trials,
        )

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study

    def test_best_model(self, study, network_name):
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
        trainer = pl.Trainer(
            max_epochs=2000, accelerator=self.accelerator, devices=self.devices  # type: ignore
        )

        # Training the model with the best hyperparameters
        trainer.fit(
            model=model,
            train_dataloaders=network.data.train_dataloader(),
            val_dataloaders=network.data.val_dataloader(),
        )

        # Testing the model with the test data
        results = trainer.test(model=model, dataloaders=network.data.test_dataloader())
        return results
