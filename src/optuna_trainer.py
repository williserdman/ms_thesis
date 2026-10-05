import optuna
import os
import torch


def suggest_common_parameters(trial, *, fixed=None):
    """Shared thesis search dimensions; explicit fixed values are not sampled."""
    fixed = fixed or {}
    return {
        "learning_rate": fixed["learning_rate"] if "learning_rate" in fixed else trial.suggest_float("learning_rate", 1e-4, 1e-2),
        "hidden_dim": fixed["hidden_dim"] if "hidden_dim" in fixed else trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256, 512]),
        "dropout_rate": fixed["dropout_rate"] if "dropout_rate" in fixed else trial.suggest_float("dropout_rate", 0.0, 0.7),
    }


def _extract_network_info(network, network_name):
    from loading.DatasetInfo import DatasetInfo
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
        from optuna.integration import PyTorchLightningPruningCallback
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
        import pytorch_lightning as pl
        from loading.LightningGraphLoader import load_datasets
        from models.MyModel import MyModel

        class _OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
            pass

        # Set the hyperparameters to optimize
        common = suggest_common_parameters(trial)
        K = trial.suggest_categorical("K", [4, 8, 10])
        multi = trial.suggest_int("multi", 1, 4)

        network = load_datasets([network_name])[network_name]
        network_info = _extract_network_info(network, network_name)

        model = MyModel(
            network_info,
            **common,
            K=K,
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

    def run_optimization(self, network_name, n_trials=20, *, objective=None,
                         storage=None, study_name=None, direction="minimize",
                         seed=None, initial_params=None):
        """Run a fresh search, or finish the requested total persistent budget."""
        if n_trials < 1:
            raise ValueError("n_trials must be positive")
        if storage is not None and study_name is None:
            raise ValueError("Persistent studies require study_name")
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction=direction, pruner=pruner,
                                   sampler=optuna.samplers.TPESampler(seed=seed),
                                   storage=storage, study_name=study_name,
                                   load_if_exists=storage is not None)
        finished = sum(trial.state.is_finished() for trial in study.get_trials(deepcopy=False))
        if initial_params is not None and not study.trials:
            study.enqueue_trial(initial_params)
        remaining = max(0, n_trials - finished)
        if remaining:
            study.optimize(objective or (lambda trial: self._objective(trial, network_name)),
                           n_trials=remaining)

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study

    def test_best_model(self, study, network_name):
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
        import pytorch_lightning as pl
        from loading.LightningGraphLoader import load_datasets
        from models.MyModel import MyModel
        # Getting the best hyperparameters
        best_params = study.best_trial.params

        print(best_params)

        network = load_datasets([network_name])[network_name]
        network_info = _extract_network_info(network, network_name)

        logger = TensorBoardLogger(
            save_dir=os.getcwd(), name=f"optuna_logs/best_params"
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=100, verbose=False, mode="min"
        )

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
            max_epochs=2000,
            accelerator=self.accelerator,
            devices=self.devices,  # type: ignore
            logger=logger,
            callbacks=[early_stop_callback],
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
