"""Two-stage Optuna trainer for the cluster-conditioned GArnoldi pipeline.

Stage 1 — Optimise cluster assignments (MinCut or MaxCut) jointly with a
          lightweight GCN classifier.
Stage 2 — Freeze clusters.  Optimise per-cluster GArnoldi polynomial filters,
          each initialised with a different spectral target function.
"""

import os

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.loading.DatasetInfo import DatasetInfo
from src.loading.LightningGraphLoader import load_datasets
from src.models.MyClusterModel import MyClusterStage1, MyClusterFilterBank


# Stack-overflow fix: bridge optuna's lightning.pytorch callback to pytorch_lightning
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


class TwoStageTrainer:
    """Orchestrates the two-stage hyperparameter search and final evaluation."""

    def __init__(
        self,
        accelerator: str,
        device: int | str | torch.device = "auto",
    ):
        self.accelerator = accelerator
        if accelerator == "cpu":
            self.devices = "auto"
        else:
            self.devices = [device]

    # ===================================================================
    # Stage 1
    # ===================================================================

    def _stage1_objective(self, trial, network_name, cut_type):
        learning_rate = trial.suggest_float("s1_learning_rate", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("s1_hidden_dim", [32, 64, 128])
        dropout_rate = trial.suggest_float("s1_dropout_rate", 0.0, 0.7)
        num_clusters = trial.suggest_int("s1_num_clusters", 2, 8)
        loss_lambda = trial.suggest_float("s1_loss_lambda", 0.01, 1.0)

        network = load_datasets([network_name])[network_name]
        network_info = _extract_network_info(network, network_name)

        model = MyClusterStage1(
            network_info,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            num_clusters=num_clusters,
            loss_lambda=loss_lambda,
            cut_type=cut_type,
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=50, verbose=False, mode="min"
        )
        pruning = _OptunaPruning(trial, monitor="val_loss")
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name=f"two_stage_logs/s1_trial_{trial.number}",
        )

        trainer = pl.Trainer(
            max_epochs=200,
            callbacks=[early_stop, pruning],
            logger=logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator=self.accelerator,
            devices=self.devices,  # type: ignore
        )

        trainer.fit(
            model=model,
            train_dataloaders=network.data.train_dataloader(),
            val_dataloaders=network.data.val_dataloader(),
        )

        return trainer.callback_metrics["val_loss"].item()

    # ------------------------------------------------------------------

    def _train_stage1_best(self, network_name, cut_type, best_params):
        """Re-train Stage 1 with the best hyper-parameters and return the
        trained Lightning model (so we can extract clusters)."""

        network = load_datasets([network_name])[network_name]
        network_info = _extract_network_info(network, network_name)

        # Strip the ``s1_`` prefix that Optuna added
        params = {k.removeprefix("s1_"): v for k, v in best_params.items()}

        model = MyClusterStage1(
            network_info,
            cut_type=cut_type,
            **params,
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=100, verbose=False, mode="min"
        )
        logger = TensorBoardLogger(save_dir=os.getcwd(), name="two_stage_logs/s1_best")

        trainer = pl.Trainer(
            max_epochs=500,
            callbacks=[early_stop],
            logger=logger,
            accelerator=self.accelerator,
            devices=self.devices,  # type: ignore
        )

        trainer.fit(
            model=model,
            train_dataloaders=network.data.train_dataloader(),
            val_dataloaders=network.data.val_dataloader(),
        )

        return model

    # ===================================================================
    # Stage 2
    # ===================================================================

    def _stage2_objective(self, trial, network_name, pretrained_clusters, num_clusters):
        learning_rate = trial.suggest_float("s2_learning_rate", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("s2_hidden_dim", [32, 64, 128])
        dropout_rate = trial.suggest_float("s2_dropout_rate", 0.0, 0.7)
        K = trial.suggest_categorical("s2_K", [4, 8, 10])
        Init = trial.suggest_categorical("s2_Init", ["Chebyshev", "Legendre", "Jacobi"])
        homophily = trial.suggest_categorical("s2_homophily", [True, False])

        network = load_datasets([network_name])[network_name]
        network_info = _extract_network_info(network, network_name)

        model = MyClusterFilterBank(
            network_info,
            learning_rate=learning_rate,
            pretrained_clusters=pretrained_clusters,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            K=K,
            num_clusters=num_clusters,
            Init=Init,
            homophily=homophily,
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=100, verbose=False, mode="min"
        )
        pruning = _OptunaPruning(trial, monitor="val_loss")
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name=f"two_stage_logs/s2_trial_{trial.number}",
        )

        trainer = pl.Trainer(
            max_epochs=200,
            callbacks=[early_stop, pruning],
            logger=logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator=self.accelerator,
            devices=self.devices,  # type: ignore
        )

        trainer.fit(
            model=model,
            train_dataloaders=network.data.train_dataloader(),
            val_dataloaders=network.data.val_dataloader(),
        )

        return trainer.callback_metrics["val_loss"].item()

    # ===================================================================
    # Public API
    # ===================================================================

    def run_optimization(
        self,
        network_name: str,
        cut_type: str = "mincut",
        n_trials_s1: int = 10,
        n_trials_s2: int = 15,
    ):
        """Run the full two-stage optimisation.

        Returns
        -------
        study_s2 : optuna.Study
            The Stage 2 Optuna study.
        pretrained_clusters : torch.Tensor
            The frozen cluster logits extracted from the best Stage 1 model.
        num_clusters : int
            Number of clusters chosen by Stage 1.
        """
        # ---- Stage 1 ----
        print(f"\n{'=' * 60}")
        print(f"STAGE 1: Learning clusters ({cut_type}) for {network_name}")
        print(f"{'=' * 60}")

        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
        study_s1 = optuna.create_study(direction="minimize", pruner=pruner)
        study_s1.optimize(
            lambda trial: self._stage1_objective(trial, network_name, cut_type),
            n_trials=n_trials_s1,
        )

        best_s1 = study_s1.best_trial
        print(f"\nStage 1 best val_loss : {best_s1.value:.4f}")
        print(f"Stage 1 best params  : {best_s1.params}")

        num_clusters = best_s1.params["s1_num_clusters"]

        # Re-train with the best params to extract cluster logits
        model_s1 = self._train_stage1_best(network_name, cut_type, best_s1.params)
        pretrained_clusters = model_s1.get_cluster_logits()
        print(f"Extracted cluster logits: shape {pretrained_clusters.shape}")

        # ---- Stage 2 ----
        print(f"\n{'=' * 60}")
        print(f"STAGE 2: Training per-cluster GArnoldi filters for {network_name}")
        print(f"{'=' * 60}")

        pruner2 = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
        study_s2 = optuna.create_study(direction="minimize", pruner=pruner2)
        study_s2.optimize(
            lambda trial: self._stage2_objective(
                trial, network_name, pretrained_clusters, num_clusters
            ),
            n_trials=n_trials_s2,
        )

        best_s2 = study_s2.best_trial
        print(f"\nStage 2 best val_loss : {best_s2.value:.4f}")
        print(f"Stage 2 best params  : {best_s2.params}")

        return study_s2, pretrained_clusters, num_clusters

    # ------------------------------------------------------------------

    def test_best_model(
        self,
        study_s2,
        network_name: str,
        pretrained_clusters: torch.Tensor,
        num_clusters: int,
    ):
        """Re-train Stage 2 with the best hyper-parameters, then test."""

        best_params = study_s2.best_trial.params
        params = {k.removeprefix("s2_"): v for k, v in best_params.items()}

        network = load_datasets([network_name])[network_name]
        network_info = _extract_network_info(network, network_name)

        model = MyClusterFilterBank(
            network_info,
            pretrained_clusters=pretrained_clusters,
            num_clusters=num_clusters,
            **params,
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=100, verbose=False, mode="min"
        )
        logger = TensorBoardLogger(save_dir=os.getcwd(), name="two_stage_logs/s2_best")

        trainer = pl.Trainer(
            max_epochs=2000,
            callbacks=[early_stop],
            logger=logger,
            accelerator=self.accelerator,
            devices=self.devices,  # type: ignore
        )

        trainer.fit(
            model=model,
            train_dataloaders=network.data.train_dataloader(),
            val_dataloaders=network.data.val_dataloader(),
        )

        # Inspect learned filter coefficients
        print("\nLearned filter coefficients:")
        for name, coeffs in model.get_filter_coefficients().items():
            print(f"  {name}: {[f'{c:.4f}' for c in coeffs]}")

        results = trainer.test(model=model, dataloaders=network.data.test_dataloader())
        return results
