"""Endpoint-only adapter to the thesis Optuna hook, with persistent study identity."""

from dataclasses import replace
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

import optuna
from optuna.trial import TrialState
import torch
import torch_geometric


_HOOK = Path(__file__).resolve().parents[2] / "src/optuna_trainer.py"
_PARAMETERS = {"learning_rate": "lr", "hidden_dim": "hidden_channels",
               "dropout_rate": "dropout", "depth": "depth", "weight_decay": "weight_decay"}


def _hook():
    spec = importlib.util.spec_from_file_location("_gnn_thesis_optuna", _HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _search(config):
    fixed = {trial_name: getattr(config, field) for trial_name, field in _PARAMETERS.items()
             if field in config.overrides}
    if config.smoke:
        fixed["hidden_dim"] = config.hidden_channels
    return {"version": 1, "common": "thesis suggest_common_parameters", "pruning_report_interval": 10,
            "depth": sorted({2, 3, 4, config.depth}),
            "weight_decay": [0., 5e-5, 5e-4, 1e-3, 5e-3], "fixed": fixed}


def cache_context(graph, metadata, config):
    """Test targets and final-run/curve seeds do not participate in tuning identity."""
    digest = hashlib.sha256()
    labeled = graph.train_mask | graph.val_mask
    tensors = {"x": graph.x, "edge_index": graph.edge_index,
               "train_mask": graph.train_mask, "val_mask": graph.val_mask,
               "train_val_targets": graph.y[labeled]}
    for name, tensor in tensors.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{list(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    local = Path(__file__).resolve().parent
    sources = [_HOOK, *(local / name for name in
                ("tuning.py", "experiment.py", "model.py", "reference_models.py", "paths.py", "presets.py", "data.py"))]
    return {
        "schema_version": 1, "dataset": metadata["name"], "graph_sha256": digest.hexdigest(),
        "num_features": metadata["num_features"], "num_classes": metadata["num_classes"],
        "loader_sha256": metadata["loader_sha256"],
        "fixed_training": {name: getattr(config, name) for name in
                           ("architecture", "preset", "normalization", "residual", "pre_linear", "heads",
                            "epochs", "selection", "tuning_seed", "threads")},
        "initial_profile": {field: getattr(config, field) for field in _PARAMETERS.values()},
        "search": _search(config),
        "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "runtime": {"torch": str(torch.__version__), "torch_geometric": str(torch_geometric.__version__),
                    "optuna": optuna.__version__, "device_type": torch.device(config.device).type,
                    "matmul_precision": torch.get_float32_matmul_precision()},
        "objective": "Maximize calibrated selected-endpoint validation accuracy; no test metrics; fixed tuning seed.",
    }


def tune_endpoints(graph, metadata, config):
    """Reuse a completed budget or extend the matching persistent study."""
    from .experiment import model_configuration, train_endpoint

    context = cache_context(graph, metadata, config)
    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    directory = Path(config.tuning_cache).expanduser().resolve() / metadata["name"] / config.architecture / key
    directory.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{directory / 'study.sqlite3'}"
    study_name = f"endpoint-{key}"
    search = context["search"]
    hook = _hook()

    def objective(trial):
        raw = hook.suggest_common_parameters(trial, fixed=search["fixed"])
        for name in ("depth", "weight_decay"):
            raw[name] = search["fixed"][name] if name in search["fixed"] else trial.suggest_categorical(name, search[name])
        candidate = replace(config, **{_PARAMETERS[name]: value for name, value in raw.items()})

        def report_epoch(epoch, loss, accuracy):
            if epoch % search["pruning_report_interval"] and epoch != candidate.epochs:
                return
            trial.report(accuracy, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Validation accuracy pruned at epoch {epoch}")

        model, result = train_endpoint(graph, model_configuration(metadata, candidate), candidate,
                                       config.tuning_seed, evaluate_test=False, epoch_callback=report_epoch)
        trial.set_user_attr("selected_epoch", result["selected_epoch"])
        trial.set_user_attr("validation_loss", result["metrics"]["val"]["loss"])
        score = result["metrics"]["val"]["accuracy"]
        del model
        return score

    with (directory / "study.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = 0
        try:
            existing = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            existing = None
        if existing is not None:
            for trial in existing.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                existing.tell(trial.number, state=TrialState.FAIL)
            before = sum(trial.state.is_finished() for trial in existing.get_trials(deepcopy=False))
        initial = {name: getattr(config, field) for name, field in _PARAMETERS.items() if name not in search["fixed"]}
        trainer = hook.OptunaTrainer("cpu")  # The custom objective owns its device.
        study = trainer.run_optimization(metadata["name"], n_trials=config.tuning_trials,
                                         objective=objective, storage=storage, study_name=study_name,
                                         direction="maximize", seed=config.tuning_seed, initial_params=initial)
        finished = sum(trial.state.is_finished() for trial in study.get_trials(deepcopy=False))
        raw_best = {**search["fixed"], **study.best_params}
        best = {_PARAMETERS[name]: value for name, value in raw_best.items()}
        info = {"key": key, "cache_hit": finished == before, "new_trials": finished - before,
                "finished_trials": finished, "requested_trials": config.tuning_trials,
                "best_trial": study.best_trial.number, "best_validation_accuracy": study.best_value,
                "best_params": best, "study_name": study_name,
                "study_file": str(directory / "study.sqlite3"),
                "best_parameters_file": str(directory / "best.json"), "context": context}
        temporary = directory / "best.json.tmp"
        temporary.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / "best.json")
    print(f"[{metadata['name']}/{config.architecture}] Optuna {'cache hit' if info['cache_hit'] else 'search complete'}: "
          f"{finished} trials, validation accuracy={study.best_value:.6f}, parameters={best}", flush=True)
    return replace(config, **best), info
