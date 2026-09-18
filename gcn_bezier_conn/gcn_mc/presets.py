"""Recovered tunedGNN profiles for the four target datasets."""

from __future__ import annotations

import copy


_COMMIT = "23f9604e8b13a9a6d3faa2f691cd844006979153"
_SOURCE = {
    "repository": "https://github.com/LUOyk1999/tunedGNN",
    "commit": _COMMIT,
    "architecture_source": "upstream/tunedGNN/medium_graph/model.py",
    "profile_source": "upstream/tunedGNN/medium_graph/run_gnn.sh",
    "license": "MIT; upstream/tunedGNN/LICENSE",
}


def _entry(hidden, depth, dropout, normalization, residual, pre_linear, epochs, lr, weight_decay):
    return {
        "model": {
            "hidden_channels": hidden,
            "depth": depth,
            "dropout": dropout,
            "normalization": normalization,
            "residual": residual,
            "pre_linear": pre_linear,
            "heads": 1,
        },
        "training": {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "selection": "val_accuracy",
        },
    }


_PROFILES = {
    "cora": {
        "gcn": _entry(512, 3, 0.7, "none", False, False, 500, 0.001, 5e-4),
        "graphsage": _entry(256, 3, 0.7, "none", False, False, 500, 0.001, 5e-4),
        "gat": _entry(512, 3, 0.2, "none", True, False, 500, 0.001, 5e-4),
    },
    "squirrel": {
        "gcn": _entry(256, 4, 0.7, "batch", True, False, 500, 0.01, 5e-4),
        "graphsage": _entry(256, 3, 0.7, "batch", True, False, 500, 0.01, 5e-4),
        "gat": _entry(512, 7, 0.5, "batch", True, False, 500, 0.005, 5e-4),
    },
    "romanempire": {
        "gcn": _entry(512, 9, 0.5, "batch", True, True, 2500, 0.001, 0.0),
        "graphsage": _entry(256, 9, 0.3, "batch", False, True, 2500, 0.001, 0.0),
        "gat": _entry(512, 10, 0.3, "batch", True, True, 2500, 0.001, 0.0),
    },
    "chameleon": {
        "gcn": _entry(512, 5, 0.2, "none", False, False, 200, 0.005, 0.001),
        "graphsage": _entry(256, 4, 0.7, "batch", True, False, 200, 0.01, 0.001),
        "gat": _entry(256, 2, 0.7, "batch", True, False, 200, 0.01, 0.001),
    },
}


def reference_profile(dataset: str, architecture: str) -> dict:
    """Return recovered model/training defaults and their pinned provenance."""
    dataset_key = dataset.strip().lower().replace("-", "").replace("_", "")
    if dataset_key not in _PROFILES:
        choices = "Cora, Squirrel, Roman-Empire, Chameleon"
        raise ValueError(f"Unknown reference dataset {dataset!r}; choose from {choices}")

    architecture_key = architecture.strip().lower()
    if architecture_key == "sage":
        architecture_key = "graphsage"
    profile_architecture = "gcn" if architecture_key == "mlp" else architecture_key
    if profile_architecture not in _PROFILES[dataset_key]:
        raise ValueError(
            f"Unknown reference architecture {architecture!r}; choose gcn, mlp, graphsage, or gat"
        )

    result = copy.deepcopy(_PROFILES[dataset_key][profile_architecture])
    result["source"] = dict(_SOURCE)
    if architecture_key == "mlp":
        result["source"]["profile_adaptation"] = (
            "No upstream MLP recipe was recovered; this baseline uses the tunedGNN GCN profile."
        )
    return result
