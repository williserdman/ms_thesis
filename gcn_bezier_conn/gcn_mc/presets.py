"""Pinned tunedGNN profiles and explicit local dataset fallbacks."""

from __future__ import annotations

import copy


_COMMIT = "23f9604e8b13a9a6d3faa2f691cd844006979153"
_SOURCE = {
    "repository": "https://github.com/LUOyk1999/tunedGNN",
    "commit": _COMMIT,
    "architecture_source": "upstream/tunedGNN/medium_graph/model.py",
    "profile_source": "upstream/tunedGNN/medium_graph/run_gnn.sh",
    "license": "MIT; upstream/tunedGNN/LICENSE",
    "profile_origin": "pinned_tunedgnn_run_gnn",
    "local_fallback": False,
}

_LOCAL_FALLBACK_SOURCE = {
    "profile_origin": "local_fallback",
    "local_fallback": True,
    "basis": "Previous compact GCN budget, applied to the selected operator.",
    "reason": "The pinned tunedGNN run script has no recipe for this dataset.",
}


def _entry(
    hidden,
    depth,
    dropout,
    normalization,
    residual,
    pre_linear,
    epochs,
    lr,
    weight_decay,
    selection="val_accuracy",
):
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
            "selection": selection,
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
    "questions": {
        "gcn": _entry(512, 10, 0.3, "none", True, True, 1500, 3e-5, 0.0),
        "graphsage": _entry(512, 6, 0.2, "layer", False, True, 1500, 3e-5, 0.0),
        "gat": _entry(512, 3, 0.2, "layer", True, True, 1500, 3e-5, 0.0),
    },
    "computers": {
        "gcn": _entry(512, 3, 0.5, "layer", False, False, 1000, 0.001, 5e-5),
        "graphsage": _entry(64, 4, 0.3, "layer", False, False, 1000, 0.001, 5e-5),
        "gat": _entry(64, 2, 0.5, "layer", False, False, 1000, 0.001, 5e-5),
    },
    "photo": {
        "gcn": _entry(256, 6, 0.5, "layer", True, False, 1000, 0.001, 5e-5),
        "graphsage": _entry(64, 6, 0.2, "layer", True, False, 1000, 0.001, 5e-5),
        "gat": _entry(64, 3, 0.5, "layer", True, False, 1000, 0.001, 5e-5),
    },
    "citeseer": {
        "gcn": _entry(512, 2, 0.5, "none", False, False, 500, 0.001, 0.01),
        "graphsage": _entry(512, 3, 0.2, "none", False, False, 500, 0.001, 0.01),
        "gat": _entry(256, 3, 0.5, "none", True, False, 500, 0.001, 0.01),
    },
    "pubmed": {
        "gcn": _entry(256, 2, 0.7, "none", False, False, 500, 0.005, 5e-4),
        "graphsage": _entry(512, 4, 0.7, "none", False, False, 500, 0.005, 5e-4),
        "gat": _entry(512, 2, 0.5, "none", False, False, 500, 0.01, 5e-4),
    },
    "amazonratings": {
        "gcn": _entry(512, 4, 0.5, "batch", True, False, 2500, 0.001, 0.0),
        "graphsage": _entry(512, 9, 0.5, "batch", True, False, 2500, 0.001, 0.0),
        "gat": _entry(512, 4, 0.5, "batch", True, False, 2500, 0.001, 0.0),
    },
    "minesweeper": {
        "gcn": _entry(64, 12, 0.2, "batch", True, False, 2000, 0.01, 0.0),
        "graphsage": _entry(64, 15, 0.2, "batch", True, False, 2000, 0.01, 0.0),
        "gat": _entry(64, 15, 0.2, "batch", True, False, 2000, 0.01, 0.0),
    },
}

_FALLBACK_DATASETS = {"actor", "texas", "cornell", "tolokers"}
for _dataset in _FALLBACK_DATASETS:
    _PROFILES[_dataset] = {
        architecture: _entry(
            64, 2, 0.5, "none", False, False, 200, 0.01, 5e-4, "val_loss"
        )
        for architecture in ("gcn", "graphsage", "gat")
    }

_DATASET_ALIASES = {
    "amazoncomputer": "computers",
    "amazonphoto": "photo",
}

_UPSTREAM_DATASET_NAMES = {
    "cora": "cora",
    "squirrel": "squirrel",
    "romanempire": "roman-empire",
    "chameleon": "chameleon",
    "questions": "questions",
    "computers": "amazon-computer",
    "photo": "amazon-photo",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "amazonratings": "amazon-ratings",
    "minesweeper": "minesweeper",
}


def reference_profile(dataset: str, architecture: str) -> dict:
    """Return model/training defaults and their pinned or local provenance."""
    dataset_key = dataset.strip().lower().replace("-", "").replace("_", "")
    dataset_key = _DATASET_ALIASES.get(dataset_key, dataset_key)
    if dataset_key not in _PROFILES:
        choices = (
            "Questions, Cora, Roman-empire, computers, photo, Citeseer, Pubmed, "
            "squirrel, chameleon, actor, texas, cornell, Amazon-ratings, "
            "Minesweeper, Tolokers"
        )
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
    if dataset_key in _FALLBACK_DATASETS:
        result["source"] = dict(_LOCAL_FALLBACK_SOURCE)
    else:
        result["source"] = dict(_SOURCE)
        result["source"]["upstream_dataset"] = _UPSTREAM_DATASET_NAMES[dataset_key]
    if architecture_key == "mlp":
        if dataset_key in _FALLBACK_DATASETS:
            result["source"]["profile_adaptation"] = (
                "No upstream dataset or MLP recipe was recovered; this baseline uses the local GCN fallback."
            )
        else:
            result["source"]["profile_adaptation"] = (
                "No upstream MLP recipe was recovered; this baseline uses the tunedGNN GCN profile."
            )
    return result
