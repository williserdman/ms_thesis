"""Thin adapter to the existing thesis loader; no dataset preprocessing here."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(name: str, thesis_root: str | Path | None = None, data_seed: int = 0):
    """Return ``(Data, metadata)`` from the thesis loader's existing split.

    The graph is loaded once before endpoint seeds are set. ``data_seed`` only
    affects datasets for which the thesis loader generates random masks.
    """
    root = Path(thesis_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    loader_path = root / "src/loading/LightningGraphLoader.py"
    if not loader_path.is_file():
        raise FileNotFoundError(f"Thesis loader missing: {loader_path}. Set --thesis-root.")
    source_dir = str(root / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    loader = importlib.import_module("loading.LightningGraphLoader")
    if Path(loader.__file__).resolve() != loader_path:
        raise RuntimeError("A loader from another thesis checkout is already imported.")
    canonical = {n.lower(): n for n in loader.ALL_DATASETS}.get(name.strip().lower())
    if canonical is None:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {loader.ALL_DATASETS}")
    seed_everything(data_seed)
    network = loader.load_datasets([canonical])[canonical]
    graph = network.data.data.clone()
    graph.y = graph.y.reshape(-1).long()
    masks = {}
    fingerprint = hashlib.sha256()
    for split in ("train", "val", "test"):
        mask = getattr(graph, f"{split}_mask")
        if mask.dtype != torch.bool or mask.shape != graph.y.shape or not mask.any():
            raise ValueError(f"Thesis loader returned invalid {split} mask for {canonical}.")
        masks[split] = mask
        fingerprint.update(mask.cpu().numpy().tobytes())
    if any((masks[a] & masks[b]).any() for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Train, validation, and test masks overlap.")
    if graph.y.min() < 0 or graph.y.max() >= network.num_classes:
        raise ValueError("Labels must be class indices in [0, num_classes).")
    metadata = {
        "name": canonical,
        "loader": str(loader_path),
        "loader_sha256": hashlib.sha256(loader_path.read_bytes()).hexdigest(),
        "data_root": str(loader.DATA_ROOT),
        "data_seed": data_seed,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_features": int(network.num_features),
        "num_classes": int(network.num_classes),
        "split_counts": {key: int(mask.sum()) for key, mask in masks.items()},
        "split_sha256": fingerprint.hexdigest(),
        "split_policy": "Unmodified thesis loader masks; public Planetoid split, fixed index 1 where supported, seeded random Amazon splits.",
        "preprocessing": "Unmodified thesis loader features and edge_index; GCNConv adds self-loops and symmetric normalization.",
        "class_weighting": "Unweighted cross entropy; loader class weights are not used.",
    }
    return graph, metadata
