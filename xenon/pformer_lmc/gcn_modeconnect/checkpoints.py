from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


SUPPORTED_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "computers",
    "photo",
    "actor",
    "texas",
    "cornell",
    "chameleon_filtered",
    "squirrel_filtered",
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "tolokers",
    "questions",
]


def resolve_checkpoint_pair(
    repo_root: Path,
    dataset: str,
    condition_id: str,
    run_a: int,
    run_b: int,
    checkpoint_a: str | None = None,
    checkpoint_b: str | None = None,
) -> tuple[Path, Path]:
    ds = dataset.lower()
    if ds not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    if checkpoint_a is not None and checkpoint_b is not None:
        return Path(checkpoint_a).expanduser().resolve(), Path(checkpoint_b).expanduser().resolve()

    base_dir = repo_root / "gcn_modeconnect" / "checkpoints" / ds / condition_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {base_dir}")

    def _find(run: int) -> Path:
        pattern = f"gcn_run{run}_seed*.pt"
        matches = sorted(base_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No checkpoint for pattern '{pattern}' in {base_dir}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple checkpoints for run {run}: {matches}")
        return matches[0]

    path_a = _find(run_a) if checkpoint_a is None else Path(checkpoint_a).expanduser().resolve()
    path_b = _find(run_b) if checkpoint_b is None else Path(checkpoint_b).expanduser().resolve()
    return path_a, path_b


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def validate_checkpoint_pair(ckpt_a: dict[str, Any], ckpt_b: dict[str, Any], dataset: str, condition_id: str) -> None:
    if str(ckpt_a.get("model", "")).lower() != "gcn":
        raise ValueError("checkpoint A model is not gcn")
    if str(ckpt_b.get("model", "")).lower() != "gcn":
        raise ValueError("checkpoint B model is not gcn")

    ds_a = str(ckpt_a.get("dataset", "")).lower()
    ds_b = str(ckpt_b.get("dataset", "")).lower()
    if ds_a != ds_b or ds_a != dataset.lower():
        raise ValueError(f"Dataset mismatch: A={ds_a}, B={ds_b}, expected={dataset.lower()}")

    c_a = str(ckpt_a.get("condition_id", ""))
    c_b = str(ckpt_b.get("condition_id", ""))
    if c_a != c_b or c_a != condition_id:
        raise ValueError(f"Condition mismatch: A={c_a}, B={c_b}, expected={condition_id}")

    for key in ("num_features", "num_classes"):
        if int(ckpt_a.get(key, -1)) != int(ckpt_b.get(key, -1)):
            raise ValueError(f"Checkpoint mismatch for {key}")
