from pathlib import Path
from typing import Any

import torch


SUPPORTED_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "computers",
    "actor",
    "chameleon_filtered",
    "squirrel_filtered",
]
DATASET_ALIASES = {
    "chameleon": "chameleon_filtered",
    "squirrel": "squirrel_filtered",
}


def canonicalize_dataset_name(dataset: str) -> str:
    ds = dataset.lower()
    return DATASET_ALIASES.get(ds, ds)


def _resolve_run_checkpoint(dataset_dir: Path, run: int, base: str = "mono", model: str = "polyformer") -> Path:
    model_name = model.lower()
    if model_name == "gcn":
        pattern = f"gcn_run{run}_seed*.pt"
    else:
        pattern = f"{base}_run{run}_seed*.pt"
    matches = sorted(dataset_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for pattern '{pattern}' in {dataset_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one checkpoint for run {run}, found {len(matches)} in {dataset_dir}: {matches}"
        )
    return matches[0]


def resolve_checkpoint_pair(
    repo_root: Path,
    dataset: str,
    base: str = "mono",
    model: str = "polyformer",
    run_a: int = 1,
    run_b: int = 2,
    checkpoint_a: str | None = None,
    checkpoint_b: str | None = None,
) -> tuple[Path, Path]:
    ds = canonicalize_dataset_name(dataset)
    if ds not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {SUPPORTED_DATASETS}")

    if checkpoint_a is not None and checkpoint_b is not None:
        return Path(checkpoint_a).expanduser().resolve(), Path(checkpoint_b).expanduser().resolve()

    dataset_dir = repo_root / "PolyFormer" / "node_classification" / "saved_models" / ds
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {dataset_dir}")

    path_a = (
        _resolve_run_checkpoint(dataset_dir, run=run_a, base=base, model=model)
        if checkpoint_a is None
        else Path(checkpoint_a).expanduser().resolve()
    )
    path_b = (
        _resolve_run_checkpoint(dataset_dir, run=run_b, base=base, model=model)
        if checkpoint_b is None
        else Path(checkpoint_b).expanduser().resolve()
    )
    return path_a, path_b


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _normalize_model_name(name: str | None) -> str:
    if not name:
        return "polyformer"
    return name.lower()


def validate_checkpoint_pair(
    ckpt_a: dict[str, Any],
    ckpt_b: dict[str, Any],
    expected_dataset: str,
    expected_base: str,
    expected_model: str = "polyformer",
) -> None:
    dataset_a = str(ckpt_a.get("dataset", "")).lower()
    dataset_b = str(ckpt_b.get("dataset", "")).lower()
    base_a = str(ckpt_a.get("base", "")).lower()
    base_b = str(ckpt_b.get("base", "")).lower()
    model_a = _normalize_model_name(ckpt_a.get("model_name", None))
    model_b = _normalize_model_name(ckpt_b.get("model_name", None))

    if dataset_a != dataset_b:
        raise ValueError(f"Checkpoint datasets differ: '{dataset_a}' vs '{dataset_b}'")
    if base_a != base_b:
        raise ValueError(f"Checkpoint bases differ: '{base_a}' vs '{base_b}'")
    if model_a != model_b:
        raise ValueError(f"Checkpoint model_name differs: '{model_a}' vs '{model_b}'")
    if dataset_a != expected_dataset.lower():
        raise ValueError(f"Requested dataset '{expected_dataset}' but checkpoint has '{dataset_a}'")
    if expected_model.lower() == "polyformer" and base_a != expected_base.lower():
        raise ValueError(f"Requested base '{expected_base}' but checkpoint has '{base_a}'")
    if model_a != expected_model.lower():
        raise ValueError(f"Requested model '{expected_model}' but checkpoint has '{model_a}'")

    shape_keys = ["num_features", "num_classes"]
    for key in shape_keys:
        if key in ckpt_a and key in ckpt_b and ckpt_a[key] != ckpt_b[key]:
            raise ValueError(f"Checkpoint mismatch for {key}: {ckpt_a[key]} vs {ckpt_b[key]}")
