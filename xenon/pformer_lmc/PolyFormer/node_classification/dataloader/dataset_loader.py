from pathlib import Path
import sys


class _SingleGraphDataset:
    def __init__(self, data, num_features: int, num_classes: int):
        self._data = data
        self.num_features = num_features
        self.num_classes = num_classes
        self.uses_external_splits = True

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("_SingleGraphDataset only has one graph")
        return self._data

    def __len__(self):
        return 1


_NAME_MAP = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "Pubmed",
    "chameleon_filtered": "chameleon",
    "squirrel_filtered": "squirrel",
    "roman-empire": "Roman-empire",
    "amazon-ratings": "Amazon-ratings",
    "minesweeper": "Minesweeper",
    "tolokers": "Tolokers",
    "questions": "Questions",
    "computers": "computers",
    "photo": "photo",
    "actor": "actor",
    "texas": "texas",
    "cornell": "cornell",
}


def _import_user_dataloader():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import demo_dataloader  # type: ignore

    return demo_dataloader


def DataLoader(name, split_index=1, split_seed=42):
    name = name.lower()
    if name in {"cs", "physics"}:
        raise ValueError(
            "datasets cs/physics are not defined in demo_dataloader.py; "
            "use one of the datasets supported there"
        )

    if name not in _NAME_MAP:
        raise ValueError(f"dataset {name} not supported in dataloader")

    demo_loader = _import_user_dataloader()
    mapped_name = _NAME_MAP[name]
    data, num_features, num_classes = demo_loader._load_single_ds(
        mapped_name,
        split_index=split_index,
        split_seed=split_seed,
    )
    return _SingleGraphDataset(data, num_features, num_classes)
