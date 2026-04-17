from pathlib import Path
import sys


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
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import demo_dataloader  # type: ignore

    return demo_dataloader


def load_single_graph(name: str, split_index: int = 0, split_seed: int = 42):
    ds_name = name.lower()
    if ds_name not in _NAME_MAP:
        raise ValueError(f"dataset {name} not supported")

    demo_loader = _import_user_dataloader()
    data, num_features, num_classes = demo_loader._load_single_ds(  # noqa: SLF001
        _NAME_MAP[ds_name], split_index=split_index, split_seed=split_seed
    )
    return data, int(num_features), int(num_classes)
