import os

import torch
from torch_geometric.datasets import (
    Planetoid,
    WikipediaNetwork,
    Amazon,
    Actor,
    WebKB,
    HeterophilousGraphDataset,
    Twitch,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader
import collections
import torch_geometric
from pytorch_lightning import LightningDataModule
import numpy as np
import torch_geometric.data as tg_data
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, cast

ALL_DATASETS = [
    "Questions",
    "Cora",
    "Roman-empire",
    "computers",
    "photo",
    "Citeseer",
    "Pubmed",
    "squirrel",
    "chameleon",
    "actor",
    "texas",
    "cornell",
    "Amazon-ratings",
    "Minesweeper",
    "Tolokers",
]


def _load_filtered_dataset(path):
    # Load the .npz file
    data = np.load(path)
    num_nodes, nf = data["node_features"].shape
    nc = len(np.unique(data["node_labels"]))

    # Convert to PyTorch tensors
    x = torch.tensor(data["node_features"], dtype=torch.float)
    y = torch.tensor(data["node_labels"], dtype=torch.long)
    edge_index = torch.tensor(data["edges"], dtype=torch.long).t().contiguous()

    # Load the 10 fixed splits (masks)
    # The file usually contains 'train_masks', 'val_masks', 'test_masks'
    # Shape: [num_nodes, 10]
    train_masks = torch.tensor(data["train_masks"], dtype=torch.bool)
    val_masks = torch.tensor(data["val_masks"], dtype=torch.bool)
    test_masks = torch.tensor(data["test_masks"], dtype=torch.bool)

    return (
        tg_data.Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_masks,
            val_mask=val_masks,
            test_mask=test_masks,
        ),
        nf,
        nc,
    )


class LightningGraph:
    def __init__(self, data, num_features, num_classes, class_weights):
        self.data = data
        self.num_features = num_features
        self.num_classes = num_classes
        self.class_weights = class_weights


def _create_ds_splits(name, data: tg_data.Data, train_split, val_split, test_split):
    # Create new train/val/test masks based on provided splits

    num_nodes = data.y.size(0)  # type: ignore
    indices = torch.randperm(num_nodes)
    train_end = int(train_split * num_nodes)
    val_end = train_end + int(val_split * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    """ print(
        f"{name}: train={int(data.train_mask.sum())}, val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())}"
    ) """
    return data


@dataclass
class SimpleDatasetWrapper:
    def __init__(self):
        self.items = []
        self.num_features: Optional[int] = None
        self.num_classes: Optional[int] = None

    def __getitem__(self, idx):
        return self.items[idx]

    def append(self, item):
        self.items.append(item)

    def __len__(self):
        return len(self.items)


def _load_single_ds(name: str):
    name = name.strip()
    tfm = (
        NormalizeFeatures()
    )  # creating instance of NormalizeFeatures to pass into transform (each row sums to one)

    if name in {"Cora", "Citeseer", "Pubmed"}:
        ds = Planetoid(
            root=os.path.join("data", name), name=name, transform=tfm
        )  # initialize planetoid dataset object, will download or use downloaded copy

    elif name in {"chameleon", "squirrel"}:
        data, nf, nc = _load_filtered_dataset(
            Path(f"data/{name}/{name}_filtered_directed.npz")
        )

        ds = SimpleDatasetWrapper()

        ds.append(data)
        ds.num_features = nf
        ds.num_classes = nc

    elif name in {"computers", "photo"}:
        ds = Amazon(root=os.path.join("data", name), name=name, transform=tfm)
        data = cast(tg_data.Data, ds[0])
        nf = ds.num_features
        nc = ds.num_classes
        ds = SimpleDatasetWrapper()
        for _ in range(10):
            split_data = _create_ds_splits(name, data, 0.6, 0.2, 0.2)
            ds.append(split_data)
        ds.num_features, ds.num_classes = nf, nc

    elif name in {"actor"}:
        ds = Actor(root=os.path.join("data", name), transform=tfm)

    elif name in {"texas", "cornell"}:
        ds = WebKB(root=os.path.join("data", name), name=name, transform=tfm)

    elif name in {
        "Roman-empire",
        "Amazon-ratings",
        "Minesweeper",
        "Tolokers",
        "Questions",
    }:
        ds = HeterophilousGraphDataset(
            root=os.path.join("data", name), name=name, transform=tfm
        )

    else:
        print(f"dataset {name} not found, continuing")
        return None

    SPLIT_INDEX = 1
    data = ds[0]

    if name in {
        "Roman-empire",
        "actor",
        "texas",
        "cornell",
        "Amazon-ratings",
        "Minesweeper",
        "Tolokers",
        "Questions",
    }:
        data.train_mask = data.train_mask[:, SPLIT_INDEX]  # type: ignore
        data.val_mask = data.val_mask[:, SPLIT_INDEX]  # type: ignore
        data.test_mask = data.test_mask[:, SPLIT_INDEX]  # type: ignore

    elif name in {
        "squirrel",
        "chameleon",
    }:
        data.train_mask = data.train_mask[SPLIT_INDEX, :]  # type: ignore
        data.val_mask = data.val_mask[SPLIT_INDEX, :]  # type: ignore
        data.test_mask = data.test_mask[SPLIT_INDEX, :]  # type: ignore

    elif name in {"computers", "photo"}:
        data = ds[SPLIT_INDEX]

    return (
        data,
        ds.num_features,
        ds.num_classes,
    )


def load_datasets(datasets: list[str]) -> dict[str, LightningGraph]:

    # we want to grab a "Data" object (torch geometric concept) for each dataset. a data object is one graph. some of the datasets have more than one graph
    out = collections.defaultdict(None)
    for n in datasets:
        ds, nf, nc = _load_single_ds(n)  # type: ignore
        # compute class frequencies from labels
        y = ds.y  # type: ignore
        if y.dim() > 1:
            y = y.view(-1)
        y = y.long()
        counts = torch.bincount(y, minlength=nc).float()  # type: ignore
        # avoid divide-by-zero for classes with no samples
        counts = counts + 1e-8
        class_frequencies = counts / counts.sum()
        # Inverse frequencies (or other weighting scheme)
        class_weights = 1.0 / class_frequencies
        # Normalize or scale if desired, e.g., to sum to 1
        class_weights = class_weights / class_weights.sum()

        # Ensure the weight tensor is of type float
        class_weights = class_weights.float()
        t = tg_data.lightning.LightningNodeData(ds, loader="full")
        out[n] = LightningGraph(t, nf, nc, torch.tensor(class_weights))
        # print(out[n].num_features)
    return out


if __name__ == "__main__":
    load_datasets(ALL_DATASETS)
