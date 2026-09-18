"""Train fixed endpoints, fit a Bézier control, and measure both paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import chain
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from torch.nn import functional as F
import torch_geometric

from .data import load_graph, seed_everything
from .model import GCN
from .paths import clone_parameters, make_control, path_logits, summarize_path


@dataclass
class Config:
    datasets: list[str] = field(default_factory=lambda: ["Cora"])
    pairs: list[tuple[int, int]] = field(default_factory=lambda: [(0, 1), (2, 3), (4, 5)])
    hidden_channels: int = 64
    depth: int = 2
    dropout: float = 0.5
    epochs: int = 200
    curve_epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    curve_lr: float = 0.01
    curve_samples: int = 1
    points: int = 21
    data_seed: int = 0
    curve_seed: int = 10000
    device: str = "cpu"
    threads: int = 1
    thesis_root: str | None = None
    smoke: bool = False


def cpu_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def metrics(logits, graph):
    result = {}
    for split in ("train", "val", "test"):
        mask = getattr(graph, f"{split}_mask")
        loss = F.cross_entropy(logits[mask], graph.y[mask])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {split} loss.")
        result[split] = {
            "loss": float(loss),
            "accuracy": float((logits[mask].argmax(-1) == graph.y[mask]).float().mean()),
        }
    return result


def train_endpoint(graph, model_config, config, seed):
    seed_everything(seed)
    model = GCN(**model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_loss = float("inf")
    best_state = None
    history = []
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss for seed {seed}.")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            val_loss = float(F.cross_entropy(logits[graph.val_mask], graph.y[graph.val_mask]))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = cpu_state(model.state_dict())
        history.append({"epoch": epoch, "train_loss_with_dropout": float(loss.detach()), "val_loss": val_loss})
    if best_state is None:
        raise FloatingPointError("No finite validation checkpoint was produced.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        endpoint_metrics = metrics(model(graph.x, graph.edge_index), graph)
    return model, {"seed": seed, "selected_epoch": best_epoch, "metrics": endpoint_metrics, "history": history}


def fit_curve(model, graph, endpoint_a, endpoint_b, config, seed):
    """Minimize sampled training CE. Only the middle control is optimized.

    Endpoint tensors are detached. Functional calls preserve the derivative of
    the loss with respect to the control. The final control is retained; no
    validation or test labels participate in this optimizer.
    """
    seed_everything(seed)
    control = make_control(endpoint_a, endpoint_b)
    optimizer = torch.optim.Adam(control.values(), lr=config.curve_lr)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    history = []
    model.train()
    for epoch in range(1, config.curve_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        average_loss = 0.0
        for _ in range(config.curve_samples):
            t = float(torch.rand(()))
            logits = path_logits(model, graph, endpoint_a, endpoint_b, t, control)
            loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite curve training loss.")
            (loss / config.curve_samples).backward()
            average_loss += float(loss.detach()) / config.curve_samples
        optimizer.step()
        history.append({"epoch": epoch, "sampled_train_loss_with_dropout": average_loss})
    model.eval()
    return {name: value.detach().clone() for name, value in control.items()}, history


@torch.no_grad()
def evaluate_path(model, graph, endpoint_a, endpoint_b, points, control=None):
    model.eval()
    ts = [i / (points - 1) for i in range(points)]
    by_split = {split: {"loss": [], "accuracy": []} for split in ("train", "val", "test")}
    for t in ts:
        result = metrics(path_logits(model, graph, endpoint_a, endpoint_b, t, control), graph)
        for split, values in result.items():
            for metric, value in values.items():
                by_split[split][metric].append(value)
    for values in by_split.values():
        values.update(summarize_path(ts, values["loss"]))
        values["min_accuracy"] = min(values["accuracy"])
    return {"t": ts, "splits": by_split}


def aggregate_pairs(pairs, methods=("linear", "bezier")):
    summaries = {}
    for method in methods:
        summaries[method] = {}
        for split in ("train", "val", "test"):
            values = [p[method]["splits"][split]["barrier"] for p in pairs]
            summaries[method][split] = {
                "barrier_mean": statistics.mean(values),
                "barrier_std": statistics.pstdev(values),
                "n_pairs": len(values),
            }
    return summaries


def run(config: Config, output: str | Path):
    torch.set_num_threads(config.threads)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or a GPU allocation.")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "paper": "https://arxiv.org/abs/2502.12608v1",
        "scope": "core GCN connectivity procedure; thesis pipeline and explicit implementation defaults",
        "config": asdict(config),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_geometric": str(torch_geometric.__version__), "device": config.device},
        "protocol": {
            "endpoint_selection": "Minimum validation cross entropy over the configured epoch budget.",
            "endpoint_optimizer": "Adam, unweighted train-mask cross entropy, configured weight decay on all parameters.",
            "curve_optimizer": "Adam, mean train-mask cross entropy at uniform random t; dropout enabled; no weight decay; final control retained.",
            "curve_initialization": "Arithmetic midpoint of the fixed endpoints.",
            "barrier": "max_t [L(t) - ((1-t)*L(0) + t*L(1))], sampled on an endpoint-inclusive grid.",
            "loss_reduction": "Mean cross entropy per labeled node. The paper displays a summed loss; absolute barriers depend on reduction.",
            "uncertainty": "Population standard deviation across supplied pairs; not a confidence interval.",
            "paper_gaps": "Training hyperparameters, split details, curve optimizer, and checkpoint selection were not specified in the paper; these settings are implementation choices.",
        },
        "datasets": [],
    }
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    for dataset_name in config.datasets:
        graph, metadata = load_graph(dataset_name, config.thesis_root, config.data_seed)
        dataset_dir = output / metadata["name"]
        dataset_dir.mkdir()
        torch.save({name: getattr(graph, name).cpu().clone() for name in ("train_mask", "val_mask", "test_mask")}, dataset_dir / "split.pt")
        graph = graph.to(config.device)
        model_config = {
            "in_channels": metadata["num_features"], "out_channels": metadata["num_classes"],
            "hidden_channels": config.hidden_channels, "depth": config.depth, "dropout": config.dropout,
        }
        dataset_result = {"data": metadata, "model": model_config, "endpoints": [], "pairs": []}
        states = {}
        for seed in dict.fromkeys(chain.from_iterable(config.pairs)):
            print(f"[{metadata['name']}] endpoint seed={seed}", flush=True)
            model, training = train_endpoint(graph, model_config, config, seed)
            states[seed] = clone_parameters(model)
            checkpoint = f"{metadata['name']}/endpoint_{seed}.pt"
            torch.save({"model_config": model_config, "state_dict": cpu_state(model.state_dict()), "seed": seed, "selected_epoch": training["selected_epoch"], "split_sha256": metadata["split_sha256"]}, output / checkpoint)
            dataset_result["endpoints"].append({**training, "checkpoint": checkpoint})
        for pair_index, (seed_a, seed_b) in enumerate(config.pairs):
            print(f"[{metadata['name']}] curve {seed_a}:{seed_b}", flush=True)
            model = GCN(**model_config).to(config.device)
            endpoint_a, endpoint_b = states[seed_a], states[seed_b]
            curve_seed = config.curve_seed + pair_index
            linear = evaluate_path(model, graph, endpoint_a, endpoint_b, config.points)
            control, history = fit_curve(model, graph, endpoint_a, endpoint_b, config, curve_seed)
            bezier = evaluate_path(model, graph, endpoint_a, endpoint_b, config.points, control)
            checkpoint = f"{metadata['name']}/curve_{seed_a}_{seed_b}.pt"
            torch.save({"model_config": model_config, "control": cpu_state(control), "endpoint_seeds": [seed_a, seed_b], "curve_seed": curve_seed, "split_sha256": metadata["split_sha256"]}, output / checkpoint)
            dataset_result["pairs"].append({"seed_a": seed_a, "seed_b": seed_b, "curve_seed": curve_seed, "checkpoint": checkpoint, "history": history, "linear": linear, "bezier": bezier})
            print(f"[{metadata['name']}] test barrier linear={linear['splits']['test']['barrier']:.6f} bezier={bezier['splits']['test']['barrier']:.6f}", flush=True)
        dataset_result["summary"] = aggregate_pairs(dataset_result["pairs"])
        report["datasets"].append(dataset_result)
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return output / "report.json"
