"""Compare alignment and graph REPAIR using a previously saved GCN run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import torch
import torch_geometric

from .data import load_graph
from .experiment import aggregate_pairs, cpu_state, metrics
from .model import GCN
from .paths import clone_parameters, interpolate, summarize_path
from .repair_adapter import align_gcn, hidden_statistics, repair_gcn


METHODS = ("linear", "aligned", "repaired", "bezier")


def _load_checkpoint(source_dir, relative, model_config, split_hash, device):
    checkpoint = torch.load(source_dir / relative, map_location=device, weights_only=True)
    if checkpoint["model_config"] != model_config:
        raise ValueError(f"Model configuration mismatch in {relative}.")
    if checkpoint["split_sha256"] != split_hash:
        raise ValueError(f"Split identity mismatch in {relative}.")
    return checkpoint


def _materialize(template, a, b, t, control=None):
    model = copy.deepcopy(template)
    model.load_state_dict(interpolate(a, b, t, control))
    return model.eval()


def _empty_path(ts):
    return {"t": list(ts), "splits": {split: {"loss": [], "accuracy": []} for split in ("train", "val", "test")}}


def run_repair(source, output, *, thesis_root=None, device="cpu", threads=1):
    """Reevaluate the source grid without training endpoints or Bézier controls."""
    torch.set_num_threads(threads)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    source = Path(source).expanduser().resolve()
    source_bytes = source.read_bytes()
    baseline = json.loads(source_bytes)
    if baseline.get("schema_version") != 1 or baseline.get("experiment") == "gnn_repair":
        raise ValueError("--source must be an original schema-v1 GCN connectivity report.")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = copy.deepcopy(baseline)
    report.update({
        "experiment": "gnn_repair",
        "methods": list(METHODS),
        "scope": "Saved GCN endpoint comparison: channel alignment and sequential REPAIR on linear paths; existing Bézier comparator.",
        "source_report": str(source),
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_environment": baseline["environment"],
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_geometric": str(torch_geometric.__version__), "device": device},
        "datasets": [],
    })
    report["config"].update({
        "device": device,
        "threads": threads,
        "thesis_root": str(Path(thesis_root or Path(baseline["datasets"][0]["data"]["loader"]).parents[2]).expanduser().resolve()),
    })
    report["protocol"]["repair"] = {
        "methods": list(METHODS),
        "training": "No retraining or curve fitting; original endpoint and control checkpoints reused.",
        "alignment": "Hungarian matching of training-node post-ReLU correlations; hidden channels only; full-graph logit invariance checked.",
        "calibration": "Training-node population preactivation moments after each hidden GCNConv, before ReLU; sequential correction; dropout disabled.",
        "transductive": "Full-graph message passing remains active; only train-mask rows enter matching and moment estimation; no labels used.",
        "targets": "Weighted endpoint means and standard deviations, not weighted variances.",
        "epsilon": 1e-5,
        "endpoints": "Exact endpoint copies at t=0 and t=1; classifier excluded from correction.",
        "path_interpretation": "The repaired path is not a straight weight-space segment. Bézier receives no REPAIR.",
        "variance_ratio": "Mean training-node channel variance divided by the t-weighted mean endpoint channel variances. Null if denominator is zero.",
    }
    import repair.core
    report["repair_implementation"] = {
        "core": str(Path(repair.core.__file__).resolve()),
        "core_sha256": hashlib.sha256(Path(repair.core.__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).with_name("repair_adapter.py").read_bytes()).hexdigest(),
    }
    for original in baseline["datasets"]:
        original_data = original["data"]
        recorded_root = Path(original_data["loader"]).parents[2]
        graph, metadata = load_graph(original_data["name"], thesis_root or recorded_root, original_data["data_seed"])
        for key in ("split_sha256", "loader_sha256", "num_nodes", "num_edges", "num_features", "num_classes"):
            if metadata[key] != original_data[key]:
                raise ValueError(f"Source dataset {key} changed. Use the original thesis loader and data.")
        dataset_dir = output / metadata["name"]
        dataset_dir.mkdir()
        original_masks = torch.load(source.parent / metadata["name"] / "split.pt", map_location="cpu", weights_only=True)
        for name, mask in original_masks.items():
            if not torch.equal(mask, getattr(graph, name)):
                raise ValueError(f"Saved {name} differs from the current loader.")
        torch.save(original_masks, dataset_dir / "split.pt")
        graph = graph.to(device)
        dataset = {"data": metadata, "model": original["model"], "endpoints": [], "pairs": []}
        endpoints = {}
        for record in original["endpoints"]:
            checkpoint = _load_checkpoint(source.parent, record["checkpoint"], original["model"], metadata["split_sha256"], device)
            model = GCN(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                replayed = metrics(model(graph.x, graph.edge_index), graph)
            for split in ("train", "val", "test"):
                if abs(replayed[split]["loss"] - record["metrics"][split]["loss"]) > 2e-5:
                    raise ValueError("Endpoint loss does not reproduce the source report; check source graph features and edges.")
            endpoints[record["seed"]] = model
            copied_record = copy.deepcopy(record)
            copied_record["metrics"] = replayed
            copied_record["checkpoint"] = f"{metadata['name']}/endpoint_{record['seed']}.pt"
            shutil.copy2(source.parent / record["checkpoint"], output / copied_record["checkpoint"])
            dataset["endpoints"].append(copied_record)
        for original_pair in original["pairs"]:
            sa, sb = original_pair["seed_a"], original_pair["seed_b"]
            print(f"[{metadata['name']}] REPAIR comparison {sa}:{sb}", flush=True)
            a, b = endpoints[sa], endpoints[sb]
            aligned, alignment = align_gcn(a, b, graph)
            state_a, state_b, state_aligned = (clone_parameters(model) for model in (a, b, aligned))
            curve = _load_checkpoint(source.parent, original_pair["checkpoint"], original["model"], metadata["split_sha256"], device)
            if curve["endpoint_seeds"] != [sa, sb]:
                raise ValueError("Bézier checkpoint endpoint seeds do not match the source pair.")
            ts = original_pair["linear"]["t"]
            if ts != original_pair["bezier"]["t"] or ts[0] != 0 or ts[-1] != 1:
                raise ValueError("Both source paths must use the same endpoint-inclusive grid.")
            pair = {"seed_a": sa, "seed_b": sb, "curve_seed": original_pair["curve_seed"], "alignment": alignment, "calibration": [], "hidden_variance": {method: {} for method in METHODS}}
            pair.update({method: _empty_path(ts) for method in METHODS})
            endpoint_variance_a = {name: float(std.square().mean()) for name, (_, std) in hidden_statistics(a, graph).items()}
            endpoint_variance_b = {name: float(std.square().mean()) for name, (_, std) in hidden_statistics(b, graph).items()}
            for t in ts:
                repaired, diagnostics = repair_gcn(a, aligned, graph, t)
                pair["calibration"].append({"t": t, "diagnostics": diagnostics})
                models = {
                    "linear": _materialize(a, state_a, state_b, t),
                    "aligned": _materialize(a, state_a, state_aligned, t),
                    "repaired": repaired,
                    "bezier": _materialize(a, state_a, state_b, t, curve["control"]),
                }
                for method, model in models.items():
                    with torch.no_grad():
                        measured = metrics(model(graph.x, graph.edge_index), graph)
                    for split, values in measured.items():
                        for metric, value in values.items():
                            pair[method]["splits"][split][metric].append(value)
                    for name, (_, std) in hidden_statistics(model, graph).items():
                        variance = float(std.square().mean())
                        endpoint_baseline = (1 - t) * endpoint_variance_a[name] + t * endpoint_variance_b[name]
                        values = pair["hidden_variance"][method].setdefault(name, {"mean_variance": [], "endpoint_variance_baseline": [], "variance_ratio": []})
                        values["mean_variance"].append(variance)
                        values["endpoint_variance_baseline"].append(endpoint_baseline)
                        values["variance_ratio"].append(variance / endpoint_baseline if endpoint_baseline > 0 else None)
            for method in METHODS:
                for values in pair[method]["splits"].values():
                    values.update(summarize_path(ts, values["loss"]))
                    values["min_accuracy"] = min(values["accuracy"])
            for method in ("linear", "bezier"):
                for split in ("train", "val", "test"):
                    before = original_pair[method]["splits"][split]["loss"]
                    after = pair[method]["splits"][split]["loss"]
                    if max(abs(x - y) for x, y in zip(before, after)) > 2e-5:
                        raise ValueError(f"The saved {method} curve does not reproduce on the current graph.")
            pair["checkpoint"] = f"{metadata['name']}/curve_{sa}_{sb}.pt"
            shutil.copy2(source.parent / original_pair["checkpoint"], output / pair["checkpoint"])
            pair["aligned_checkpoint"] = f"{metadata['name']}/aligned_{sa}_{sb}.pt"
            torch.save({"model_config": original["model"], "state_dict": cpu_state(aligned.state_dict()), "seed": sb, "alignment": alignment, "split_sha256": metadata["split_sha256"]}, output / pair["aligned_checkpoint"])
            midpoint, _ = repair_gcn(a, aligned, graph, 0.5)
            pair["repaired_midpoint_checkpoint"] = f"{metadata['name']}/repaired_{sa}_{sb}_midpoint.pt"
            torch.save({"model_config": original["model"], "state_dict": cpu_state(midpoint.state_dict()), "endpoint_seeds": [sa, sb], "alpha": 0.5, "split_sha256": metadata["split_sha256"]}, output / pair["repaired_midpoint_checkpoint"])
            dataset["pairs"].append(pair)
            print("  test barriers: " + ", ".join(f"{method}={pair[method]['splits']['test']['barrier']:.6f}" for method in METHODS), flush=True)
        dataset["summary"] = aggregate_pairs(dataset["pairs"], methods=METHODS)
        report["datasets"].append(dataset)
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(json.dumps({"source": str(source), "thesis_root": str(thesis_root) if thesis_root else None, "device": device, "threads": threads}, indent=2) + "\n")
    return output / "report.json"
