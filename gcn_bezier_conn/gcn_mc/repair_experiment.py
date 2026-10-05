"""Compare alignment and graph REPAIR using a saved connectivity run."""

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
from .experiment import Config, aggregate_pairs, cpu_state, fit_curve, metrics
from .model import build_model
from .paths import calibrate_batchnorm, clone_parameters, interpolate, summarize_path
from .repair_adapter import align_gcn, hidden_statistics, repair_gcn


METHODS = ("linear", "aligned", "repaired", "bezier")
BEZIER_METHODS = METHODS + ("aligned_bezier", "repaired_bezier")


def _load_checkpoint(source_dir, relative, model_config, split_hash, device):
    checkpoint = torch.load(source_dir / relative, map_location=device, weights_only=True)
    if checkpoint["model_config"] != model_config:
        raise ValueError(f"Model configuration mismatch in {relative}.")
    if checkpoint["split_sha256"] != split_hash:
        raise ValueError(f"Split identity mismatch in {relative}.")
    return checkpoint


def _materialize(template, a, b, t, control=None):
    model = copy.deepcopy(template)
    state = model.state_dict()
    state.update(interpolate(a, b, t, control))
    model.load_state_dict(state)
    return model.eval()


def _materialize_original(template, graph, a, b, t, control=None):
    model = _materialize(template, a, b, t, control)
    calibrate_batchnorm(model, graph)
    return model.eval()


def load_repaired_model(checkpoint, device="cpu"):
    """Load either a normal checkpoint or a reference-affine REPAIR checkpoint."""
    from .reference_repair import load_repaired_model as load_checkpoint_model
    return load_checkpoint_model(checkpoint, device)


def _moment_variances(statistics):
    return {name: float(std.square().mean()) for name, (_, std) in statistics.items()}


def _assert_replay(expected, actual, context):
    for split in ("train", "val", "test"):
        if abs(actual[split]["loss"] - expected[split]["loss"]) > 2e-5:
            raise ValueError(f"{context} loss does not reproduce on the current graph.")
        if abs(actual[split]["accuracy"] - expected[split]["accuracy"]) > 1e-6:
            raise ValueError(f"{context} accuracy does not reproduce on the current graph.")


def _validate_source_paths(original_pair, measured_pair, graph):
    replay = {}
    for method in ("linear", "bezier"):
        replay[method] = {}
        for split in ("train", "val", "test"):
            expected = original_pair[method]["splits"][split]
            actual = measured_pair[method]["splits"][split]
            loss_error = max(abs(x - y) for x, y in zip(expected["loss"], actual["loss"]))
            accuracy_error = max(abs(x - y) for x, y in zip(expected["accuracy"], actual["accuracy"]))
            count = int(getattr(graph, f"{split}_mask").sum())
            count_difference = max(
                abs(round(x * count) - round(y * count))
                for x, y in zip(expected["accuracy"], actual["accuracy"])
            )
            if loss_error > 2e-5:
                raise ValueError(f"The saved {method} curve loss does not reproduce on the current graph.")
            if count_difference > 1:
                raise ValueError(
                    f"The saved {method} curve accuracy differs by {count_difference} correct predictions."
                )
            replay[method][split] = {
                "max_loss_error": loss_error,
                "max_accuracy_error": accuracy_error,
                "max_correct_count_difference": count_difference,
            }
    return replay


def _empty_path(ts):
    return {"t": list(ts), "splits": {split: {"loss": [], "accuracy": []} for split in ("train", "val", "test")}}


def run_repair(source, output, *, thesis_root=None, device="cpu", threads=1, include_bezier=False):
    """Reuse endpoints and optionally fit and repair an aligned Bézier path."""
    methods = BEZIER_METHODS if include_bezier else METHODS
    torch.set_num_threads(threads)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    source = Path(source).expanduser().resolve()
    source_bytes = source.read_bytes()
    baseline = json.loads(source_bytes)
    if baseline.get("schema_version") != 1 or baseline.get("experiment") == "gnn_repair":
        raise ValueError("--source must be an original schema-v1 connectivity report.")
    for dataset in baseline.get("datasets", []):
        model_config = dataset.get("model", {})
        family = model_config.get("family", "legacy")
        if family not in {"legacy", "tunedgnn"}:
            raise ValueError(f"Unsupported model family {family!r} in source report.")
        if family == "legacy" and model_config.get("architecture", "gcn") != "gcn":
            raise ValueError("Unsupported legacy architecture in source report.")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = copy.deepcopy(baseline)
    report.update({
        "experiment": "gnn_repair",
        "methods": list(methods),
        "scope": "Saved endpoint comparison: channel alignment and sequential REPAIR on linear paths; existing Bézier comparator.",
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
        "methods": list(methods),
        "training": "No retraining or curve fitting; original endpoint and control checkpoints reused.",
        "source_replay": "Source linear and Bézier paths require loss errors <=2e-5 and at most one correct-prediction difference per split; original source metrics are retained verbatim.",
        "alignment": "Hungarian matching of training-node hidden correlations; hidden channels only; full-graph logit invariance checked.",
        "calibration": "Training-node population moments after each complete hidden block, before ReLU; optional input projection before dropout; sequential correction; dropout disabled.",
        "normalization": "Original full-graph BatchNorm calibration retained for linear/Bézier/aligned paths. Repaired interiors calibrate the aligned linear base once, then freeze native buffers during sequential corrections.",
        "transductive": "Full-graph message passing remains active; only train-mask rows enter matching and moment estimation; no labels used.",
        "targets": "Weighted endpoint means and standard deviations, not weighted variances.",
        "epsilon": 1e-5,
        "endpoints": "Exact endpoint copies at t=0 and t=1; classifier excluded from correction.",
        "path_interpretation": "The repaired path is not a straight weight-space segment. Bézier receives no REPAIR.",
        "variance_ratio": "Mean training-node channel variance divided by the t-weighted mean endpoint channel variances. Null if denominator is zero.",
    }
    if include_bezier:
        report["scope"] = "Saved endpoints: raw, aligned, and posthoc-repaired linear and quadratic Bézier paths."
        report["protocol"]["repair"].update({
            "training": "Original endpoints and raw Bézier controls reused. A separate aligned Bézier control is fitted with the source dataset's effective curve settings and pair seed.",
            "normalization": "Unrepaired interiors use full-graph, label-free BatchNorm calibration. Each repaired linear or aligned Bézier interior calibrates its base once, then freezes native buffers during sequential corrections.",
            "path_interpretation": "Align endpoint B, fit the aligned Bézier control, then apply REPAIR to each sampled model. REPAIR is posthoc, never part of curve optimization. Targets remain the aligned endpoint means and standard deviations.",
            "method_names": {"linear": "Raw linear", "aligned": "Aligned linear", "repaired": "Aligned linear + REPAIR", "bezier": "Raw Bézier", "aligned_bezier": "Aligned Bézier", "repaired_bezier": "Aligned Bézier + REPAIR"},
        })
    import repair.core
    report["repair_implementation"] = {
        "core": str(Path(repair.core.__file__).resolve()),
        "core_sha256": hashlib.sha256(Path(repair.core.__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).with_name("repair_adapter.py").read_bytes()).hexdigest(),
    }
    reference_adapter = Path(__file__).with_name("reference_repair.py")
    if reference_adapter.exists():
        report["repair_implementation"]["reference_adapter_sha256"] = hashlib.sha256(reference_adapter.read_bytes()).hexdigest()
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
        dataset = {key: copy.deepcopy(value) for key, value in original.items()
                   if key not in {"data", "model", "endpoints", "pairs", "summary"}}
        dataset.update({"data": metadata, "model": copy.deepcopy(original["model"]), "endpoints": [], "pairs": []})
        is_reference = original["model"].get("family") == "tunedgnn"
        if is_reference:
            from .reference_repair import align_reference, reference_statistics, repair_reference
            align_model, statistics, repair_model = align_reference, reference_statistics, repair_reference
        else:
            align_model, statistics, repair_model = align_gcn, hidden_statistics, repair_gcn
        endpoints = {}
        for record in original["endpoints"]:
            checkpoint = _load_checkpoint(source.parent, record["checkpoint"], original["model"], metadata["split_sha256"], device)
            model = build_model(checkpoint["model_config"]).to(device).eval()
            model.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                replayed = metrics(model(graph.x, graph.edge_index), graph)
            _assert_replay(record["metrics"], replayed, "Endpoint")
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
            aligned, alignment = align_model(a, b, graph)
            state_a, state_b, state_aligned = (clone_parameters(model) for model in (a, b, aligned))
            curve = _load_checkpoint(source.parent, original_pair["checkpoint"], original["model"], metadata["split_sha256"], device)
            if curve["endpoint_seeds"] != [sa, sb]:
                raise ValueError("Bézier checkpoint endpoint seeds do not match the source pair.")
            ts = original_pair["linear"]["t"]
            if ts != original_pair["bezier"]["t"] or ts[0] != 0 or ts[-1] != 1:
                raise ValueError("Both source paths must use the same endpoint-inclusive grid.")
            pair = {"seed_a": sa, "seed_b": sb, "curve_seed": original_pair["curve_seed"], "alignment": alignment, "calibration": [], "hidden_variance": {method: {} for method in methods}}
            pair.update({method: _empty_path(ts) for method in methods})
            aligned_control = None
            if include_bezier:
                effective_config = Config(**original.get("config", baseline["config"]))
                effective_config.device = device
                effective_config.threads = threads
                aligned_control, pair["aligned_history"] = fit_curve(
                    copy.deepcopy(a), graph, state_a, state_aligned,
                    effective_config, original_pair["curve_seed"],
                )
                pair["bezier_calibration"] = []
            endpoint_variance_a = _moment_variances(statistics(a, graph))
            endpoint_variance_b = _moment_variances(statistics(b, graph))
            for t in ts:
                if t == 0:
                    repaired, diagnostics = copy.deepcopy(a).eval(), {"endpoint": "a"}
                elif t == 1:
                    repaired, diagnostics = copy.deepcopy(aligned).eval(), {"endpoint": "aligned_b"}
                else:
                    repaired, diagnostics = repair_model(a, aligned, graph, t)
                pair["calibration"].append({"t": t, "diagnostics": diagnostics})
                aligned_path = (copy.deepcopy(a).eval() if t == 0 else
                                copy.deepcopy(aligned).eval() if t == 1 else
                                _materialize_original(a, graph, state_a, state_aligned, t))
                models = {
                    "linear": _materialize_original(a, graph, state_a, state_b, t),
                    "aligned": aligned_path,
                    "repaired": repaired,
                    "bezier": _materialize_original(a, graph, state_a, state_b, t, curve["control"]),
                }
                if include_bezier:
                    # Leave the base uncalibrated: the repair adapter calibrates once.
                    bezier_base = _materialize(a, state_a, state_aligned, t, aligned_control)
                    repaired_bezier, bezier_diagnostics = repair_model(
                        a, aligned, graph, t, path_model=bezier_base,
                    )
                    pair["bezier_calibration"].append({"t": t, "diagnostics": bezier_diagnostics})
                    models["aligned_bezier"] = (
                        copy.deepcopy(a).eval() if t == 0 else
                        copy.deepcopy(aligned).eval() if t == 1 else
                        _materialize_original(a, graph, state_a, state_aligned, t, aligned_control)
                    )
                    models["repaired_bezier"] = repaired_bezier
                for method, model in models.items():
                    with torch.no_grad():
                        measured = metrics(model(graph.x, graph.edge_index), graph)
                    for split, values in measured.items():
                        for metric, value in values.items():
                            pair[method]["splits"][split][metric].append(value)
                    for name, variance in _moment_variances(statistics(model, graph)).items():
                        endpoint_baseline = (1 - t) * endpoint_variance_a[name] + t * endpoint_variance_b[name]
                        values = pair["hidden_variance"][method].setdefault(name, {"mean_variance": [], "endpoint_variance_baseline": [], "variance_ratio": []})
                        values["mean_variance"].append(variance)
                        values["endpoint_variance_baseline"].append(endpoint_baseline)
                        values["variance_ratio"].append(variance / endpoint_baseline if endpoint_baseline > 0 else None)
            for method in methods:
                for values in pair[method]["splits"].values():
                    values.update(summarize_path(ts, values["loss"]))
                    values["min_accuracy"] = min(values["accuracy"])
            pair["source_replay"] = _validate_source_paths(original_pair, pair, graph)
            for method in ("linear", "bezier"):
                pair[method] = copy.deepcopy(original_pair[method])
            pair["checkpoint"] = f"{metadata['name']}/curve_{sa}_{sb}.pt"
            shutil.copy2(source.parent / original_pair["checkpoint"], output / pair["checkpoint"])
            pair["aligned_checkpoint"] = f"{metadata['name']}/aligned_{sa}_{sb}.pt"
            aligned_checkpoint = {"model_config": original["model"], "state_dict": cpu_state(aligned.state_dict()), "seed": sb, "alignment": alignment, "split_sha256": metadata["split_sha256"]}
            if include_bezier:
                aligned_checkpoint.update({"control": cpu_state(aligned_control), "endpoint_seeds": [sa, sb], "curve_seed": original_pair["curve_seed"], "curve_history": pair["aligned_history"], "path_family": "aligned_bezier"})
            torch.save(aligned_checkpoint, output / pair["aligned_checkpoint"])
            midpoint_methods = ("repaired", "repaired_bezier") if include_bezier else ("repaired",)
            for method in midpoint_methods:
                kwargs = {}
                if method == "repaired_bezier":
                    kwargs["path_model"] = _materialize(a, state_a, state_aligned, 0.5, aligned_control)
                midpoint, _ = repair_model(a, aligned, graph, 0.5, **kwargs)
                checkpoint_key = f"{method}_midpoint_checkpoint"
                pair[checkpoint_key] = f"{metadata['name']}/{method}_{sa}_{sb}_midpoint.pt"
                checkpoint = {"model_config": original["model"], "state_dict": cpu_state(midpoint.state_dict()), "endpoint_seeds": [sa, sb], "alpha": 0.5, "split_sha256": metadata["split_sha256"], "path_family": method}
                if is_reference:
                    checkpoint["repair_format"] = "reference-affine-v1"
                torch.save(checkpoint, output / pair[checkpoint_key])
                replay_model = load_repaired_model(torch.load(output / pair[checkpoint_key], map_location=device, weights_only=True), device)
                with torch.no_grad():
                    replay_metrics = metrics(replay_model(graph.x, graph.edge_index), graph)
                    midpoint_metrics = metrics(midpoint(graph.x, graph.edge_index), graph)
                _assert_replay(midpoint_metrics, replay_metrics, f"Saved {method} midpoint")
                prefix = "midpoint" if method == "repaired" else "repaired_bezier_midpoint"
                for metric, tolerance in (("loss", 2e-5), ("accuracy", 1e-6)):
                    errors = [abs(replay_metrics[s][metric] - midpoint_metrics[s][metric]) for s in ("train", "val", "test")]
                    if 0.5 in ts:
                        index = ts.index(0.5)
                        errors.extend(abs(replay_metrics[s][metric] - pair[method]["splits"][s][metric][index]) for s in ("train", "val", "test"))
                    pair[f"{prefix}_replay_max_{metric}_error"] = max(errors)
                    if max(errors) > tolerance:
                        raise ValueError(f"The saved {method} midpoint does not reproduce its recorded path.")
            dataset["pairs"].append(pair)
            print("  test barriers: " + ", ".join(f"{method}={pair[method]['splits']['test']['barrier']:.6f}" for method in methods), flush=True)
        dataset["summary"] = aggregate_pairs(dataset["pairs"], methods=methods)
        report["datasets"].append(dataset)
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(json.dumps({"source": str(source), "thesis_root": str(thesis_root) if thesis_root else None, "device": device, "threads": threads, "include_bezier": include_bezier}, indent=2) + "\n")
    return output / "report.json"
