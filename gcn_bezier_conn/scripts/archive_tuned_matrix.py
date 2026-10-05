"""Audit and archive the 16 completed endpoint-tuned runs and comparison figures."""

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcn_mc.presets import reference_profile
from gcn_mc.plotting import plot_report

DATASETS = ("Cora", "Roman-empire", "squirrel", "chameleon")
MODELS = {"gcn": "GCN", "mlp": "MLP", "graphsage": "GraphSAGE", "gat": "GAT"}


def archive(run_root, output, datasets=DATASETS):
    output.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "Optuna endpoint tuning only; three independent final pairs; fixed Bézier settings",
               "runs": {}, "endpoint_path_max_loss_error": 0., "endpoint_path_max_accuracy_error": 0.}
    comparison, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    lines = ["# Tuned endpoint results", "", summary["scope"] + ".", "",
             "Twenty total Optuna trials per configuration, including pruned trials; tuning seed 42. "
             "Selected configurations maximize validation accuracy. Final seeds 0–5 are independent of tuning. "
             "The thesis loader/masks and reference endpoint epoch budgets are retained.", "",
             "![Endpoint comparison](endpoint_comparison.png)", "",
             "| Dataset | Model | Fixed / tuned endpoint test accuracy | Linear / Bézier test barrier | Selected width / depth |",
             "|---|---|---:|---:|---:|"]
    for row, name in enumerate(datasets):
        summary["runs"][name] = {}
        fixed_means, tuned_means, fixed_stds, tuned_stds = [], [], [], []
        for architecture in MODELS:
            path = run_root / name / architecture / "report.json"
            report = json.loads(path.read_text())
            data = report["datasets"][0]
            config, tuning, model = data["config"], data["tuning"], data["model"]
            profile = reference_profile(name, architecture)
            baseline_path = ROOT / "results/reference-full-20260918" / name / architecture / "report.json"
            baseline = json.loads(baseline_path.read_text())["datasets"][0]
            assert data["data"] == baseline["data"], (name, architecture, "data identity")
            assert config["epochs"] == profile["training"]["epochs"]
            assert config["curve_epochs"] == 200 and config["curve_lr"] == .01 and config["curve_samples"] == 1
            assert config["selection"] == "val_accuracy" and config["tuning_seed"] == 42
            assert [endpoint["seed"] for endpoint in data["endpoints"]] == list(range(6))
            assert [[pair["seed_a"], pair["seed_b"]] for pair in data["pairs"]] == [[0, 1], [2, 3], [4, 5]]
            assert tuning["finished_trials"] == tuning["requested_trials"] == 20
            assert hashlib.sha256(json.dumps(tuning["context"], sort_keys=True).encode()).hexdigest() == tuning["key"]
            for field, value in tuning["best_params"].items():
                assert config[field] == value
                if field in model:
                    assert model[field] == value
            for field in ("normalization", "residual", "pre_linear", "heads"):
                assert model[field] == profile["model"][field]
            study = optuna.load_study(study_name=tuning["study_name"], storage=f"sqlite:///{tuning['study_file']}")
            assert study.best_trial.number == tuning["best_trial"] and study.best_value == tuning["best_validation_accuracy"]
            assert sum(trial.state.is_finished() for trial in study.trials) == 20
            endpoints = {endpoint["seed"]: endpoint for endpoint in data["endpoints"]}
            for endpoint in endpoints.values():
                assert len(endpoint["history"]) == config["epochs"]
                assert endpoint["selected_epoch"] == max(endpoint["history"], key=lambda item: item["val_accuracy"])["epoch"]
            for pair in data["pairs"]:
                assert len(pair["history"]) == 200
                for method in ("linear", "bezier"):
                    assert pair[method]["t"] == [i / 20 for i in range(21)]
                    for split in ("train", "val", "test"):
                        for metric, tolerance in (("loss", 2e-5), ("accuracy", 1e-6)):
                            values = pair[method]["splits"][split][metric]
                            assert len(values) == 21 and np.isfinite(values).all()
                            for index, seed in ((0, pair["seed_a"]), (-1, pair["seed_b"])):
                                error = abs(values[index] - endpoints[seed]["metrics"][split][metric])
                                summary[f"endpoint_path_max_{metric}_error"] = max(summary[f"endpoint_path_max_{metric}_error"], error)
                                assert error <= tolerance, (name, architecture, method, seed, split, metric, error)
            fixed = [endpoint["metrics"]["test"]["accuracy"] for endpoint in baseline["endpoints"]]
            tuned = [endpoint["metrics"]["test"]["accuracy"] for endpoint in data["endpoints"]]
            barriers = {method: data["summary"][method]["test"] for method in ("linear", "bezier")}
            fixed_means.append(statistics.mean(fixed)); fixed_stds.append(statistics.pstdev(fixed))
            tuned_means.append(statistics.mean(tuned)); tuned_stds.append(statistics.pstdev(tuned))
            summary["runs"][name][architecture] = {
                "best_params": tuning["best_params"], "best_validation_accuracy": study.best_value,
                "trial_states": {state: sum(trial.state.name == state for trial in study.trials) for state in ("COMPLETE", "PRUNED", "FAIL")},
                "cache_key": tuning["key"], "model": model,
                "fixed_endpoint_test_accuracy_mean": statistics.mean(fixed),
                "tuned_endpoint_test_accuracy_mean": statistics.mean(tuned),
                "tuned_endpoint_test_accuracy_std": statistics.pstdev(tuned), "test_barriers": barriers,
            }
            lines.append(f"| {name} | {MODELS[architecture]} | {statistics.mean(fixed):.3f} / {statistics.mean(tuned):.3f} | "
                         f"{barriers['linear']['barrier_mean']:.3f} / {barriers['bezier']['barrier_mean']:.3f} | "
                         f"{model['hidden_channels']} / {model['depth']} |")
            destination = output / name / architecture
            destination.mkdir(parents=True, exist_ok=True)
            plot_report(path)
            shutil.copy2(path, destination / "report.json")
            shutil.copy2(tuning["best_parameters_file"], destination / "best.json")
            for suffix in ("png", "pdf"):
                shutil.copy2(path.parent / name / f"connectivity.{suffix}", destination / f"connectivity.{suffix}")
        ax = axes.flat[row]
        positions = np.arange(4)
        ax.bar(positions - .18, fixed_means, .36, yerr=fixed_stds, label="Reference preset", color="#687e99", capsize=3)
        ax.bar(positions + .18, tuned_means, .36, yerr=tuned_stds, label="Optuna endpoints", color="#35946f", capsize=3)
        ax.set_xticks(positions, list(MODELS.values()))
        ax.set_title(name)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Endpoint test accuracy")
        ax.grid(axis="y", alpha=.2)
    comparison.suptitle("All four datasets and architectures: six final endpoints each\nError bars: population standard deviation across endpoint seeds")
    comparison.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    comparison.tight_layout(rect=(0, .06, 1, .91))
    for suffix in ("png", "pdf"):
        comparison.savefig(output / f"endpoint_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(comparison)
    lines.extend(["", "Each dataset/model folder contains the report, cached best parameters, and labeled PNG/PDF path plots. "
                  "Raw model checkpoints and SQLite studies stay in the local ignored `runs/` directories.", "",
                  "This is a bounded validation search, not proof of globally optimal hyperparameters or an exact numerical paper reproduction. "
                  "Bézier fitting was not tuned. Test metrics were used only for this final comparison.", "",
                  "The audit checks reports against persisted Optuna studies, source budgets, data identities, and endpoint/path boundary metrics. "
                  "It does not retrain models or replay every checkpoint."])
    (output / "README.md").write_text("\n".join(lines) + "\n")
    summary["verified_reports"] = len(datasets) * len(MODELS)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    archive(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
