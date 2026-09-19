# Developer onboarding

This package reproduces the paper's core mode-connectivity procedure while
reusing the parent thesis data pipeline. It trains independently seeded
endpoints, evaluates their straight parameter path, learns a quadratic Bézier
control point, and records loss and accuracy along both paths. The baseline
runner supports the original legacy GCN and reference-profile GCN, MLP,
GraphSAGE, and GAT models.

The paper does not publish its splits or most training hyperparameters. The defaults below are explicit implementation choices. See [docs/paper_protocol.md](docs/paper_protocol.md) for the paper facts and replication gaps.

Completed full matrices cover all four architectures and datasets, with three
seed pairs each. Start with the [fixed-reference results](docs/reference_full_results.md)
and [endpoint-tuned comparison and graphics](results/tuned-endpoints-20260918/README.md).
The [alignment/REPAIR matrix](results/repair-tuned-20260918/README.md) adds all four
methods to the same tuned endpoints. Read the
[verification notes](docs/reference_repair_verification.md) for numerical checks and findings.

## Environment

Run commands from `/home/wge3/ms_thesis/.worktrees/gnn-repair/gcn_bezier_conn`.
This isolated worktree is on branch `exp/gnn-repair`. The original checkout and
its uncommitted files remain at `/home/wge3/ms_thesis`.

The working interpreter is:

```text
/home/wge3/miniconda3/envs/py312/bin/python
Python 3.14.0
PyTorch 2.9.0+cu128
PyG 2.7.0
Lightning 2.5.5
```

Despite its name, the `py312` environment currently contains Python 3.14. The separate `ms_thesis` environment lacks PyTorch. The login node has no GPU, so use `--device cpu` there. Use `--device cuda` only inside a GPU allocation.

No installation is needed from this source directory. For the commands below:

```bash
cd /home/wge3/ms_thesis/.worktrees/gnn-repair/gcn_bezier_conn
PY=/home/wge3/miniconda3/envs/py312/bin/python
$PY -m gcn_mc --help
```

Ordinary `run` commands infer their containing checkout as the thesis root.
Here, pass `--thesis-root /home/wge3/ms_thesis` to use the current loader and
existing datasets. A `repair` command defaults to the loader location recorded
in its source report. In another checkout, pass `--thesis-root /path/to/ms_thesis`.

## Run the reference architecture baselines

Use the pinned tunedGNN profile for one architecture and dataset:

```bash
$PY -m gcn_mc run \
  --thesis-root /home/wge3/ms_thesis \
  --preset reference \
  --architecture gat \
  --datasets Cora \
  --pairs 0:1 \
  --output runs/reference-cora-gat
```

`--architecture` accepts `gcn`, `mlp`, `graphsage`, or `gat`. The reference
preset resolves width, depth, dropout, endpoint epochs, endpoint learning rate,
weight decay, normalization, residual connections, input projection, attention
heads, and checkpoint selection separately for each dataset. The supported
reference datasets are Cora, Roman-empire, squirrel, and chameleon. Reference
endpoints select the epoch with highest validation accuracy.

Explicit CLI values override the resolved profile. The available overrides are
`--hidden-channels`, `--depth`, `--dropout`, `--epochs`, `--lr`,
`--weight-decay`, `--normalization`, `--residual` or `--no-residual`,
`--pre-linear` or `--no-pre-linear`, `--heads`, and `--selection`. The report and
checkpoint store the effective model configuration. Normalization accepts
`none`, `batch`, or `layer`; selection accepts `val_loss` or `val_accuracy`. The
pinned GAT profiles all use one attention head; multiple GAT heads are currently
rejected. Checkpoints store model configuration and weights; the report also
records optimizer settings, overrides, source provenance, and selection history.

Run a short check across all four architectures with a new output path for each:

```bash
for architecture in gcn mlp graphsage gat; do
  $PY -m gcn_mc run \
    --thesis-root /home/wge3/ms_thesis \
    --preset reference \
    --architecture "$architecture" \
    --datasets Cora \
    --smoke \
    --output "runs/smoke-cora-$architecture"
done
```

For the cluster's GPU partition, the included array script maps all 16
dataset/architecture combinations. Submit from this project directory:

```bash
sbatch --array=0-15%4 scripts/reference_baselines.sbatch runs/reference-smoke smoke
# Cora only, source endpoint budgets, one seed pair:
sbatch --array=0-3%4 scripts/reference_baselines.sbatch runs/reference-cora full --pairs 0:1
```

The script defaults to the interpreter and thesis root above. Override
`GNN_PYTHON` and `GNN_THESIS_ROOT` for another environment. See
[architecture verification](docs/architecture_verification.md) for completed runs.

The reference source is the official
[`LUOyk1999/tunedGNN`](https://github.com/LUOyk1999/tunedGNN) repository pinned
at commit `23f9604e8b13a9a6d3faa2f691cd844006979153`. The copied source subset,
checksums, and MIT License are under
[`upstream/tunedGNN`](upstream/tunedGNN/PROVENANCE.md). See the
[architecture source check](docs/architecture_sources.md) for exact profiles and
differences from this package. tunedGNN provides GCN, GraphSAGE, and GAT. The
reference MLP uses the matching GCN profile and replaces each graph convolution
with a linear layer. This is a local implementation choice because no upstream
MLP recipe was found.

## Tune and cache endpoint hyperparameters

The existing thesis `src/optuna_trainer.py` supplies the common search dimensions,
sampler/pruner, and study runner. Enable its GNN endpoint objective with:

```bash
$PY -m gcn_mc run --preset reference --architecture gcn --datasets Cora \
  --thesis-root /home/wge3/ms_thesis --tune-endpoints --tuning-trials 20 \
  --tuning-cache runs/optuna-cache --output runs/tuned-cora-gcn
```

Searches maximize validation accuracy of the selected, calibrated endpoint.
Training uses train-mask labels; tuning does not compute test metrics. It varies
learning rate, hidden width, dropout, depth, and weight decay. Normalization,
residuals, input projection, and attention heads retain their reference settings.
Explicit CLI overrides fix the corresponding search dimensions. The reference
configuration is the first trial. The default is 20 trials with tuning seed 42
and the dataset's full endpoint epoch budget; intermediate validation accuracy
supports pruning every ten epochs. These are best observed parameters within a
bounded search, not guaranteed global optima.

Each dataset/architecture/context gets `study.sqlite3` and `best.json` under the
cache directory. Matching completed studies run zero new trials. Increasing
`--tuning-trials` adds only the remaining trials; interrupted studies resume.
The cache persists independently of experiment output directories. A per-study
lock prevents simultaneous duplicate searches. Data, split, model/search settings,
training implementation, runtime, and tuning seed changes create distinct studies.
Final endpoint seeds, test targets, output paths, and Bézier settings do not.
Smoke searches use a separate identity and cannot supply full-budget parameters.

Final experiments retrain seeds 0–5 with the selected parameters and compare
three independent endpoint pairs. Bézier hyperparameters remain fixed, per the
user's request. Reports contain cache identity, hit/miss, best parameters,
validation score, and trial counts. The ignored cache databases remain local;
archived reports preserve the selected settings.

```bash
sbatch --array=0-15%4 --time=04:00:00 scripts/reference_baselines.sbatch \
  runs/tuned-matrix full --tune-endpoints --tuning-trials 20 \
  --tuning-cache runs/optuna-cache
```

To archive and audit a completed full matrix against its saved Optuna studies:

```bash
$PY scripts/archive_tuned_matrix.py runs/tuned-endpoints-20260918 \
  results/tuned-endpoints-20260918
```

This copies reports, selected parameters, and labeled plots. SQLite studies and
model weights remain in the ignored `runs/` directories; preserve those directories
to reuse searches and replay checkpoints.

## Compare REPAIR with existing paths

```bash
$PY -m gcn_mc repair \
  --source /home/wge3/ms_thesis/gcn_bezier_conn/runs/cora/report.json \
  --thesis-root /home/wge3/ms_thesis \
  --output runs/cora-repair
```

The command reuses saved endpoints and compares raw linear interpolation, channel
alignment, alignment plus REPAIR, and the saved Bézier baseline. Only the linear
path receives REPAIR. No training or new data splits occur. The output includes
aligned and repaired-midpoint state dictionaries, per-layer variance diagnostics,
four loss/accuracy curves, and plots. Read the
[integration guide](docs/repair_integration.md) for calibration policy and replay.
The [REPAIR verification record](docs/repair_verification.md) contains the completed
three-pair Cora results and checkpoint replay checks. Choose a new output directory
when repeating the command; `runs/cora-repair` already contains that result.
See the [four-dataset sweep](docs/dataset_sweep.md) for Roman-empire, squirrel,
chameleon, and Cora, including dataset variants and the fixed GCN configuration.
REPAIR also accepts the reference GCN, MLP, GraphSAGE and GAT reports. It reuses
each saved endpoint pair and Bézier control. Run the full tuned matrix with:

```bash
sbatch --array=0-15%4 scripts/repair_matrix.sbatch \
  runs/tuned-endpoints-20260918 runs/repair-tuned-20260918
```

Reference repaired checkpoints contain explicit affine corrections; load them
with `gcn_mc.reference_repair.load_repaired_model`. The integration guide explains
normalization, attention permutations and the training-node calibration policy.

## Run the experiment

Start with the short end-to-end run:

```bash
$PY -m gcn_mc run \
  --thesis-root /home/wge3/ms_thesis \
  --smoke \
  --datasets Cora \
  --pairs 0:1 \
  --output runs/cora_smoke
```

`--smoke` first resolves the legacy or reference profile, then caps the effective
configuration at width 8, 5 endpoint epochs, 5 curve steps, 5 path points, and
one seed pair. It checks the workflow and does not support a scientific claim.

Run one full endpoint pair:

```bash
$PY -m gcn_mc run \
  --thesis-root /home/wge3/ms_thesis \
  --datasets Cora \
  --pairs 0:1 \
  --output runs/cora_pair_0_1
```

Run the default three independent pairs on several graphs:

```bash
$PY -m gcn_mc run \
  --thesis-root /home/wge3/ms_thesis \
  --datasets Cora Citeseer Pubmed \
  --output runs/citation_three_pairs
```

The default pairs are `0:1 2:3 4:5`. `--datasets` accepts one or more names. Dataset matching is case-insensitive, and reports use the canonical loader name. Every `--output` target must be new or empty. Add `--no-plots` to save metrics without rendering figures.

Regenerate PNG and PDF figures without retraining:

```bash
$PY -m gcn_mc plot runs/cora_pair_0_1/report.json
```

Run the focused unit tests:

```bash
$PY -m unittest discover -s tests -v
```

These commands are the intended interfaces. Consult the [baseline verification](docs/verification.md) and [REPAIR verification](docs/repair_verification.md) for commands that have actually been executed and their results.

## Data behavior

`gcn_mc.data.load_graph` is a thin adapter around the public function `loading.LightningGraphLoader.load_datasets`. It delegates all downloads, splitting, and feature preprocessing to that loader. The adapter clones the returned PyG graph, validates disjoint nonempty masks, and records the loader hash, split counts, and split hash.

The parent loader supports:

```text
Questions, Cora, Roman-empire, computers, photo, Citeseer, Pubmed,
squirrel, chameleon, actor, texas, cornell, Amazon-ratings,
Minesweeper, Tolokers
```

Split and preprocessing behavior comes from that loader:

- Cora, CiteSeer, and PubMed use their public Planetoid masks.
- Datasets with multiple published masks use fixed split index 1.
- `computers` and `photo` use seeded random 60/20/20 masks. `--data-seed` controls this loader randomness.
- `squirrel` and `chameleon` require local filtered NPZ files. The loader prefers `<name>_filtered_directed.npz`, then `<name>_filtered.npz`; Squirrel also accepts `sqf.npz`. These filtered graphs and features can differ from the paper's data.
- The adapter preserves features and edges. It does not apply tunedGNN's graph
  conversion, self-loop rewrite, feature processing, or split rules. Individual
  PyG operators retain their configured behavior. Endpoint training uses
  unweighted cross-entropy even though the parent loader computes class weights.

The run saves the exact masks and a hash. Compare `split_sha256` before comparing or replaying results. Reusing the thesis masks and preprocessing is a documented deviation because the paper does not provide exact splits.

## Training and path definitions

Legacy endpoint models keep the existing interface:

```python
GCN(in_channels, hidden_channels, out_channels, depth=2, dropout=0.5)
```

Reference models are created through `build_model(model_config)`. They share a
`forward(x, edge_index)` interface and save the complete effective model
configuration. Legacy endpoints retain the epoch with minimum validation
cross-entropy. Reference endpoints retain the epoch with maximum validation
accuracy. Both use full-batch Adam on train-mask cross-entropy. Test labels
participate only in evaluation.

Reference profiles that use BatchNorm need fresh path-point statistics. Before
evaluating a selected endpoint or interpolation point, the runner performs one
full-graph, label-free calibration forward with dropout disabled and every
BatchNorm layer using batch statistics. Each point receives fresh buffers, so
statistics do not carry across the path. Calibration does not use train,
validation, or test labels. This transductive normalization policy is an explicit
implementation choice because the mode-connectivity paper does not specify
BatchNorm buffer handling. It is separate from REPAIR.

Keep the calibration device consistent when comparing BatchNorm paths. The
depth-9, width-8 Roman-empire GraphSAGE smoke model changes predictions when
statistics are recalibrated on CPU after GPU training; recalibrating both the
direct endpoint and its path point on CPU gives identical logits. The
[verification record](docs/architecture_verification.md) preserves this numerical
limitation and the one-node accuracy discrepancy from its original GPU run.

For endpoint parameters `a` and `b`, the straight path at `t` is

```text
(1 - t) a + t b
```

The quadratic Bézier path with learned control `c` is

```text
(1 - t)^2 a + 2 t (1 - t) c + t^2 b
```

`make_control` initializes `c` to the arithmetic midpoint. `path_logits` substitutes interpolated parameter tensors through `torch.func.functional_call`, so the train loss remains differentiable with respect to the control. Curve fitting freezes both endpoint dictionaries, samples `t` uniformly, and updates only the control with train-mask cross-entropy. It keeps the final control. Validation and test labels do not select or optimize it. Evaluation disables dropout.

The implementation uses mean cross-entropy for training, evaluation, and barriers. The paper displays summed cross-entropy. Absolute barrier values will not match unless the split and loss reduction also match.

Legacy defaults are:

| Setting | Value |
|---|---:|
| Hidden width, depth, dropout | 64, 2, 0.5 |
| Endpoint epochs | 200 |
| Endpoint Adam learning rate, weight decay | 0.01, 0.0005 |
| Curve steps, Adam learning rate | 200, 0.01 |
| Uniform `t` samples per curve step | 1 |
| Evaluation grid | 21 endpoint-inclusive points |
| Data seed, base curve seed | 0, 10000 |
| Device, PyTorch CPU threads | `cpu`, 1 |

Running `run` without `--preset reference` preserves these defaults and legacy
GCN behavior. Override them with `python -m gcn_mc run --help`. The paper does
not specify these values. `report.json` records all effective choices and
describes the barrier calculation.

## Artifacts

A completed run has this layout:

```text
<output>/
├── config.json
├── report.json
└── <Dataset>/
    ├── split.pt
    ├── endpoint_<seed>.pt
    ├── curve_<seed-a>_<seed-b>.pt
    ├── connectivity.png
    └── connectivity.pdf
```

`config.json` contains the requested configuration and explicit overrides.
`report.json` has `schema_version: 1`, environment and protocol metadata, then one
entry per dataset. Each dataset's `config` records its resolved preset, overrides,
and smoke caps; `model` records architecture settings. Entries also contain data
and source provenance, endpoint histories and metrics, both paths' raw metrics,
barriers, pair summaries, and checkpoint paths. Population standard deviation
across endpoint pairs is descriptive, not a confidence interval.

`split.pt` stores the three boolean masks. An endpoint checkpoint contains `model_config`, `state_dict`, `seed`, `selected_epoch`, and `split_sha256`. A curve checkpoint contains `model_config`, the control tensors, endpoint seeds, curve seed, and `split_sha256`.

Reload an endpoint and replay a curve point as follows:

```python
import json
from pathlib import Path

import torch

from gcn_mc.data import load_graph
from gcn_mc.model import build_model
from gcn_mc.paths import clone_parameters, path_logits

run = Path("runs/cora_pair_0_1")
report = json.loads((run / "report.json").read_text())
dataset = report["datasets"][0]
pair = dataset["pairs"][0]
endpoint_by_seed = {item["seed"]: item for item in dataset["endpoints"]}
record_a = endpoint_by_seed[pair["seed_a"]]
record_b = endpoint_by_seed[pair["seed_b"]]

checkpoint_a = torch.load(run / record_a["checkpoint"], map_location="cpu", weights_only=True)
checkpoint_b = torch.load(run / record_b["checkpoint"], map_location="cpu", weights_only=True)
curve = torch.load(run / pair["checkpoint"], map_location="cpu", weights_only=True)

model_a = build_model(checkpoint_a["model_config"])
model_b = build_model(checkpoint_b["model_config"])
model_a.load_state_dict(checkpoint_a["state_dict"])
model_b.load_state_dict(checkpoint_b["state_dict"])
template = build_model(curve["model_config"]).eval()

graph, metadata = load_graph(
    dataset["data"]["name"],
    thesis_root=report["config"]["thesis_root"],
    data_seed=report["config"]["data_seed"],
)
assert metadata["split_sha256"] == curve["split_sha256"]

with torch.no_grad():
    logits = path_logits(
        template,
        graph,
        clone_parameters(model_a),
        clone_parameters(model_b),
        0.5,
        curve["control"],
    )
```

## Code map

- `gcn_mc/data.py` adapts the parent loader, seeds data creation, validates masks, and records provenance.
- `gcn_mc/model.py` defines the legacy GCN and model factory.
- `gcn_mc/reference_models.py` adapts the pinned tunedGNN models and adds the MLP.
- `gcn_mc/presets.py` records dataset-specific tunedGNN profiles and provenance.
- `gcn_mc/paths.py` clones endpoint parameters, creates and interpolates controls,
  handles isolated BatchNorm calibration, runs differentiable functional calls,
  and computes sampled barriers.
- `gcn_mc/experiment.py` trains endpoints and controls, evaluates paths, and writes schema-versioned artifacts.
- `gcn_mc/__main__.py` validates CLI arguments and dispatches runs or plotting.
- `gcn_mc/plotting.py` renders saved reports with no retraining.
- `gcn_mc/repair_adapter.py` aligns GCN hidden channels and calibrates training-node preactivations after message passing.
- `gcn_mc/repair_experiment.py` compares REPAIR with existing linear and Bézier paths using saved checkpoints.
- `tests/test_paths.py` checks endpoint equivalence, midpoint behavior, control gradients, endpoint immutability, and barrier arithmetic.
- `tests/test_training.py` checks that held-out labels cannot affect endpoint training or selection, or control optimization.
- `tests/test_repair_adapter.py` checks graph permutation and calibration behavior.
- `tests/test_reference_models.py` checks source-model equivalence, source
  profiles, legacy replay, and MLP edge independence.
- `tests/test_batchnorm_paths.py` checks BatchNorm state isolation and label-free
  full-graph calibration.
- `tests/test_architecture_runner.py` checks architecture checkpoint replay and
  early rejection of unsupported REPAIR input.
- `docs/paper_protocol.md` records source-backed paper facts and implementation choices.
- `docs/verification.md` records executed checks and known runtime limits.

## Extension points

Add data support in the parent thesis loader so this package continues to use one preprocessing and split path. Both endpoints must have identical parameter names and shapes. Add path families and their differentiable interpolation in `paths.py`. Add metrics and new artifact fields in `experiment.py`, incrementing `schema_version` for incompatible report changes. Keep plotting as a report-only consumer.

The current phase defers extending REPAIR beyond legacy GCN, CSBM property
sweeps, the generalization study, and cross-domain Wasserstein experiments. The
parent loader does not currently supply Coauthor-CS, Coauthor-Physics, or WikiCS.

## Troubleshooting

- `Thesis loader missing`: pass the checkout containing `src/loading/LightningGraphLoader.py` with `--thesis-root`.
- `Unknown dataset`: use one of the canonical names above. Spelling is otherwise case-insensitive.
- `Unknown reference dataset`: reference presets cover only Cora, Roman-empire,
  squirrel, and chameleon. Use the legacy preset or add a source-backed profile.
- `REPAIR currently supports only legacy GCN reports`: run `repair` against a
  report created without the reference preset.
- `No filtered dataset found`: place the required Squirrel or Chameleon NPZ under `<thesis-root>/data/<name>/` using one of the filenames listed above.
- `CUDA requested but unavailable`: use `--device cpu` on the login node or run inside a GPU allocation.
- `Output directory must be empty`: choose a new run directory. This guard prevents mixed or overwritten artifacts.
- Import errors for Torch or PyG: invoke `/home/wge3/miniconda3/envs/py312/bin/python`; the `ms_thesis` environment does not contain PyTorch.
- Split hash mismatch during replay: use the original `--data-seed`, thesis checkout, and loader version recorded in `report.json`.
