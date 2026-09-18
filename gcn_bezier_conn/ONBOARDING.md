# Developer onboarding

This package reproduces the paper's core GCN mode-connectivity procedure while reusing the parent thesis data pipeline. It trains independently seeded GCN endpoints, evaluates their straight parameter path, learns a quadratic Bézier control point, and records loss and accuracy along both paths.

The current experiment and plotting workflow support the GCN architecture only.

The paper does not publish its splits or most training hyperparameters. The defaults below are explicit implementation choices. See [docs/paper_protocol.md](docs/paper_protocol.md) for the paper facts and replication gaps.

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

`--smoke` caps endpoint and curve training at 5 steps each, uses width 8, evaluates 5 path points, and keeps only the first seed pair. It checks the workflow and does not support a scientific claim.

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
- The adapter preserves features and edges. Each `GCNConv` adds self-loops and symmetric normalization. Endpoint training uses unweighted cross-entropy even though the parent loader computes class weights.

The run saves the exact masks and a hash. Compare `split_sha256` before comparing or replaying results. Reusing the thesis masks and preprocessing is a documented deviation because the paper does not provide exact splits.

## Training and path definitions

The endpoint model has the interface:

```python
GCN(in_channels, hidden_channels, out_channels, depth=2, dropout=0.5)
```

Endpoints use full-batch Adam on train-mask cross-entropy. The run retains the epoch with minimum validation cross-entropy. Test labels participate only in evaluation.

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

Current defaults are:

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

Override these with `python -m gcn_mc run --help`. The paper does not specify these values. `report.json` records all effective choices and describes the barrier calculation.

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

`config.json` contains the effective CLI configuration. `report.json` has `schema_version: 1`, environment and protocol metadata, then one entry per dataset. Each dataset entry contains data provenance, model configuration, endpoint histories and metrics, raw linear and Bézier curves for every split, sampled barriers, pair summaries, and checkpoint paths. Population standard deviation across endpoint pairs is descriptive and is not a confidence interval.

`split.pt` stores the three boolean masks. An endpoint checkpoint contains `model_config`, `state_dict`, `seed`, `selected_epoch`, and `split_sha256`. A curve checkpoint contains `model_config`, the control tensors, endpoint seeds, curve seed, and `split_sha256`.

Reload an endpoint and replay a curve point as follows:

```python
import json
from pathlib import Path

import torch

from gcn_mc.data import load_graph
from gcn_mc.model import GCN
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

model_a = GCN(**checkpoint_a["model_config"])
model_b = GCN(**checkpoint_b["model_config"])
model_a.load_state_dict(checkpoint_a["state_dict"])
model_b.load_state_dict(checkpoint_b["state_dict"])
template = GCN(**curve["model_config"]).eval()

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
- `gcn_mc/model.py` defines the configurable GCN.
- `gcn_mc/paths.py` clones endpoint parameters, creates and interpolates controls, runs differentiable functional calls, and computes sampled barriers.
- `gcn_mc/experiment.py` trains endpoints and controls, evaluates paths, and writes schema-versioned artifacts.
- `gcn_mc/__main__.py` validates CLI arguments and dispatches runs or plotting.
- `gcn_mc/plotting.py` renders saved reports with no retraining.
- `gcn_mc/repair_adapter.py` aligns GCN hidden channels and calibrates training-node preactivations after message passing.
- `gcn_mc/repair_experiment.py` compares REPAIR with existing linear and Bézier paths using saved checkpoints.
- `tests/test_paths.py` checks endpoint equivalence, midpoint behavior, control gradients, endpoint immutability, and barrier arithmetic.
- `tests/test_training.py` checks that held-out labels cannot affect endpoint training or selection, or control optimization.
- `tests/test_repair_adapter.py` checks graph permutation and calibration behavior.
- `docs/paper_protocol.md` records source-backed paper facts and implementation choices.
- `docs/verification.md` records executed checks and known runtime limits.

## Extension points

Add data support in the parent thesis loader so this package continues to use one preprocessing and split path. Add architectures in `model.py`; both endpoints must have identical parameter names and shapes. Add path families and their differentiable interpolation in `paths.py`. Add metrics and new artifact fields in `experiment.py`, incrementing `schema_version` for incompatible report changes. Keep plotting as a report-only consumer.

The MVP defers the paper's MLP, GraphSAGE, and GAT comparison, CSBM property sweeps, generalization study, and cross-domain Wasserstein experiments. The parent loader does not currently supply Coauthor-CS, Coauthor-Physics, or WikiCS.

## Troubleshooting

- `Thesis loader missing`: pass the checkout containing `src/loading/LightningGraphLoader.py` with `--thesis-root`.
- `Unknown dataset`: use one of the canonical names above. Spelling is otherwise case-insensitive.
- `No filtered dataset found`: place the required Squirrel or Chameleon NPZ under `<thesis-root>/data/<name>/` using one of the filenames listed above.
- `CUDA requested but unavailable`: use `--device cpu` on the login node or run inside a GPU allocation.
- `Output directory must be empty`: choose a new run directory. This guard prevents mixed or overwritten artifacts.
- Import errors for Torch or PyG: invoke `/home/wge3/miniconda3/envs/py312/bin/python`; the `ms_thesis` environment does not contain PyTorch.
- Split hash mismatch during replay: use the original `--data-seed`, thesis checkout, and loader version recorded in `report.json`.
