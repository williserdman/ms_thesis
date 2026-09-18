# GCN comparison across thesis datasets

This sweep uses **GCN only** on Cora, Roman-empire, squirrel, and chameleon.
The model has two graph-convolution layers, hidden width 64, ReLU, and dropout
0.5. The four plot legends are interpolation methods, not four architectures.
The paper's separate architecture comparison includes GCN, GraphSAGE, GAT, and
MLP. GraphSAGE, GAT, and MLP are not implemented in this experiment.
[Paper, section 3.1](https://arxiv.org/html/2502.12608v1#S3.SS1).

## Protocol

Every dataset uses endpoint pairs `0:1`, `2:3`, and `4:5`, 200 endpoint epochs,
200 Bézier optimization steps, and 21 equally spaced evaluation positions.
Endpoint selection minimizes validation loss. Hyperparameters match the original
Cora run; this sweep does not tune them for each dataset.

The four methods are raw linear interpolation, aligned linear interpolation,
aligned linear plus REPAIR, and the learned Bézier curve. REPAIR uses training-node
statistics and does not modify the Bézier comparator. See the
[integration guide](repair_integration.md) for the calibration definition.

The experiment calls `/home/wge3/ms_thesis/src/loading/LightningGraphLoader.py`.
Cora uses its public Planetoid masks. Roman-empire uses published split index 1
and normalized features. Squirrel and chameleon use the local
`*_filtered_directed.npz` files and split index 1; the loader preserves their
features and edges. These dataset variants and splits need not match the paper.

The loaded graphs have these dimensions and labeled-node counts:

| Dataset | Nodes | Edge entries | Features | Classes | Train / validation / test |
|---|---:|---:|---:|---:|---|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 / 500 / 1,000 |
| Roman-empire | 22,662 | 65,854 | 300 | 18 | 11,331 / 5,665 / 5,666 |
| squirrel | 2,223 | 65,718 | 2,089 | 5 | 1,080 / 700 / 443 |
| chameleon | 890 | 13,584 | 2,325 | 5 | 427 / 302 / 161 |

The fixed split is zero-based index 1, not an average over ten splits. Squirrel
and chameleon retain directed edge lists and their stored feature scaling;
`GCNConv` normalization does not convert them to undirected graphs.

## Run

Worktree: `/home/wge3/ms_thesis/.worktrees/gnn-repair/gcn_bezier_conn`.
CPU Slurm array `3834313`, tasks 0/1/2 for Roman-empire/squirrel/chameleon, was
submitted on 2026-09-18. Each task requests one CPU, 8 GB RAM, and one hour on
the `batch` partition. Cora reuses the completed `runs/cora-repair` result.

The submission script and logs are under `runs/dataset-sweep-20260918/`.
Equivalent commands for a new dataset run are:

```bash
cd /home/wge3/ms_thesis/.worktrees/gnn-repair/gcn_bezier_conn
PY=/home/wge3/miniconda3/envs/py312/bin/python
DATASET=Roman-empire
OUTPUT=runs/new-comparison/$DATASET
$PY -m gcn_mc run --datasets "$DATASET" \
  --thesis-root /home/wge3/ms_thesis --device cpu --threads 1 \
  --output "$OUTPUT/baseline"
$PY -m gcn_mc repair --source "$OUTPUT/baseline/report.json" \
  --thesis-root /home/wge3/ms_thesis --device cpu --threads 1 \
  --output "$OUTPUT/repair"
```

Run training inside a compute allocation. Output directories must be new or
empty. Replace `DATASET` with `squirrel`, `chameleon`, or `Cora` as needed.

## Results

All three Slurm tasks completed with exit code 0. Wall times were 12m35s for
Roman-empire, 9m48s for squirrel, and 7m32s for chameleon, including imports and
plotting. Cora reuses its completed comparison and was verified again.

Sampled test mean-cross-entropy barriers, mean ± population SD across three pairs:

| Dataset | Linear | Aligned linear | Aligned + REPAIR | Bézier |
|---|---:|---:|---:|---:|
| Cora | 0.342250 ± 0.026034 | 0.048018 ± 0.009413 | 0.033269 ± 0.008544 | 0.002499 ± 0.003534 |
| Roman-empire | 0.028559 ± 0.009636 | 0.002191 ± 0.001404 | 0.002213 ± 0.001405 | 0.000000 ± 0.000000 |
| squirrel | 0.004871 ± 0.003811 | 0.000129 ± 0.000183 | 0.000857 ± 0.000682 | 1.852222 ± 0.100071 |
| chameleon | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 2.641857 ± 0.065239 |

Mean test accuracy, percent. Endpoint values average six models; path values
average the three midpoints:

| Dataset | Endpoints | Linear | Aligned linear | Aligned + REPAIR | Bézier |
|---|---:|---:|---:|---:|---:|
| Cora | 81.283 | 79.967 | 80.300 | 80.367 | 79.133 |
| Roman-empire | 14.108 | 17.826 | 16.043 | 16.043 | 17.984 |
| squirrel | 35.139 | 34.462 | 34.011 | 34.011 | 33.258 |
| chameleon | 37.992 | 37.888 | 37.474 | 37.888 | 33.954 |

REPAIR's improvement beyond alignment on Cora does not extend consistently to
these runs. Its mean test barrier is slightly higher than aligned interpolation
on Roman-empire and squirrel. Chameleon's three linear variants all have zero
sampled test barriers.

Bézier fitting on the filtered datasets improves training loss while producing
large held-out loss barriers, consistent with overfitting. Mean midpoint
train/test losses are 0.5417/3.4327 for squirrel and 0.3851/4.1853 for chameleon.
The protocol retains the final control and does not select it using validation
loss. Endpoint validation selection retained epochs 3 to 4 for squirrel and 4 to 7
for chameleon.

Roman-empire's low endpoint accuracy is a material limitation. All six endpoints
selected epoch 200, the training-budget boundary. This run does not establish
convergence to strong solutions; low barriers alone are insufficient evidence
for a claim about connectivity between well-trained modes.

Compare methods within each dataset. These are fixed-settings trials using one
split, not a tuned architecture benchmark. A zero sampled barrier need not remain
zero between grid points. Population SD across three pairs is not a confidence
interval. The repaired path is not a straight parameter-space line.

## Verification and graphics

The existing runners completed end to end on all four datasets. REPAIR reproduced
the saved raw-linear and Bézier loss curves and checked loader/split provenance.
An independent verification process then reloaded all 12 repaired midpoint
checkpoints, checked all path endpoints, checked finite loss/accuracy values,
and confirmed the source report hashes and fresh split hashes. The largest
midpoint metric discrepancy was `2.384186e-7`, below the `2e-5` check tolerance.

Execution used the existing Python 3.14.0 / PyTorch 2.9.0+cu128 / PyG 2.7.0
environment on CPU, one PyTorch thread per task. No training, loader, or REPAIR
implementation changes were needed. Plot titles now identify GCN, depth, hidden
width, dropout, and the number of endpoint pairs.

The [four-dataset overview PNG](../results/gcn-four-datasets/comparison.png)
and [PDF](../results/gcn-four-datasets/comparison.pdf) show test loss and accuracy.
The overview PNG was visually inspected. Shaded bands are population SD across
pairs. Per-dataset figures also show train
and validation curves:

| Dataset | Full figure | Report |
|---|---|---|
| Cora | [PNG](../results/gcn-four-datasets/Cora/connectivity.png) | [JSON](../results/gcn-four-datasets/Cora/report.json) |
| Roman-empire | [PNG](../results/gcn-four-datasets/Roman-empire/connectivity.png) | [JSON](../results/gcn-four-datasets/Roman-empire/report.json) |
| squirrel | [PNG](../results/gcn-four-datasets/squirrel/connectivity.png) | [JSON](../results/gcn-four-datasets/squirrel/report.json) |
| chameleon | [PNG](../results/gcn-four-datasets/chameleon/connectivity.png) | [JSON](../results/gcn-four-datasets/chameleon/report.json) |

The [verified summary](../results/gcn-four-datasets/verified_summary.json)
contains the numbers used above. The run directory also retains the Slurm script,
logs, verification script/log, and overview plotting script. Full run artifacts
are ignored by git; compact report and figure copies are tracked in
[`results/gcn-four-datasets`](../results/gcn-four-datasets/README.md), without
model checkpoints. Reproduce the overview from the original local runs with:

```bash
$PY runs/dataset-sweep-20260918/plot_overview.py
```

Deferred work: validation-based endpoint and curve tuning, convergence checks,
multiple dataset splits, and the paper's MLP/GraphSAGE/GAT comparison with
architecture-specific alignment and REPAIR rules. Test metrics must remain
evaluation-only when selecting future settings.
