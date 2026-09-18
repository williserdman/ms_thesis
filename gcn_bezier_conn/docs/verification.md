# Verification

Date: 2026-09-17. CPU execution using
`/home/wge3/miniconda3/envs/py312/bin/python`, Python 3.14.0, PyTorch 2.9.0+cu128,
PyG 2.7.0, and PyTorch Lightning 2.5.5. No packages or shared configuration changed.
Cold imports from the shared filesystem took several minutes.

## Focused tests

Seven tests passed in 1.318 seconds after imports. They check path endpoints,
the midpoint control, control gradient flow and updates, fixed endpoint weights,
GCN depth, hand-calculated barriers, and isolation from held-out labels.

Reproduce with:

```bash
cd /home/wge3/ms_thesis/gcn_bezier_conn
/home/wge3/miniconda3/envs/py312/bin/python -m unittest discover -s tests -v
```

Verification used `unittest.defaultTestLoader.discover('tests')` and
`unittest.TextTestRunner(verbosity=2)` in one interpreter, followed by the CLI's
`main()` function with the arguments shown below. This avoided repeated cold
imports. Checkpoints were reloaded in that same process for comparison with the
saved reports.

## Cora smoke run

```bash
/home/wge3/miniconda3/envs/py312/bin/python -m gcn_mc run \
  --smoke --output runs/smoke
```

Completed and saved both endpoints, the learned control, split masks, JSON,
PNG, and PDF. The PNG was visually inspected. Effective settings were five
epochs per endpoint, five curve steps, hidden width eight, one seed pair, and
five path points. Application time was 67.59 seconds, including loader import
and data loading but excluding plot rendering and initial package imports.

The adapter loaded cached Cora through the thesis pipeline: 2,708 nodes, 10,556
edge entries, 1,433 features, seven classes, and 140/500/1,000 train/validation/test
labels. Split SHA-256:
`6bd8cc27013b1270c884603c41a93723dd92bf413c1e43b55cf5401c9cc13a2e`.

The smoke run checks execution only; its short endpoint training is unsuitable
for assessing the paper's claims.

## Three-pair Cora run

```bash
/home/wge3/miniconda3/envs/py312/bin/python -m gcn_mc run \
  --datasets Cora --output runs/cora
```

Completed in 93.81 application seconds, excluding initial imports and plotting.
Used the default six endpoint seeds, paired as `0:1`, `2:3`, and `4:5`.
Each endpoint trained for 200 epochs; each control trained for 200 steps.
Both paths were evaluated at 21 positions. All reported values were finite.

| Split | Linear barrier, mean ± population SD | Bézier barrier, mean ± population SD |
|---|---:|---:|
| Train | 0.444850 ± 0.035857 | 0.000000 ± 0.000000 |
| Validation | 0.324542 ± 0.022200 | 0.019892 ± 0.018307 |
| Test | 0.342250 ± 0.026034 | 0.002499 ± 0.003534 |

Endpoint test accuracies for seeds 0 through 5 were 81.7%, 80.9%, 80.3%,
82.0%, 81.6%, and 81.2%. Per-pair test barriers were:

| Seeds | Linear | Bézier |
|---|---:|---:|
| 0:1 | 0.375768 | 0.000000 |
| 2:3 | 0.312297 | 0.000000 |
| 4:5 | 0.338685 | 0.007497 |

Reloaded all six endpoint checkpoints and all three control checkpoints with
`torch.load(..., weights_only=True)`. Recomputed every Bézier midpoint and
matched its stored loss and accuracy on all three splits within `1e-7`.
The sampled endpoints of both paths also matched the separately recorded
endpoint losses within `1e-7`. A fresh loader call reproduced the split hash.

The PNG was visually inspected; PNG and PDF show all three splits with mean
curves and population-standard-deviation bands. Local artifacts are ignored by
git and remain in the original checkout:
[report](/home/wge3/ms_thesis/gcn_bezier_conn/runs/cora/report.json),
[figure](/home/wge3/ms_thesis/gcn_bezier_conn/runs/cora/Cora/connectivity.png),
[PDF](/home/wge3/ms_thesis/gcn_bezier_conn/runs/cora/Cora/connectivity.pdf). Rerun the command to regenerate them
in another checkout, using a new output directory.

These results demonstrate the core procedure on Cora with the thesis pipeline.
A zero sampled barrier does not prove a zero barrier between evaluation points.
They do not establish numerical reproduction of the original paper.

## Limits

Only Cora and CPU execution were exercised. Other datasets supported by the
thesis loader can be selected by CLI, but have not been verified here. GPU
execution, wider architecture and synthetic-graph sweeps, convergence studies,
and exact numerical agreement with the paper remain unverified.

The parent loader emits an existing tensor-copy warning when constructing
class weights. These experiments ignore class weights and use unweighted
mean cross-entropy. No parent-loader files were changed.
