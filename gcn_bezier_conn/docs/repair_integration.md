# REPAIR integration for GCN connectivity

This postprocessing experiment asks how much of a sampled GCN interpolation barrier is associated with hidden-channel ordering and activation-variance collapse. It compares four methods on the same trained endpoint pairs, graph, masks, and interpolation grid:

1. raw linear interpolation;
2. linear interpolation after aligning endpoint B's hidden channels to endpoint A;
3. the aligned linear path followed by REPAIR;
4. the saved quadratic Bézier path from the original GCN experiment.

The Bézier curve is a comparator. This integration applies REPAIR only to the aligned linear path.

REPAIR matches each interpolated hidden channel's activation mean and standard deviation to the weighted endpoint values. The method and equations come from the [REPAIR paper](https://arxiv.org/abs/2211.08403). The generic interpolation behavior follows the authors' [official source protocol at commit `e90263d`](https://github.com/KellerJordan/REPAIR/tree/e90263d7a4d48376091327274ae541d8d6d34743), especially the [VGG11 merge and REPAIR notebook](https://github.com/KellerJordan/REPAIR/blob/e90263d7a4d48376091327274ae541d8d6d34743/notebooks/Train-Merge-REPAIR-VGG11.ipynb). Graph-specific alignment, statistics, and fusion live in this project rather than changing the validated MLP/VGG package.

## Run

Run from this worktree's `gcn_bezier_conn` directory:

```bash
cd /home/wge3/ms_thesis/.worktrees/gnn-repair/gcn_bezier_conn
/home/wge3/miniconda3/envs/py312/bin/python -m gcn_mc repair \
  --source /home/wge3/ms_thesis/gcn_bezier_conn/runs/cora/report.json \
  --thesis-root /home/wge3/ms_thesis \
  --output runs/cora-repair \
  --device cpu
```

The source must be an original schema-v1 GCN connectivity report. The output directory must be new or empty. The command does not train endpoints or refit Bézier controls. It loads the original checkpoints, reproduces the raw linear and Bézier curves, and computes the aligned and repaired curves. Add `--no-plots` to skip plot generation or `--threads N` to change the default single CPU thread.

The source report and its artifacts remain unchanged. The new result directory copies the endpoint, Bézier-control, and split checkpoints needed for self-contained replay, then adds aligned and repaired artifacts.

## Provenance checks

Postprocessing stops rather than mixing incompatible artifacts. It checks:

- the source report schema;
- model settings in every endpoint and curve checkpoint;
- checkpoint split hashes and endpoint-pair identities;
- the current thesis loader hash, graph dimensions, class and feature counts, node count, and edge count;
- each saved train, validation, and test mask against the graph returned by the current loader;
- replayed endpoint losses against the source report;
- recomputed raw linear and Bézier losses against the source values within `2e-5`.

The new report records the source report's SHA-256 for provenance. Use the original thesis checkout's loader and cached datasets. The worktree has its own tracked files, but this command reads the original run and copies its checkpoints into the new result directory.

## Alignment

For each hidden layer, `align_gcn` captures post-ReLU activations from both endpoints. It forms channel correlations using training-node rows, applies the baseline's regularized correlation calculation, and solves the assignment with the Hungarian algorithm.

The assignment permutes hidden feature channels, not graph nodes. For a hidden `GCNConv`, it permutes the output rows of `conv.lin.weight` and the matching outer `conv.bias`, then permutes the input columns of the following convolution's projection weight. The final class channels are never permuted. The aligned endpoint must reproduce endpoint B's full-graph logits within floating-point tolerance before any merge is evaluated.

Node permutations would change graph identity and are outside this method. Channel permutations only choose an equivalent parameterization of the same endpoint function.

## Sequential graph REPAIR

`repair_gcn` first creates an aligned linear interpolation with the sibling package's `repair.core.interpolate`. It calibrates every hidden convolution in forward order and excludes the final classifier.

For hidden channel `j` at interpolation coefficient `alpha`, let the endpoint preactivation moments on training nodes be `(mu_a, sigma_a)` and `(mu_b, sigma_b)`. The targets are

```text
mu_target    = (1 - alpha) mu_a    + alpha mu_b
sigma_target = (1 - alpha) sigma_a + alpha sigma_b
```

These are weighted standard deviations, not weighted variances. For the current interpolated layer, population moments `(mu, sigma)` give

```text
scale = sigma_target / sqrt(sigma^2 + 1e-5)
shift = mu_target - scale * mu
```

Statistics are exact population moments accumulated over the selected node rows. Calibration is sequential: the implementation measures a layer after corrections to earlier layers are already installed. Dropout is disabled.

The measured preactivation is the complete `GCNConv` output after normalized message passing and its outer bias, immediately before ReLU. Measuring `GCNConv.lin` would omit propagation and the outer bias, so it is not equivalent.

The affine correction `scale * z + shift` is fused into the convolution as

```text
conv.lin.weight rows <- scale * conv.lin.weight rows
conv.bias            <- scale * conv.bias + shift
```

Only the outer `GCNConv.bias` receives the shift. Adding a bias to the inner projection would send that constant through graph propagation and degree normalization, changing the function.

At `alpha=0` and `alpha=1`, the method returns endpoint copies without calibration. Endpoint B is in its aligned, functionally equivalent parameterization at the right endpoint.

## Transductive policy

Each forward pass uses the full graph because the existing node-classification experiment is transductive. Only rows selected by `train_mask` enter channel matching and moment accumulation. Validation and test nodes affect training-node representations through message passing, but their rows, labels, and metrics do not set alignment or REPAIR statistics. Labels are used only when the runner evaluates train, validation, and test loss and accuracy.

## Output

The new directory contains:

```text
runs/cora-repair/
├── config.json
├── report.json
└── Cora/
    ├── split.pt
    ├── endpoint_<seed>.pt
    ├── curve_<seed-a>_<seed-b>.pt
    ├── aligned_<seed-a>_<seed-b>.pt
    ├── repaired_<seed-a>_<seed-b>_midpoint.pt
    ├── connectivity.png
    └── connectivity.pdf
```

`report.json` records the source path and hash, source and current environments, loader and split provenance, model configuration, REPAIR source and adapter hashes, channel assignments, per-grid calibration diagnostics, and raw curves for all four methods. Each method has train, validation, and test loss, accuracy, sampled loss barrier, barrier location, and minimum accuracy. Pair summaries report the mean and population standard deviation.

Hidden-variance diagnostics store each layer's mean channel variance, its alpha-weighted endpoint-variance baseline, and their ratio. A null ratio means the baseline variance was zero. The saved repaired checkpoint is the midpoint; other repaired grid points can be reconstructed from the source endpoints, saved alignment, graph, and coefficient.

The report uses the same sampled barrier definition and grid as the source experiment. A sampled maximum can miss a barrier between grid points.

## Code map

- `gcn_mc/repair_adapter.py` defines `align_gcn`, `repair_gcn`, and `hidden_statistics` for the sequential PyG `GCN`.
- `gcn_mc/repair_experiment.py` verifies source provenance, replays checkpoints, evaluates the four methods, and writes the new report.
- `gcn_mc/__main__.py` exposes the `repair` subcommand.
- `../repair/src/repair/core.py` supplies generic parameter interpolation. Its existing MLP/VGG behavior is unchanged.

The adapter supports this project's configurable sequential GCN only. Attention, residual branches, normalization layers, spectral-filter models, REPAIR on Bézier points, and other architectures require separate alignment and fusion rules.

## Interpretation

Lower barriers after alignment isolate sensitivity to a function-preserving hidden-channel ordering. A further reduction after REPAIR is consistent with activation variance collapse contributing to the sampled barrier. Neither comparison proves a causal decomposition of the loss landscape.

The repaired curve is not a straight parameter-space line because its fused corrections depend on `alpha` and measured activations. It therefore cannot establish linear mode connectivity. REPAIR also provides no guarantee that loss decreases or accuracy improves. Report measured results without an accuracy or novelty claim.

## Current verification status

All 11 focused tests passed. The three-pair Cora comparison completed on CPU, and reloaded repaired midpoint checkpoints reproduced the saved metrics. Mean test loss barriers were 0.342250 for raw linear, 0.048018 for aligned linear, 0.033269 for aligned plus REPAIR, and 0.002499 for Bézier. See the [verification record](repair_verification.md) for uncertainty, replay checks, artifacts, and limits.

Development is isolated on branch `exp/gnn-repair` in `/home/wge3/ms_thesis/.worktrees/gnn-repair`. The main checkout and its current branch remain unchanged.

The subsequent [four-dataset sweep](dataset_sweep.md) completed Roman-empire,
squirrel, and chameleon with the same GCN settings. It records mixed REPAIR
results, weak Roman-empire endpoints, and Bézier overfitting on the filtered
datasets; the Cora result alone should not be generalized to those graphs.
