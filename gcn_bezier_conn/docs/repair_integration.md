# REPAIR integration for GNN connectivity

This postprocessing experiment asks how much of a sampled interpolation barrier is associated with hidden-channel ordering and activation-variance collapse. It compares four methods on the same trained endpoint pairs, graph, masks, and interpolation grid:

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

The source must be an original schema-v1 connectivity report for legacy GCN or reference GCN, MLP, GraphSAGE or GAT. The output directory must be new or empty. The command does not train endpoints or refit Bézier controls. It loads the original checkpoints, reproduces the raw linear and Bézier curves, and computes the aligned and repaired curves. Add `--no-plots` to skip plot generation or `--threads N` to change the default single CPU thread.

The source report and its artifacts remain unchanged. The new result directory copies the endpoint, Bézier-control, and split checkpoints needed for self-contained replay, then adds aligned and repaired artifacts.

## Provenance checks

Postprocessing stops rather than mixing incompatible artifacts. It checks:

- the source report schema;
- model settings in every endpoint and curve checkpoint;
- checkpoint split hashes and endpoint-pair identities;
- the current thesis loader hash, graph dimensions, class and feature counts, node count, and edge count;
- each saved train, validation, and test mask against the graph returned by the current loader;
- replayed endpoint losses against the source report;
- recomputed raw linear and Bézier losses against the source values within `2e-5`;
- source-path accuracy within one correctly classified node per split, with any
  replay differences recorded. Original source-path metrics remain the comparator.

The new report records the source report's SHA-256 for provenance. Use the original thesis checkout's loader and cached datasets. The worktree has its own tracked files, but this command reads the original run and copies its checkpoints into the new result directory.

## Legacy GCN alignment

For each hidden layer, `align_gcn` captures post-ReLU activations from both endpoints. It forms channel correlations using training-node rows, applies the baseline's regularized correlation calculation, and solves the assignment with the Hungarian algorithm.

The assignment permutes hidden feature channels, not graph nodes. For a hidden `GCNConv`, it permutes the output rows of `conv.lin.weight` and the matching outer `conv.bias`, then permutes the input columns of the following convolution's projection weight. The final class channels are never permuted. The aligned endpoint must reproduce endpoint B's full-graph logits within floating-point tolerance before any merge is evaluated.

Node permutations would change graph identity and are outside this method. Channel permutations only choose an equivalent parameterization of the same endpoint function.

## Legacy GCN sequential REPAIR

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

The legacy adapter remains unchanged. `gcn_mc/reference_repair.py` adds the
reference architectures using the rules below. Spectral-filter models and REPAIR
on Bézier points remain outside this analysis.

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


## Reference GCN, MLP, GraphSAGE and GAT

Use the completed tuned endpoints directly. The source report resolves all six
endpoint checkpoints and three saved Bézier controls:

```bash
$PY -m gcn_mc repair \
  --source runs/tuned-endpoints-20260918/Cora/gat/report.json \
  --output runs/repair-tuned-example/Cora/gat \
  --thesis-root /home/wge3/ms_thesis --device cuda --threads 2
```

For all sixteen configurations:

```bash
sbatch --array=0-15%4 scripts/repair_matrix.sbatch \
  runs/tuned-endpoints-20260918 runs/repair-tuned-20260918
```

The optional input projection and each hidden block have their own channel
assignment. These models use learned residual projections, so each projection
receives the same output permutation as its graph/linear branch and the same
input permutation as the preceding stage. GraphSAGE permutes both neighbor and
root projections. Single-head GAT permutes projection channels and both attention
vectors together. Normalization affine parameters and running statistics follow
the output permutation. The classifier's input columns follow the last stage;
its class outputs remain fixed. The runner checks full-graph logit invariance.
The initial check uses float32 tolerances `rtol=1e-5, atol=1e-6`. If it fails,
independent float64 copies must pass `rtol=1e-9, atol=1e-10`. This distinguishes
channel-reduction roundoff from a changed function; a failed float64 check still
stops the run. Reports retain both errors and predicted-class disagreements.

Matching uses post-ReLU activations, except for the input projection, which has
no ReLU in the reference model. Corrections act after the complete block,
including residual addition and active normalization, immediately before ReLU.
The optional input projection also receives a correction before dropout.
Calibration uses the same weighted-mean/weighted-standard-deviation equations
and epsilon as the legacy adapter, measured on training nodes in forward order.

Original linear/Bézier paths and aligned linear interiors retain the baseline's
full-graph, label-free BatchNorm recalibration. A repaired interior starts from
that calibrated aligned linear model. Its native BatchNorm buffers are then
frozen while sequential REPAIR measures training-node moments and installs
corrections. No subsequent BatchNorm recalibration cancels those corrections.
The full-graph normalization policy is inherited from the source experiment;
REPAIR itself does not select validation/test rows or use labels.

Reference corrections remain explicit channel-wise affine modules. In
particular, scaling a GAT projection would also change attention scores, so
fusing an output correction into that projection would implement a different
operation. Legacy GCN checkpoints still use fused weights. Reference midpoint
checkpoints carry `repair_format="reference-affine-v1"`; load either format with:

```python
from gcn_mc.reference_repair import load_repaired_model
model = load_repaired_model("path/to/repaired_0_1_midpoint.pt", device="cuda")
logits = model(graph.x, graph.edge_index)
```

The repaired path returns original endpoint copies at the two boundaries,
with B in its aligned parameterization. It generally leaves the original model
parameterization between endpoints because of explicit affine corrections.
This is an activation-statistics comparison, not evidence of a straight
low-loss segment in the original parameter space.


GPU graph reductions can perturb logits near a class tie. The Roman-empire GAT
probe reproduced one changed test prediction at Bézier t=0.95, with a top-two
margin of 2.86e-6 and unchanged cross-entropy. The original path evaluator showed
the same effect across repeated calls. Source-path validation therefore counts
correct predictions and permits at most one node of disagreement per split,
while retaining the strict loss check. `source_replay` records those differences;
the original linear/Bézier curves are copied after verification. Endpoint and
repaired-checkpoint replay retain their stricter accuracy checks. See the saved
[precision probe](../results/repair-tuned-20260918/source_replay_precision_probe.json).
