# Original architecture baselines

The user authorized implementing the original linear/Bézier comparison for GCN,
MLP, GraphSAGE, and GAT before extending REPAIR. Use the existing worktree and
thesis loader. Existing GCN runs and checkpoints remain valid.

## Models and configuration

Keep `GCN` unchanged as the legacy model. Add a `build_model(model_config)` factory
and a reference model family adapted from the pinned, MIT-licensed `tunedGNN`
source. `model_config` identifies `family`, `architecture`, dimensions, depth,
dropout, normalization, residual connections, input projection, and GAT heads.
Missing family/architecture denotes the existing legacy GCN configuration.

`run --preset reference --architecture NAME` resolves the official reference's
dataset settings. Explicit CLI overrides are applied afterward and saved in
each dataset's effective configuration. Preserve the existing default legacy
run. The reference MLP replaces message-passing blocks with linear blocks and
uses the corresponding reference GCN profile; this is an explicit choice because
no MLP recipe was recovered. Document this and all other remaining paper gaps.

The thesis loader continues to supply the graph, features, and masks. Do not copy
the upstream loaders or silently make directed graphs undirected. Document the
upstream preprocessing discrepancy. Preserve upstream licenses and provenance.

## Paths and normalization

Parameter interpolation remains generic. BatchNorm requires explicit buffer
handling: use fresh temporary buffers during control fitting, and recompute
running statistics deterministically for each evaluated path point using one
full-graph, label-free forward with dropout disabled. All BatchNorm layers use
batch statistics in that calibration pass. Use the same calibration for selected
endpoints before saving/evaluating them so path endpoints reproduce checkpoints.
This is transductive normalization over all graph nodes, not REPAIR. Record the
policy as an implementation choice absent from the mode-connectivity paper.

Calibration must preserve caller modes/parameters and avoid stale or accumulating
temporary buffers. Plain legacy GCN behavior remains unchanged. The curve learns
only its control parameters on train-mask loss. Dataset labels never enter
normalization calibration.

## Experiment and output

Reference endpoint selection follows validation accuracy, with the upstream
training budget and optimizer settings. Legacy selection remains validation
loss. Record pre-calibration and calibrated endpoint metrics if they differ.
Continue the current three independent seed pairs and 21-point evaluation grid.
The curve optimizer settings remain explicit choices because the paper does not
provide them. Reports retain enough model and normalization metadata for replay.

Plots name the actual architecture and configuration. REPAIR rejects reference
models and non-GCN architectures before creating output. Existing GCN REPAIR
reports remain readable.

## Verification and scope

Focused tests cover source-model equivalence, MLP edge independence, path endpoints,
control gradients and frozen endpoints, BatchNorm state isolation/calibration,
and checkpoint replay. Run end-to-end smoke checks for every architecture and
full-budget runs where the measured pilot runtime permits. Record any pilots
separately from completed full-budget results. No new REPAIR adapters, data
pipelines, synthetic-graph experiments, or broad tuning sweep in this phase.
