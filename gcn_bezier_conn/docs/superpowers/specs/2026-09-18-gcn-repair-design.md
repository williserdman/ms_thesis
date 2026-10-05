# REPAIR for GCN connectivity

Add a graph-specific REPAIR comparison to the existing GCN study, using the
existing trained endpoints and thesis loader. Read the baseline
[`repair/ONBOARDING.md`](../../../../repair/ONBOARDING.md) and
[`repair/handoff.md`](../../../../repair/handoff.md) before changing the method.

## Scientific question

How much of the sampled GCN interpolation loss barrier is associated with hidden
channel ordering and activation variance loss? Compare raw linear interpolation,
aligned linear interpolation, aligned linear plus REPAIR, and the existing learned
Bézier curve using identical checkpoint pairs, graph, masks, and evaluation grid.

REPAIR corrects interpolated hidden activation means and standard deviations.
It changes the parameter path, so improvements cannot establish straight-line
mode connectivity. Applying REPAIR to Bézier points is a separate extension and
is deferred from this first comparison.

Sources: [REPAIR paper](https://arxiv.org/abs/2211.08403),
[GCN connectivity paper](https://arxiv.org/abs/2502.12608).

## Minimum implementation

- Support this project's sequential PyG `GCN`, including its configurable depth.
  No attention, residual, normalization, or spectral-filter adapters.
- Reuse `repair.core.interpolate` from the sibling REPAIR source. Implement the
  necessary graph-specific alignment, statistics, and fusion separately. Do not
  change the validated MLP/VGG implementation or recreate data pipelines.
- Match hidden channels by post-ReLU correlations on training nodes, using the
  same regularized correlation and Hungarian assignment as the REPAIR baseline.
- Permute each hidden convolution's output weight rows and outer bias, and the
  following convolution's input weight columns. Do not permute nodes or classes.
  Verify full-graph endpoint logits remain equal within floating-point tolerance.
- Measure hidden preactivations at the complete `GCNConv` output, after message
  passing and bias, before ReLU. Never calibrate the inner `GCNConv.lin` output.
- Collect moments on training nodes only, with dropout disabled. The forward
  pass still uses the full graph, as in the existing transductive study.
- Correct hidden layers sequentially to weighted endpoint means and standard
  deviations. Use the baseline's epsilon `1e-5`. Keep exact endpoint models at
  positions zero and one. Exclude the final class convolution.
- Fuse output scaling into the inner projection weight rows and the outer
  convolution bias. Adding an inner projection bias would be degree-scaled by
  propagation and is not equivalent.
- Save aligned and repaired checkpoints, channel permutations, training-node
  variance diagnostics, train/validation/test curves and barriers, and plots.
- A postprocessing command reads an existing `report.json`; no retraining or
  new splits. Verify loader and split identity before reusing checkpoints.

## Workspace

Branch `exp/gnn-repair`, worktree `/home/wge3/ms_thesis/.worktrees/gnn-repair`.
The existing untracked GCN and REPAIR source files were copied and hash-checked.
The main checkout remains on `exp/spectral-mc-pocs` with its edits intact.
Use `--thesis-root /home/wge3/ms_thesis` to reuse the current loader and data.
Existing Cora artifacts live at `/home/wge3/ms_thesis/gcn_bezier_conn/runs/cora`.

## Verification

Focused tests must prove channel permutations preserve logits, fused graph
corrections equal corrections after aggregation on a graph with unequal node
degrees, endpoint parameters remain unchanged, and calibration uses training
nodes only. Verify corrected moments with epsilon accounted for. Run the saved
three-pair Cora study and reload a repaired checkpoint. Report observed results
without assuming REPAIR improves accuracy or loss.
