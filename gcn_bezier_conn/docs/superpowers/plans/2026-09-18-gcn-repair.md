# GCN REPAIR implementation plan

Use subagent implementation for the graph adapter while the main agent adds the
saved-run comparison command. User instructions exclude review cycles and broad
hardening. Changes stay in the isolated worktree.

**Goal:** Compare GCN interpolation with alignment and activation REPAIR.

**Architecture:** A GCN adapter supplies alignment, masked hidden statistics, and
sequential calibration. A postprocessing command reuses existing checkpoints and
produces reports compatible with the current path plotting structure.

**Tech stack:** PyTorch, PyG, SciPy, matplotlib, sibling REPAIR source.

**Spec:** `../specs/2026-09-18-gcn-repair-design.md`

## Tasks

- [x] Create worktree, copy existing source, inspect both REPAIR handoff documents.
- [x] Verify the copied GCN baseline with the existing seven tests.
- [x] Add `gcn_mc/repair_adapter.py` and `tests/test_repair_adapter.py`.
  Interface: `align_gcn(reference, candidate, graph)` returns an aligned copy and
  JSON-compatible alignment diagnostics; `repair_gcn(reference, aligned,
  graph, alpha)` returns a repaired copy and JSON-compatible layer diagnostics.
  `hidden_statistics(model, graph)` returns per-hidden-layer `(mean, std)` tensors
  from training-node preactivations. Models supplied by callers remain unchanged.
  Use focused tests first for permutation invariance, graph fusion, masks, and
  preserved endpoint states. Run only these tests during adapter development.
- [x] Add `gcn_mc/repair_experiment.py` and a `repair` CLI command.
  Accept a source report, `--thesis-root`, `--output`, and `--device`. Load and
  validate saved endpoint states and masks, evaluate the existing grid, retain
  the original linear/Bézier curves, and add aligned/repaired curves plus variance
  diagnostics. Save aligned B and a fused repaired midpoint for each pair.
- [x] Extend report plotting to the new method names without changing existing
  plot behavior. Add SciPy to project dependencies; reuse the installed environment.
- [x] Run adapter tests and the saved three-pair Cora comparison, verify artifact
  replay, and inspect the plot. Record measured results.
- [x] Update README and onboarding with the command, graph-specific semantics,
  worktree/data locations, and deferred Bézier calibration and architecture work.

## Decisions

The first extension calibrates linear interpolants and treats the trained Bézier
path as a comparator. It does not retrain endpoints, refit curves, or modify the
original reports. New output directories protect the baseline artifacts.
