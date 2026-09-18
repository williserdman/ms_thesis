# GCN connectivity implementation plan

> Use subagent-driven implementation for independent tasks. User instructions
> limit verification to the critical path and exclude code-review cycles.

**Goal:** Run the paper's core GCN connectivity procedure through the thesis loader.

**Architecture:** A thin data adapter supplies a full PyG graph. A GCN and
differentiable parameter paths support endpoint training and curve fitting.
A CLI writes checkpoints, measured curves, summaries, and plots.

**Tech stack:** Python, PyTorch, PyG, existing thesis loader, matplotlib.

**Spec:** `../specs/2026-09-17-gcn-connectivity-design.md`

## Global constraints

- Work only under `gcn_bezier_conn`; use the parent loader without rewriting it.
- Reuse the current environment. No package or machine-configuration changes.
- Keep endpoint weights fixed and fit controls using training labels only.
- Preserve parent edits. No unsolicited review cycle, refactoring, or sweeps.

## Tasks

- [x] Inspect the parent pipeline and locate the original paper.
- [x] Record the paper protocol, equations, defaults, and replication gaps.
- [x] Implement `gcn_mc/data.py`: canonical names, public loader call, full graph,
  split metadata, and mask validation. Use `load_graph(name, thesis_root,
  data_seed)` returning a graph and metadata.
- [x] Implement `gcn_mc/model.py` and `gcn_mc/paths.py`, with focused behavioral
  tests first. Use `GCN(in_channels, hidden_channels, out_channels, depth,
  dropout)` and parameter dictionaries compatible with `torch.func.functional_call`.
  Compare path outputs with independent endpoint predictions. Backpropagate a
  masked cross-entropy loss and confirm control updates preserve endpoint states.
- [x] Implement `gcn_mc/experiment.py` and `gcn_mc/__main__.py`: load once, train
  seeds, select checkpoints by validation loss, fit curves, evaluate all splits,
  save checkpoints and reports. Defaults follow verified protocol where stated.
- [x] Implement `gcn_mc/plotting.py`, README, and dependency manifest. Plot saved
  reports without retraining.
- [x] Run focused tests and one short Cora experiment, reload artifacts, and
  record exact results and limitations in `docs/verification.md`.
- [x] Write `ONBOARDING.md` with tested commands, extension points, and deferred
  work. Check documentation commands against the actual CLI.

## Execution notes

Use the existing empty target directory for file isolation. A clean worktree
would omit the user's current loader changes. No commit, merge, or publication
is needed to make this local result runnable and reviewable.
