# Architecture baseline implementation plan

**Goal:** Run the paper's four model families through linear/Bézier experiments.

**Architecture:** Reuse the thesis loader and current runner. Add a reference model
factory/presets, explicit BatchNorm path calibration, and architecture metadata.

**Tech stack:** Existing PyTorch/PyG environment; pinned MIT-licensed tunedGNN source.

**Spec:** `../specs/2026-09-18-architecture-baselines-design.md`

## Constraints

Work in the existing `exp/gnn-repair` worktree. Preserve legacy GCN checkpoints,
results, and REPAIR semantics. Keep thesis graph preprocessing and masks. No new
packages, machine changes, review cycles, or REPAIR extensions.

## Tasks

- [x] Confirm and save reference sources/license and exact profiles under
  `upstream/tunedGNN`; update `docs/architecture_sources.md`.
- [x] Add `build_model(model_config)` and reference model implementations in
  `gcn_mc/model.py` or `reference_models.py`, retaining the legacy `GCN`.
  Add `reference_profile(dataset, architecture)` in `gcn_mc/presets.py` returning
  model/training defaults and provenance. Verify real predictions against the
  pinned upstream models and MLP edge independence.
- [x] Extend `gcn_mc/paths.py` with isolated BatchNorm buffer handling and export
  `calibrate_batchnorm(model, graph)` for endpoint calibration. Test deterministic
  endpoints, control gradients, unchanged template buffers/modes, and held-out
  label independence with a tiny real graph/model before implementation.
- [x] Extend `Config`, CLI, and runner with architecture/reference preset selection,
  explicit overrides, effective per-dataset settings, calibrated checkpoints,
  model provenance, and architecture-aware plots. Reject unsupported REPAIR input.
  Preserve old commands and reports.
- [x] Run focused tests, all-model smoke checks, and timed compute pilots. Run
  feasible full-budget comparisons, verify saved endpoints/control replay, and
  inspect labeled figures. Record actual scope and limitations.
- [x] Update onboarding/handoff/results.

Publish on the authorized `exp/gnn-repair` branch and verify remote HEAD before
reporting the push as complete.

Verification: 25 tests; 16 completed GPU smoke runs; four Cora source-budget
single-pair runs with saved-checkpoint/control replay. See
`../../architecture_verification.md` for the numerical smoke caveat and
`../../../results/reference-architectures/README.md` for results. Full-budget
Roman-empire/squirrel/chameleon and three-pair reference comparisons are deferred
until curve-training choices are reevaluated. No new REPAIR adapters were added.

Independent agents own reference source recovery, models/profiles, and path
calibration. The main agent owns runner/CLI integration and end-to-end verification.
Use focused test-first checks for mathematical behavior; no broad review cycles.
