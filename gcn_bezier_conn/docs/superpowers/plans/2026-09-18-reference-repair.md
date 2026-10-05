# Reference alignment and REPAIR implementation plan

**Goal:** Add alignment and REPAIR comparisons to every saved tuned endpoint pair.

**Architecture:** Extend the current postprocessing runner with a reference-model
adapter. Keep baseline training, source checkpoints and cached studies unchanged.

**Spec:** [Reference REPAIR design](../specs/2026-09-18-reference-repair-design.md)

- [x] Add `gcn_mc/reference_repair.py` and focused tests for model permutations,
  sequential explicit affine corrections and replayable wrapper checkpoints.
- [x] Extend `repair_experiment.py` to build reference models, preserve original
  BatchNorm path semantics, check source replay, and record new provenance.
  Replace the old reference-model rejection test with real four-model replay.
- [x] Add `scripts/repair_matrix.sbatch`, generate all 16 postprocessed runs from
  the saved tuned matrix, and verify every task succeeds and all 48 pairs exist.
- [x] Audit saved midpoint checkpoints and report agreement; archive reports,
  labeled four-method plots and a matrix summary under `results/`.
- [x] Update onboarding, integration notes and handoff, then commit and push.

Use focused tests and the existing interpreter. Do not launch review cycles,
change the data pipeline, rerun Optuna, or refit endpoints/Bézier controls.
