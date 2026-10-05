# Handoff

Updated: 2026-10-05. Integration target: `main`.

## Current state

The user requested a handoff in this project folder, merging the completed
alignment/Bézier/REPAIR study into `main`, and cleanup of the session worktree.
The study is merged into `main`. Its session worktree and local feature branch
are removed. Raw artifacts are preserved in the permanent project folder.
The primary checkout has unrelated pending work on `exp/spectral-mc-pocs`.
Preserve that work and all other experiment worktrees.

Permanent project folder: `/home/wge3/ms_thesis/gcn_bezier_conn`.
Use `/home/wge3/miniconda3/envs/py312/bin/python`, which contains Python 3.14,
PyTorch 2.9.0+cu128 and PyG 2.7.0 despite the environment's name. No environment
or machine configuration was changed.

## Completed experiment

All 60 model/dataset configurations completed and passed the archive audit:
15 thesis-loader datasets, GCN/MLP/GraphSAGE/GAT, three seed pairs per
configuration, 21 path points, and 200 steps per Bézier control. Slurm array
`3922820` and audit `3922821` completed with exit code 0. The project suite
passed all 50 tests before and after integration; the five existing root tests
also passed. The relocated archive audit passed all 60 configurations, and
repaired linear/Bézier midpoint checkpoints replayed with maximum loss and
accuracy errors of `5.97e-8`. Verification logs are in
`runs/integration-20261005/`.

Start with [results and graphics](results/six-methods-20260929/README.md),
[summary JSON](results/six-methods-20260929/summary.json), and
[summary CSV](results/six-methods-20260929/summary.csv).
[Six-method protocol](docs/six_method_matrix.md) defines report keys and order:
train endpoints, align endpoint B, fit a separate aligned Bézier control, then
apply posthoc REPAIR to sampled models. The original raw linear and Bézier
comparators are retained. REPAIR is absent from curve optimization.

The matrix combines 16 reused endpoint-tuned configurations, 28 pinned source
profiles, and 16 explicit local fallbacks. Preserve these provenance groups.
[Extended profiles](docs/extended_dataset_profiles.md) documents the local
choices and upstream binary ROC AUC deviation. The raw weights, split masks,
training histories, calibration diagnostics and tuning databases remain under
`runs/`; compact reports and PNG/PDF plots are versioned under `results/`.
Historical result folders and the original legacy GCN runs remain valid.

## Interpretation and next work

Alignment was the most consistent improvement. Bézier training often obtains
low training barriers but high validation/test barriers. A quadratic Bézier
with its control at the endpoint midpoint reproduces the straight path, so a
worse fitted curve does not establish absence of a useful connecting path.
REPAIR can reduce cross-entropy barriers while reducing classification accuracy.
Low path barriers do not establish that an interior model beats either endpoint.
See the saved reports for the measured comparisons and checkpoint replay checks.

A suggested follow-up is to hold endpoints fixed, record train/validation path
loss throughout aligned control fitting, and select the control checkpoint using
validation loss. This experiment is proposed, not implemented or launched.
Distinguish studying whole-path connectivity from selecting one useful merged
model. Use validation for selection and reserve test metrics for evaluation.

## Artifact preservation and relocation

The session worktree was `/home/wge3/ms_thesis/.worktrees/gnn-repair`.
Its 3,400 run files and 17 Optuna studies were moved into the permanent folder.
All 342 files from the earlier legacy runs remain. Earlier legacy source is
archived under `runs/legacy-source-20261005/` for historical replay.
The temporary main integration checkout is removed after publication.

Operational `source_report` references are relocated to the permanent folder;
source baseline bytes and SHA-256 values remain unchanged. Original analysis
JSON is in `runs/relocation-originals-20261005/`; the relocation mapping is
`runs/artifact-relocation-20261005.json`. The archive audit passed after relocation. Historical cache and implementation metadata may still
record the former worktree path; treat that as provenance, not an active path.
Do not rewrite baseline reports or delete Optuna `study.sqlite3`/`best.json`.

## Read before changing behavior

- [Developer onboarding](ONBOARDING.md): commands, dependencies, code map.
- [Paper protocol](docs/paper_protocol.md): replication scope and missing details.
- [Architecture source check](docs/architecture_sources.md): pinned tunedGNN code.
- [REPAIR integration](docs/repair_integration.md): alignment and normalization.
- [Reference verification](docs/reference_repair_verification.md): numerical replay.
- Existing designs/plans under `docs/superpowers/` document implemented decisions.

Use the original thesis loader/cache with explicit
`--thesis-root /home/wge3/ms_thesis`. Its hash and split masks identify all saved
runs. Preserve unrelated dirty files. Keep changes small, reversible, and scoped.
Do not initiate review cycles, add authorship attribution, or select settings on
test metrics. Do not rerun the completed matrix merely to regenerate graphics.

## Suggested skills

- `superpowers:using-superpowers` for session setup.
- `superpowers:brainstorming` before choosing a follow-up experiment.
- `superpowers:systematic-debugging` for unexpected fitting or replay behavior.
- `superpowers:test-driven-development` for a small runner extension.
- `superpowers:verification-before-completion` before reporting results or merging.
- `unslop` for concise documentation.
