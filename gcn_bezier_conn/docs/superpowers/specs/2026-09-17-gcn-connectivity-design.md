# GCN mode connectivity replication

Implement the core experiment from [Li et al., 2025](https://arxiv.org/abs/2502.12608)
in `gcn_bezier_conn`. Reuse the current thesis data pipeline, including its
preprocessing and masks. Record differences from the paper explicitly.

## Approach

A small Python CLI loads one graph once, trains independently seeded GCN
endpoints on that same split, and compares straight parameter interpolation
with a learned quadratic Bézier curve. Curve training changes only the control
point, uses training labels, and differentiates through interpolated parameters.
Evaluation disables dropout and records train, validation, and test loss and
accuracy along both paths. Save endpoints, controls, configuration, split
fingerprints, raw curves, summaries, and publication-exportable plots.

The existing spectral experiments remain separate because their model and
coefficient-only interpolation do not implement the paper's GCN experiment.
Adapting author code wholesale would import another data pipeline and is outside
the requested approach. The implementation uses PyTorch and PyG with a thin
adapter around `loading.LightningGraphLoader.load_datasets`.

## Scope and constraints

- Core GCN procedure first. Wider architecture, synthetic graph, generalization,
  and domain-alignment sweeps are deferred.
- All edits stay in the existing `gcn_bezier_conn` directory. The parent has
  uncommitted loader changes needed for this task and remains read-only.
- No new data downloader, split generator, preprocessing pipeline, or shared
  environment changes. Select data through the thesis loader.
- Keep both endpoints fixed during curve fitting. Preserve gradient flow to
  every learned control parameter.
- Separate model seeds, data seed, and curve seeds. Save actual masks for replay.
- Select checkpoints without test labels. Report all evaluation splits.
- A short real-data run verifies the workflow, not the paper's empirical claims.
- Provide developer onboarding with setup, commands, file map, artifact schema,
  method limitations, and next experiments.

## Verification

Focused checks cover curve endpoint equivalence, a midpoint control reproducing
a straight line, gradients reaching the control, endpoint immutability, and
loss-barrier arithmetic. A short Cora run through the actual thesis loader must
produce reloadable checkpoints, finite metrics, JSON, and plots. Paper protocol
facts and implementation choices are recorded in `docs/paper_protocol.md`.
