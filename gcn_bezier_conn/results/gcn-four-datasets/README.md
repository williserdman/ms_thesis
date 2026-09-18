# Saved GCN results

Tracked copies of the completed four-dataset GCN comparison, recorded 2026-09-18.
See [the run record](../../docs/dataset_sweep.md) for settings, interpretation,
verification, and original artifact locations.

- [Overview PNG](comparison.png) and [PDF](comparison.pdf).
- Each dataset directory contains its full loss/accuracy PNG/PDF and JSON report.
- `verified_summary.json` records the cross-dataset summary and replay errors.

Reports retain original provenance and checkpoint paths. Model weights, raw data,
and full run directories are not included in this snapshot. Plotting a report
does not require weights; replaying predictions does. The original checkpoints
remain in the local `runs/` directories documented in the run record.
