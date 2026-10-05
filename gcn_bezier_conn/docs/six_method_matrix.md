# Alignment, Bézier, and REPAIR across all datasets

The September 29 extension compares six paths for GCN, MLP, GraphSAGE, and GAT
on all 15 thesis-loader datasets. Each configuration uses seeds `0:1`, `2:3`,
and `4:5`, 200 optimizer steps per Bézier control, and 21 evaluation points.

| Report key | Path |
|---|---|
| `linear` | Original endpoint linear interpolation |
| `bezier` | Original endpoint quadratic Bézier |
| `aligned` | Linear interpolation after aligning endpoint B to A |
| `repaired` | Posthoc REPAIR of aligned linear samples |
| `aligned_bezier` | Separately fitted Bézier between A and aligned B |
| `repaired_bezier` | Posthoc REPAIR of aligned Bézier samples |

Endpoint training precedes alignment. The aligned control minimizes sampled
training cross entropy with the same optimizer settings and pair seed as the raw
control. REPAIR then corrects each interior sample using weighted endpoint means
and standard deviations. REPAIR does not participate in curve optimization.
Native BatchNorm buffers are calibrated once before correction and frozen while
corrections are fitted. Matching and correction use training-node activations;
full-graph message passing remains transductive. Endpoints remain exact copies.

Cora, Roman-empire, squirrel, and chameleon reuse the 16 completed tuned runs
under `runs/tuned-endpoints-20260918/`, including their original raw Bézier
controls. Seven other datasets use pinned source profiles, and actor, texas,
cornell, and Tolokers use documented compact local defaults. MLP inherits each
dataset's GCN profile with linear operators. See
[extended profiles](extended_dataset_profiles.md). This mixes tuned, source,
and local settings; cross-dataset results must retain those labels. New endpoint
tuning and matching the upstream binary ROC AUC objective are deferred.

The original four-method repair command remains available. Enable all six with:

```bash
python -m gcn_mc repair --include-bezier \
  --source runs/tuned-endpoints-20260918/Cora/gcn/report.json \
  --thesis-root /home/wge3/ms_thesis --device cuda --threads 2 \
  --output runs/example-six-methods
```

Submit the full matrix from this project directory:

```bash
sbatch --array=0-59%4 scripts/six_method_matrix.sbatch \
  runs/six-methods-20260929 runs/tuned-endpoints-20260918
```

The array trains only the 44 configurations without existing tuned endpoints.
Every task uses one GPU. Reports, weights, calibration diagnostics, variance
curves, training histories, and PNG/PDF plots remain under the run root. Each
pair saves the aligned endpoint and aligned control, plus both repaired midpoint
checkpoints. The runner reloads both midpoints and compares them against their
recorded path metrics. Raw source curves are checked before retaining the saved
metrics verbatim. All original results remain in place.

After the array, `scripts/archive_six_method_matrix.py RUN_ROOT OUTPUT` audits
completion and writes compact reports, tables, and per-architecture graphics.
The archive records missing or failed configurations and exits nonzero if the
matrix is incomplete.
