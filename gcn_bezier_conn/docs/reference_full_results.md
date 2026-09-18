# Full reference-preset results

These 16 runs are fixed reference-preset comparators for four architectures and four datasets. They use three independent seed pairs per configuration. They do not include Optuna endpoint search and should not be described as tuned results.

This is not an exact full-paper reproduction. The paper does not specify every training, split, checkpoint-selection, or curve-optimization setting. The reports preserve the recovered source profile and this project's explicit choices.

## Verification

All 16 expected reports passed the report audit. The audit checked source profiles and provenance, six endpoints, three pairs, endpoint-inclusive 21-point grids, 200 curve steps, finite metrics, split and loader hashes, earliest maximum-validation-accuracy endpoint selection, and reported endpoint-to-path agreement. This was a report/split-artifact audit; it did not recompute all 16 runs from model checkpoints.

The largest endpoint-to-path loss deviation was `3.57627869e-07`. The largest accuracy deviation was `0`. Fixed tolerances were `2e-05` for loss and `1e-06` for accuracy.

## Results

Barrier values are mean cross-entropy excess above the endpoint chord. Lower is better. Values are population mean +/- standard deviation across three seed pairs.

| Dataset | Architecture | Endpoint epochs | Endpoint test accuracy | Linear test barrier | Bezier test barrier | Bezier train barrier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora | gcn | 500 | 0.788 +/- 0.007 | 0.631 +/- 0.039 | 2.643 +/- 0.343 | 0.000 +/- 0.000 |
| Cora | mlp | 500 | 0.518 +/- 0.026 | 0.000 +/- 0.000 | 2.989 +/- 0.667 | 0.000 +/- 0.000 |
| Cora | graphsage | 500 | 0.782 +/- 0.007 | 0.157 +/- 0.050 | 2.245 +/- 0.464 | 0.000 +/- 0.000 |
| Cora | gat | 500 | 0.795 +/- 0.006 | 0.459 +/- 0.096 | 5.330 +/- 1.752 | 0.047 +/- 0.065 |
| Roman-empire | gcn | 2500 | 0.846 +/- 0.003 | 3.541 +/- 0.117 | 0.028 +/- 0.013 | 0.478 +/- 0.020 |
| Roman-empire | mlp | 2500 | 0.652 +/- 0.001 | 0.993 +/- 0.041 | 0.291 +/- 0.035 | 0.329 +/- 0.037 |
| Roman-empire | graphsage | 2500 | 0.841 +/- 0.004 | 1.808 +/- 0.123 | 0.000 +/- 0.000 | 0.553 +/- 0.031 |
| Roman-empire | gat | 2500 | 0.761 +/- 0.014 | 1.035 +/- 0.138 | 0.000 +/- 0.000 | 0.666 +/- 0.128 |
| squirrel | gcn | 500 | 0.412 +/- 0.020 | 0.000 +/- 0.000 | 4.988 +/- 0.229 | 0.000 +/- 0.000 |
| squirrel | mlp | 500 | 0.386 +/- 0.010 | 0.006 +/- 0.003 | 3.910 +/- 0.167 | 0.000 +/- 0.000 |
| squirrel | graphsage | 500 | 0.397 +/- 0.011 | 0.004 +/- 0.005 | 5.096 +/- 0.097 | 0.000 +/- 0.000 |
| squirrel | gat | 500 | 0.412 +/- 0.019 | 0.029 +/- 0.035 | 5.108 +/- 1.016 | 0.000 +/- 0.000 |
| chameleon | gcn | 200 | 0.389 +/- 0.022 | 0.000 +/- 0.000 | 6.022 +/- 2.472 | 0.142 +/- 0.102 |
| chameleon | mlp | 200 | 0.380 +/- 0.019 | 0.000 +/- 0.000 | 71.453 +/- 38.972 | 0.024 +/- 0.034 |
| chameleon | graphsage | 200 | 0.424 +/- 0.024 | 0.037 +/- 0.009 | 5.064 +/- 0.109 | 0.000 +/- 0.000 |
| chameleon | gat | 200 | 0.421 +/- 0.028 | 0.000 +/- 0.000 | 3.909 +/- 0.497 | 0.000 +/- 0.000 |

The summary JSON contains exact endpoint distributions, validation accuracy, selected epochs, train/validation/test barriers, minimum path accuracy, midpoint metrics, hashes, and per-run audit results. Each result directory also contains the full report and its PNG/PDF connectivity plot.
