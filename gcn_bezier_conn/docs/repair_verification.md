# GCN REPAIR verification

Date: 2026-09-18. Branch: `exp/gnn-repair`. Worktree:
`/home/wge3/ms_thesis/.worktrees/gnn-repair`.

CPU execution used `/home/wge3/miniconda3/envs/py312/bin/python`:
Python 3.14.0, PyTorch 2.9.0+cu128, PyG 2.7.0, Lightning 2.5.5.
No packages or machine configuration changed.

## Commands and checks

Equivalent commands from the worktree's `gcn_bezier_conn` directory:

```bash
PY=/home/wge3/miniconda3/envs/py312/bin/python
$PY -m unittest discover -s tests -v
$PY -m gcn_mc repair \
  --source /home/wge3/ms_thesis/gcn_bezier_conn/runs/cora/report.json \
  --thesis-root /home/wge3/ms_thesis \
  --output runs/cora-repair \
  --device cpu
```

Verification ran unittest discovery, the CLI's `main()` function, and checkpoint
replay in one interpreter to avoid repeated cold imports. All 11 tests passed
in 0.269 seconds with one PyTorch CPU thread. Four adapter tests cover hidden
permutation recovery at two depths, function preservation, train-mask population
statistics, unchanged caller state, endpoint preservation, and equivalence of
fused REPAIR to explicit corrections after message passing on an unequal-degree
graph. The other seven tests exercise the existing GCN/Bézier procedure.

Postprocessing completed in 29.67 application seconds, excluding initial imports
and plotting. It reused the six saved endpoints and three trained Bézier controls
for pairs `0:1`, `2:3`, and `4:5`, with the original 21-point grid. No training ran.
Use a new output directory when repeating the command.

The runner verified loader provenance, graph dimensions, exact masks, checkpoint
configuration and split hashes, endpoint metrics, and the replayed raw linear and
Bézier loss curves. Every method's endpoints matched the recorded endpoint losses
within `2e-5`. Reloaded repaired midpoint checkpoints reproduced loss and accuracy
on all three splits within `1e-7`. The largest alignment logit discrepancy was
`2.861023e-6`; all pairs passed the function-preservation check. The PNG was
visually inspected and displays all four methods across all three splits.

The original report's SHA-256 remained unchanged:
`af0a541f9e179d85ed3481bbca1f0a5fa0467b57630e515460c470a91c036b04`.
A fresh loader call reproduced split SHA-256:
`6bd8cc27013b1270c884603c41a93723dd92bf413c1e43b55cf5401c9cc13a2e`.

## Measured results

Sampled mean-cross-entropy barriers, mean ± population SD across three pairs:

| Method | Train | Validation | Test |
|---|---:|---:|---:|
| Linear | 0.444850 ± 0.035857 | 0.324542 ± 0.022200 | 0.342250 ± 0.026034 |
| Aligned linear | 0.047019 ± 0.004400 | 0.046457 ± 0.010900 | 0.048018 ± 0.009413 |
| Aligned + REPAIR | 0.030480 ± 0.003755 | 0.032748 ± 0.010509 | 0.033269 ± 0.008544 |
| Bézier | 0.000000 ± 0.000000 | 0.019892 ± 0.018307 | 0.002499 ± 0.003534 |

Per-pair test barriers:

| Seeds | Linear | Aligned | Repaired | Bézier |
|---|---:|---:|---:|---:|
| 0:1 | 0.375768 | 0.060062 | 0.043613 | 0.000000 |
| 2:3 | 0.312297 | 0.046907 | 0.033505 | 0.000000 |
| 4:5 | 0.338685 | 0.037085 | 0.022690 | 0.007497 |

Midpoint diagnostics, averaged across pairs:

| Method | Test accuracy | Hidden variance / weighted endpoint variance |
|---|---:|---:|
| Linear | 79.967% | 0.514170 |
| Aligned linear | 80.300% | 0.905842 |
| Aligned + REPAIR | 80.367% | 0.979164 |
| Bézier | 79.133% | 3.845566 |

Alignment accounts for most of the observed barrier reduction. REPAIR reduces
the mean test barrier a further 30.7% relative to aligned linear interpolation
and brings the midpoint hidden variance closer to its endpoint baseline. Bézier
has the lowest loss barrier but lower midpoint test accuracy in this small sample.
REPAIR targets weighted standard deviations; its diagnostic variance ratio need
not equal one.

## Artifacts and limits

Saved [report](../runs/cora-repair/report.json),
[PNG](../runs/cora-repair/Cora/connectivity.png), and
[PDF](../runs/cora-repair/Cora/connectivity.pdf). The result directory includes
copied endpoint/control/split checkpoints and new aligned endpoint and repaired
midpoint checkpoints. Generated artifacts are ignored by git.

This initial verification exercised Cora on CPU. The subsequent
[four-dataset sweep](dataset_sweep.md) adds Roman-empire, squirrel, and chameleon.
Three pairs and a sampled grid do not establish
generalization or exact reproduction of either paper. Calibration uses training
node rows in full-graph transductive forwards. The repaired path is not a straight
line in parameter space, so these results do not establish linear mode connectivity.

GPU execution, further datasets and architectures, and REPAIR on Bézier points remain
deferred. The existing loader's class-weight tensor-copy warning remains; this
experiment uses unweighted cross-entropy. See the
[integration guide](repair_integration.md) for the method and replay policy.
