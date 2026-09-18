# Architecture baseline verification

Verification date: 2026-09-18.

## Focused tests

The saved log at
[`runs/reference-smoke-20260918/tests.log`](../runs/reference-smoke-20260918/tests.log)
records 25 tests passed in 1.104 seconds. These tests cover source-model
equivalence, reference profiles, MLP edge independence, BatchNorm calibration
and state isolation, path gradients and endpoints, checkpoint replay, training
label isolation, and the legacy GCN REPAIR adapter.

## Smoke matrix

Slurm array job `3834723` ran all 16 combinations of four datasets and four
architectures. Tasks `0` through `15` completed with exit code `0:0`. Task
elapsed times ranged from 42 to 116 seconds.

Each run contains a nonempty `config.json`, `report.json`, split checkpoint, two
endpoint checkpoints, curve checkpoint, PNG, and PDF. All path metrics are
finite. No report contains a REPAIR result. REPAIR remains limited to legacy
GCN.

The smoke transform ran after profile resolution. Every effective run used
width 8, 5 endpoint epochs, 5 curve epochs, 5 path points, and the single seed
pair `0:1`. Explicit overrides were empty. Source depth, dropout, normalization,
residual, input projection, optimizer settings, and one-head GAT configuration
match the pinned tunedGNN profiles. The MLP uses the corresponding GCN profile
with linear operators, which is a local choice because tunedGNN has no MLP
recipe.

The four architectures have identical loader hashes, graph counts, split counts,
and split hashes within each dataset:

| Dataset | Nodes | Edges | Train/val/test | Split SHA-256 |
|---|---:|---:|---:|---|
| Cora | 2,708 | 10,556 | 140 / 500 / 1,000 | `6bd8cc27013b1270c884603c41a93723dd92bf413c1e43b55cf5401c9cc13a2e` |
| Roman-empire | 22,662 | 65,854 | 11,331 / 5,665 / 5,666 | `a3a172af36f96a338161be357714d68c9cd31cfeeca5e0cf6ec74ae2412772c2` |
| squirrel | 2,223 | 65,718 | 1,080 / 700 / 443 | `c939f177af32b334543763c465dd3c5881a70bca01f1bc82f25e20c122709210` |
| chameleon | 890 | 13,584 | 427 / 302 / 161 | `3df216c1db9ef1083f86fddedfe6d221b50606be1cf7f8e7c594c67648c93456` |

The machine-readable audit is
[`results/reference-architectures/smoke_summary.json`](../results/reference-architectures/smoke_summary.json).

## Endpoint/path discrepancy

Endpoint metrics and the linear and Bézier path endpoints were compared on every
split with an absolute tolerance of `2e-5`. One endpoint metric differed for
both paths; all other comparisons passed. The largest passing difference was
`3.814697265625e-6`.

Roman-empire GraphSAGE, a width-8, depth-9 BatchNorm smoke model, differs at seed
0 on validation accuracy. The endpoint record is `0.05525154620409012`; both
linear and Bézier `t=0` values are `0.05542806535959244`. The absolute difference
is `0.00017651915550231934`, which is one of 5,665 validation nodes. This exceeds
the requested tolerance. The original GPU report is unchanged.

A focused CPU replay used the original loader, two CPU threads, and the saved
checkpoints. Comparing the stored GPU-calibrated endpoint buffers directly with
a freshly CPU-calibrated `t=0` path gave a maximum logit difference of
`0.854838490486145` and 3,060 argmax changes across all nodes. This comparison
mixes stored GPU calibration with fresh CPU calibration and does not establish
cross-device invariance.

After one-pass CPU calibration of the endpoint as well, the endpoint and `t=0`
path logits matched exactly. Maximum and mean logit differences were zero, with
no argmax changes. Their validation accuracies both equaled
`0.06107678636908531`. This supports the path implementation when both sides use
the same device and calibration. GPU scatter order, deep BatchNorm amplification,
or near-tied logits may explain the original one-node difference, but that cause
has not been demonstrated.

## Cora source-budget pilots

Slurm array job `3834752` completed all four architecture tasks with exit code
`0:0`. Each run used the source endpoint budget of 500 epochs, maximum validation
accuracy selection, 200 curve steps, a 21-point grid, and seed pair `0:1`. These
are single-pair pilots. The paper reports three repeats without specifying its
endpoint pairing; our broader repeated-seed comparison remains undone.

The CPU replay record at
[`results/reference-architectures/cora_replay.json`](../results/reference-architectures/cora_replay.json)
verifies the original loader and mask hashes, finite report metrics, selected
epochs, endpoints, and linear and Bézier points at `t=0`, `0.5`, and `1`. All
four architectures passed. The maximum absolute loss difference was
`7.152557373046875e-7` against a `2e-5` tolerance. The maximum absolute accuracy
difference was `5.960464477539063e-8` against a `1e-6` tolerance.

The [Cora result summary](../results/reference-architectures/README.md) contains
the endpoint accuracies, path barriers, comparison plot, and detailed plot links.
It also records the central pilot result: all four sampled Bézier training-loss
barriers are zero, but validation and test loss barriers exceed their linear
counterparts. The curve settings need training and validation analysis before
any conclusion about the paper's test-loss result.

## Limits and next steps

The smoke runs prove that each architecture completes the endpoint, linear path,
Bézier path, artifact, and plotting workflow. Five training steps at width 8 do
not measure scientific performance. The Cora runs use full source endpoint
budgets but only one seed pair. Neither set is a three-pair numerical
replication.

All runs use the existing thesis loader and its masks rather than tunedGNN
preprocessing. Full-graph BatchNorm calibration is label-free and transductive.
It is an implementation choice and is not REPAIR. Full-budget runs on
Roman-empire, squirrel, and chameleon, three-pair Cora replication, curve
settings chosen with training and validation data, and any new REPAIR adapters
remain deferred.
