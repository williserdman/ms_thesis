# Reference alignment and REPAIR verification

The extension reuses the completed tuned matrix for GCN, MLP, GraphSAGE and GAT
on Cora, Roman-empire, squirrel and chameleon. Each configuration has three
independent saved endpoint pairs and a 21-point interpolation grid. No endpoints,
Optuna studies or Bézier controls were trained again.

## Focused checks

The combined suite passed **42 tests in 6.360 seconds**, after imports. The local
log is `runs/reference-repair-tests.log`. New checks cover nonidentity channel
permutations including GAT attention, residuals and normalization; endpoint
invariance; training-node moments and label isolation; the explicit affine
wrapper; frozen native BatchNorm statistics; epsilon-adjusted correction;
checkpoint replay; and a real training-to-postprocessing run for all four models.
A deliberately corrupted alignment still fails the float64 verification.
Source replay checks accept at most one changed correct prediction and reject
two or a cross-entropy difference above 2e-5.

Training implementation hashes still match the completed Optuna contexts. The
original data pipeline, endpoint models and cached search results were untouched.

## Numerical precision

The first sweep stopped on five configurations because channel permutations
changed floating-point summation order. CPU float64 probes on the exact trained
permutations reduced logit differences from roughly 1e-5 to at most 6.04e-14,
with no class changes in those probes. The adapter retains its original strict
float32 check and requires strict float64 equivalence if that fails. The raw
errors and verification dtype remain in each report. See the
[alignment probe](../results/repair-tuned-20260918/alignment_precision_probe.json).

A separate GPU probe found one accuracy discrepancy among 126 Roman-empire GAT
path points. At pair 0:1, Bézier t=0.95, one of 5,666 test nodes had a top-two logit
margin of 2.86e-6. Cross-entropy matched exactly. Repeated identical calls to the
original evaluator varied by up to 3.86e-5 in logits. See the
[source replay probe](../results/repair-tuned-20260918/source_replay_precision_probe.json).

Source-path replay now records loss and integer correct-count differences,
requires loss agreement within 2e-5 and at most one changed correct prediction
per split, then retains the saved source metrics. Endpoint and repaired-midpoint
checks still require accuracy agreement within 1e-6. Earlier successful sweep
reports passed that stricter source accuracy check; their tiny loss differences
are retained in the archive audit.


## Completed matrix

All 16 configurations completed with three pairs each. Successful artifacts came
from array 3835188, retries 3835207 and final Roman-empire GAT job 3835240.
The GPU numerical probe was job 3835218. Failed attempts remain separately under
ignored `runs/repair-tuned-initial-failures-20260918/` and
`runs/repair-tuned-replay-failure-20260918/`; no endpoint data was discarded.

The [result archive](../results/repair-tuned-20260918/README.md) includes full
reports, sixteen labeled four-method plots, and 4-by-4 loss/accuracy grids in
PNG and PDF. Raw weights and copied source artifacts remain in
`runs/repair-tuned-20260918/`. The archive audit passed for all 48 pairs. Maximum
stored source-curve loss difference was 9.54e-7. All 48 repaired midpoint
checkpoints reloaded successfully, with maximum loss difference 4.77e-7 and no
accuracy difference. Twelve alignment checks required float64 confirmation.
The later fifteen pairs record predicted-class disagreement counts explicitly;
all were zero. Earlier pairs passed the strict float32 logit check.

Alignment reduced Cora mean test barriers from 0.239–0.559 to 0.001–0.019;
aligned REPAIR reached zero sampled test barrier for all four models. On
Roman-empire, alignment substantially reduced each barrier, but REPAIR increased
it slightly relative to alignment alone. Roman-empire GAT's Bézier comparator
retained zero sampled test barrier. Squirrel GCN and GAT worsened with alignment
and REPAIR. These are measured results, not a guarantee that either correction
improves performance. A zero sampled barrier measures excess over the endpoint
loss chord; it does not imply low absolute loss or high accuracy.

Deferred work: REPAIR on Bézier curves, broader hyperparameter/seed searches,
inductive calibration policies, and fusing reference affine wrappers. The wrapper
follows the reference block order explicitly; architecture changes must preserve
its identity-forward and permutation tests.
