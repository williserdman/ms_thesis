# Alignment and REPAIR across reference architectures

Extend the existing saved-endpoint analysis to the completed Optuna matrix:
GCN, MLP, GraphSAGE and GAT on Cora, Roman-empire, squirrel and chameleon.
Use all three saved pairs per configuration, each containing two independently
trained endpoints. Retain the 21-point grid and each pair's saved Bézier control.
No endpoint training, tuning, data-pipeline changes or curve fitting occurs.

Compare raw linear, aligned linear, aligned linear plus sequential REPAIR, and
Bézier. Alignment uses training-node hidden-channel correlations and Hungarian
assignment. It must preserve full-graph endpoint logits. Permutations cover every
consumer, including attention vectors, both GraphSAGE projections, learned
residual projections, normalization state and the classifier input. Class outputs
and graph nodes remain fixed. The optional input projection is also aligned.

Reference REPAIR measures each complete hidden block after residual addition and
normalization, before ReLU. Include the optional input projection, which has no
ReLU. Use training-mask rows for weighted endpoint means and standard deviations,
with epsilon 1e-5 and sequential calibration after earlier corrections. Retain
explicit channel-affine corrections so GAT attention logits are not changed by
incorrectly fusing output corrections into attention projections.

Raw, aligned and Bézier paths retain the original full-graph BatchNorm calibration
policy. Each interior repaired model starts with a calibrated aligned linear
model, then freezes its BatchNorm buffers while applying sequential corrections.
This separates inherited full-graph normalization from training-node REPAIR
statistics. Dropout is disabled. Return exact endpoint copies at t=0 and t=1.
The existing legacy GCN adapter and fused checkpoint format remain supported.

Save alignment assignments, invariance errors, calibration diagnostics, all four
curves, aligned endpoints and replayable repaired midpoints. New reference repair
checkpoints identify their explicit affine wrapper. Preserve source provenance
and require raw linear/Bézier replay to agree with source reports. Report numeric
failures rather than silently dropping a pair.

Focused verification covers architecture symmetries, affine correction equations,
label/mask isolation, state preservation and checkpoint replay. Complete the
16-configuration Slurm sweep, archive labeled graphics/reports and update
onboarding/handoff. Changing the REPAIR adapter must not invalidate endpoint
Optuna studies. Extending REPAIR to Bézier or tuning the correction is deferred.
