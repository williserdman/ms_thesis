# Re-Basin, Bézier, and REPAIR framework

Date: 2026-09-29
Status: proposed written design, awaiting user review

## Purpose and agreed scope

Build a small PyTorch framework that combines endpoint alignment, quadratic
Bézier control fitting, and post-hoc REPAIR. An agent adding a model architecture
should normally implement an adapter and use the existing pipeline unchanged.
The first supported architectures are a sequential ReLU MLP and a plain PyG GCN.

The user requested the shared structure and MLP/GCN examples. Deliver a Python
package, two runnable examples, and a short adapter guide. Endpoint training
belongs to the examples or caller. A benchmark runner, standalone verification
CLI, automatic architecture discovery, and a general plugin system are outside
this version.

Success means both examples exercise the same pipeline, return usable repaired
models, and demonstrate how to add an architecture without rewriting matching,
curve optimization, or REPAIR statistics.

## Existing code and reuse

All three source projects are siblings under the thesis checkout.

| Source | Reuse |
| --- | --- |
| `git_re_basin/src/git_re_basin/core.py` and `matching.py` | Import public state/spec validation, `weight_matching`, and `apply_permutation`. |
| `gcn_bezier_conn/gcn_mc/paths.py` and `experiment.py` | Adapt quadratic interpolation, midpoint initialization, and control-only optimization to caller-supplied model execution and loss. |
| `gcn_bezier_conn/gcn_mc/repair.py` and `activations.py` | Adapt correction of arbitrary sampled states and sequential hidden-layer calibration. |
| `repair/src/repair/core.py` | Reuse the population-moment accumulation and affine-folding approach for minibatch calibration. |

The existing GCN experiment already combines Pearson activation alignment,
Bézier fitting, and REPAIR. This project adds a reusable architecture interface
and uses Git Re-Basin weight matching. It does not rerun completed paper
replications or promise the same results after changing the alignment method.

Depend on the locally installed `git-re-basin-thesis` sibling package. Document
its editable installation rather than assuming it is available from a public
package index or modifying `sys.path` at runtime. Keep PyG optional for the GCN
adapter and example. The MLP path must import and run without PyG.

Adapt only the small curve/calibration functions needed here; do not copy the
experiment runners or import their dataset pipelines. Record local source paths
and preserve existing attribution/notices for adapted material. Do not copy the
upstream REPAIR notebook or assign a new license to upstream material lacking an
explicit license.

## Public workflow

The main operation, `connect`, accepts two trained models, an architecture
adapter, re-iterable training batches, a loss callback, re-iterable calibration
batches, and small matching/curve settings. The loss callback receives logits
and the current batch and returns a scalar differentiable training loss.

1. Validate compatible model states and the adapter's complete permutation spec.
2. Copy the endpoints. Weight-match B to A and apply the resulting permutation.
3. On calibration inputs in evaluation mode, verify that original B and aligned B
   produce equal logits within configured floating-point tolerance.
4. Initialize one control state to the aligned endpoints' arithmetic midpoint.
   Fit only this control using the supplied training loss.
5. Measure aligned endpoint preactivation statistics once, in evaluation mode.
6. Return a path object containing the frozen endpoint states, aligned B, control,
   permutation, endpoint statistics, and an owned model template.

The path provides `model_at(t, repair=False, calibration_data=None)`. It returns
an independent ordinary model in evaluation mode. For an interior point with
`repair=True`, the caller supplies calibration data so REPAIR can measure that
point after each preceding correction. The path does not serialize or retain
data loaders. At exact endpoints REPAIR is bypassed and no recalibration is
needed. At `t=1` the returned state is aligned B, with original B's function.

The returned model can be saved and restored through ordinary PyTorch
`state_dict` operations with the same architecture. Full path serialization and
custom report/checkpoint schemas are deferred.

## Architecture adapter

Keep architecture knowledge in one adapter per supported family. The interface
has four responsibilities:

1. **Permutation specification.** Identify the hidden-channel groups attached to
   each tensor axis, accounting for every parameter and buffer. Input features
   and output classes retain their meaning and order.
2. **Execution.** Evaluate a supplied full state on a batch and return logits via
   strict functional execution. This handles the MLP's tensor input and the GCN's
   feature/edge inputs without changing the pipeline.
3. **Repair sites.** List hidden preactivation module outputs in forward order
   and describe their channel axes. Shared code installs temporary hooks and
   accumulates selected observations. A caller-supplied observation selector
   chooses calibration examples or training-node rows before reduction; dataset
   splits are not hard-coded into the framework.
4. **Correction application.** Apply a supplied per-channel scale and shift to
   a copied state at a repair site. Provide a shared helper for affine weight-row
   and bias corrections. An architecture with a different valid correction can
   override this operation in its adapter.

The MLP adapter supports biased `nn.Linear`/`nn.ReLU` stacks ending in a Linear
classifier. The GCN adapter supports biased, uncached `GCNConv` stacks with ReLU
and optional dropout between convolutions. Both exclude the classifier/output
convolution from REPAIR. GCN repair sites are complete convolution outputs, not
the internal linear transform before graph aggregation.

The GCN example selects only training nodes for calibration, while message
passing uses the full graph. The MLP example uses training examples. Neither
uses calibration labels for moment estimation or alignment.

Adapters explicitly reject structures they do not support. Arbitrary residual,
attention, tied-weight, or normalization arrangements are not inferred from
module names. Their future adapters must describe valid coupled permutations
and correction sites. Some architectures may also require extending the current
buffer or matcher capabilities; the interface does not guarantee compatibility
with every possible network.

## State and optimization rules

Use complete state dictionaries for functional execution and distinguish
parameters through `named_parameters()`. Interpolate floating parameters only.
Non-parameter state remains fixed and must match between A and aligned B.
Native BatchNorm models are unsupported in this version; the supplied adapters
have no stateful normalization.

Neither alignment, fitting, nor calibration may change caller-owned endpoint
parameters, buffers, gradients, or training flags. Fit and calibrate using owned
model copies. Keep frozen endpoint tensors detached from the control graph.

For aligned endpoints A and B and learned control C:

```text
theta(t) = (1-t)^2 A + 2t(1-t) C + t^2 B
```

Default fitting uses Adam, learning rate 0.01, 100 optimizer steps, and one uniform
sample of t per training batch/step. Cycle the finite training source as needed.
Expose step count, learning rate, and seed as explicit settings. Use training
mode on the owned template during fitting, allowing GCN dropout, and retain the
final control. No validation/test selection or endpoint optimization occurs.

## REPAIR behavior

Repair each requested interior sample after curve fitting. Sequentially measure
each hidden site's current preactivations after corrections to earlier sites.
Use population moments pooled by observation count across calibration batches.
Blend aligned endpoint means and standard deviations linearly in t:

```text
target_mean = (1-t) mean_A + t mean_B
target_std  = (1-t) std_A  + t std_B
scale      = target_std / max(current_std, epsilon)
shift      = target_mean - scale * current_mean
```

Use `epsilon=1e-8` by default. A constant channel cannot acquire positive variance
through affine correction. Calibration uses fixed inputs, evaluation mode, and
no gradients. Its source must be finite, nonempty, and re-iterable, with at least
two selected observations overall.

Fold corrections into supported affine parameters so returned models need no
calibration data at inference. Do not repair only the control point and do not
differentiate through REPAIR. The repaired family is a post-hoc repaired path,
generally not an exact quadratic Bézier parameter curve. Examples label repaired
and unrepaired results separately and make no promise of lower test loss.

## Package and examples

Use the distribution name `rebasin-bezier-repair` and import package
`rebasin_bezier_repair`, with a conventional `src/` layout.

```text
src/rebasin_bezier_repair/
    __init__.py
    pipeline.py
    paths.py
    repair.py
    adapters/
        base.py
        mlp.py
        gcn.py
examples/
    mlp.py
    gcn.py
docs/adding-an-architecture.md
tests/
```

`pipeline.py` composes the stages and owns the path object. `paths.py` handles
differentiable control fitting/interpolation. `repair.py` owns hooks, pooled
moments, and sequential calibration. Adapters contain model-specific execution,
permutation descriptions, and correction locations.

The MLP example trains two small classifiers on fixed synthetic data. The GCN
example trains two small classifiers on a fixed synthetic graph with explicit
train/validation/test masks. Both run on CPU, require no download or thesis data
loader, and use independent endpoint seeds.

Each example calls the same public interface, materializes an unrepaired and a
repaired midpoint, and prints endpoint and midpoint loss/accuracy. Small example
training budgets demonstrate execution rather than reproduce paper results.
Keep plotting, sweeps, a checkpoint-loading CLI, and Slurm outside scope.

The README documents installation and both example commands. The adapter guide
uses the working examples as its template, explains the four adapter
responsibilities, and lists the focused checks needed for a new architecture.

## Errors and focused verification

Raise clear errors for incompatible states, incomplete/invalid permutation
specs, unsupported adapter structures, invalid t, unequal fixed buffers, missing
calibration data for a repaired interior point, empty data, or nonfinite losses.
Do not silently omit a requested stage.

Verify only the requested path and its central numerical contracts:

- Both examples execute end to end through the same interface on CPU.
- Alignment preserves endpoint logits; all operations preserve caller models.
- Path endpoints reproduce A and aligned B; control fitting has nonzero
  gradients and updates the control without changing endpoints.
- A nondegenerate calibration case reaches the requested moments within
  numerical tolerance after sequential correction.
- GCN calibration selects training-node rows; changing held-out labels cannot
  alter fitting or calibration when the supplied callbacks use training data.
- A repaired model's state dictionary reloads into its original architecture and
  reproduces its predictions.

Use focused automated tests for these contracts and short executions of the
examples. No broad benchmark or review cycle is part of this scope.

## Deferred work

Additional architecture adapters, activation/STE matching options, native
BatchNorm recalibration policies, REPAIR inside curve optimization, full-path
serialization, benchmark reporting, and performance optimization remain future
work. Add them only when a concrete architecture or experiment needs them.
