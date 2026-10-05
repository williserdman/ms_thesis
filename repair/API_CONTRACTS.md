# API contracts and SDK

This is a Python SDK. It provides no HTTP server or network endpoints.
The distribution is `repair-interpolation`; import it as `repair`.

## SDK example

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from repair import align_models, hidden_layer_names, interpolate, repair, tiny_mlp

# Replace these initializations with independently trained endpoint checkpoints.
model_a = tiny_mlp()
model_b = tiny_mlp()
calibration = DataLoader(TensorDataset(torch.randn(128, 16)), batch_size=32)

aligned_b = align_models(model_a, model_b, calibration)
merged = interpolate(model_a, aligned_b, alpha=0.5)
repaired = repair(
    model_a,
    aligned_b,
    calibration,
    alpha=0.5,
    layer_names=hidden_layer_names(model_a),
    method="batchnorm",
    fuse=True,
)

with torch.no_grad():
    logits = repaired(torch.randn(8, 16))  # [8, 4]
torch.save(repaired.state_dict(), "repaired.pt")

restored = tiny_mlp()
restored.load_state_dict(torch.load("repaired.pt", weights_only=True))
restored.eval()
```

`merged` is the uncorrected comparison. Pass endpoints to `repair()`, not that
already interpolated model. The end-to-end training example is `python -m repair demo`.

## Common input contract

Endpoint models must have the same architecture, state keys, tensor shapes,
channel interpretation, preprocessing, and class order. Put both endpoints on the
same device with the same floating-point dtype before calling the SDK.
Permutation matching does not reorder class labels.

`calibration_data` is a finite, re-iterable source, normally a `DataLoader` or list.
Each item is a batch tensor, or a tuple/list whose first item is the batch tensor.
Labels, if present, are ignored. Use `[batch]` for one tensor batch. Single-use
generators are unsupported because the algorithm needs multiple passes.

Models receive one positional input tensor. Dictionaries, graph `Data` objects,
attention masks and multiple model arguments require an adapter and are outside
this SDK's current contract. MLP inputs are floating tensors `[N, input_dim]`;
VGG11 inputs are floating tensors `[N, 3, 32, 32]`. Calibration moves inputs to the
model's device and dtype. Inputs and activations must contain finite values.

Use training data with fixed preprocessing for repeatable calibration. A sequential
pass must revisit the same samples to make measured moment targets comparable.
`max_batches=None` consumes every batch on each pass; a positive integer consumes
at most that many batches per pass. Empty calibration sources are invalid.
BatchNorm calibration requires more than one value per channel per batch. For
Linear layers, each batch must contain at least two examples. Sequential calibration
can measure across smaller batches.

Operations return new model copies in evaluation mode. Inputs' parameters, buffers
and training flags remain unchanged. Calibration runs without gradient recording.
The returned model can be used for inference; subsequent training changes its
calibrated statistics or fused weights and invalidates the original moment result.

## Models

```python
vgg11(width=1, num_classes=10) -> VGG
tiny_mlp(input_dim=16, hidden_dims=(32, 32), num_classes=4) -> torch.nn.Sequential
hidden_layer_names(model) -> list[str]
```

Factories construct CPU models. Move them with `.to(device)`.
VGG11 uses eight biased convolutions, ReLUs, five max-pooling stages, and one
classifier. It has no native BatchNorm. Integer widths match the authors' checkpoint
layout: `features.<index>.weight/bias` and `classifier.weight/bias`.
Fractional widths round each base channel count with a minimum of one.

`tiny_mlp` alternates Linear/ReLU modules and ends in a Linear classifier.
Default hidden names are `["0", "2"]`; its classifier is `"4"`.
`hidden_layer_names` returns VGG's convolutions or the MLP's hidden Linear layers
in execution order. It excludes the classifier. These architecture adapters do
not recognize arbitrary PyTorch graphs.

## Channel alignment

```python
align_models(reference, candidate, calibration_data, *, max_batches=None) -> torch.nn.Module
```

Return a permuted copy of `candidate`. For each hidden layer, measure cross-model
post-ReLU channel correlations, maximize total correlation with SciPy's Hungarian
assignment, and apply the assignment to both the current layer's output channels
and the next layer's input channels. Matching moments weight observations across
batches, including a shorter final batch.

Candidate predictions remain equal within floating-point roundoff. Alignment
does not make the endpoints' predictions equal. It only changes their hidden
channel ordering. Supported architectures are the provided VGG and sequential
Linear/ReLU MLPs. Residual streams, grouped convolutions, tied weights, and arbitrary
message-passing graphs need their own valid permutation rules.

## Parameter interpolation

```python
interpolate(model_a, model_b, alpha=0.5) -> torch.nn.Module
```

`alpha` is a finite scalar in `[0, 1]`, the weight of B. Floating parameters and
buffers use `(1 - alpha) * A + alpha * B`. At the endpoints, the returned model
matches A or B and remains a distinct copy. Interpolation itself does not align
neurons or calibrate activations.

Integer buffers come from A for `alpha < 0.5` and B otherwise. They are not
fractionally averaged. This choice is irrelevant to the normalization-free model
factories but matters when interpolating a model with counters.

## REPAIR calibration

```python
repair(
    model_a,
    model_b,
    calibration_data,
    alpha=0.5,
    *,
    layer_names,
    max_batches=None,
    method="batchnorm",
    fuse=True,
) -> torch.nn.Module
```

`model_b` must already be aligned with A. `layer_names` is a nonempty sequence of
unique dotted module names in forward execution order. Selected layers must be
`nn.Linear` with `[N, C]` output or `nn.Conv2d` with `[N, C, H, W]` output. Do not
include the classifier unless deliberately repairing logits. Select each hidden
layer once; shared layers or repeated invocation need a separate adapter.
The caller supplies the execution order; the SDK does not trace the model to check it.

For each selected channel, the target moments are:

```text
target_mean = (1 - alpha) * mean_A + alpha * mean_B
target_std  = (1 - alpha) * std_A  + alpha * std_B
```

Average standard deviations, not variances. Statistics are preactivations before
ReLU. Convolutions aggregate over batch and spatial dimensions.

| Method | Behavior |
| --- | --- |
| `batchnorm` | Authors' forward-pass approach: collect endpoint BatchNorm running moments, set added normalization affine parameters to target moments, calibrate all added layers together. Running variances and batch averaging approximate dataset moments. |
| `sequential` | Measure population moments over calibration observations, then correct each layer after earlier corrections. Needs another model pass per selected layer. |

Native modules stay in evaluation mode during calibration; only the inserted
BatchNorm modules collect training statistics. Epsilon is `1e-5`. Stabilization means moment
equality is numerical rather than symbolic. A constant merged channel cannot
acquire positive variance by affine rescaling alone. The method does not optimize
weights against labels and does not guarantee a smaller test loss.

`fuse=True` folds each correction into its Linear/Conv2d layer. The result uses
ordinary modules and needs no calibration loader at inference. A selected biasless
layer gains a bias parameter; instantiate a matching biased layer when restoring
that state dictionary. Provided VGG/MLP factories already use biases.
`fuse=False` retains correction wrappers; its state dictionary has different keys
and must be loaded into an identically wrapped model. Prefer fused artifacts for
SDK consumers.

For a wrapped result, `from repair.core import fuse_repair` exposes
`fuse_repair(model) -> torch.nn.Module`. It returns another eval-mode copy with
REPAIR wrappers folded into their affine layers.

Invalid alpha, incompatible endpoints, missing/unsupported layer names, invalid
limits, and empty calibration are errors, never silent fallback to unrepaired
weights. Runtime shape/device problems also propagate as PyTorch errors.

`repair()` raises `ValueError` for invalid numeric ranges, incompatible state,
unknown/duplicate/empty names and empty data. It raises `TypeError` for unsupported
layer types, malformed batches and single-use calibration iterators.

## Evaluation helper

```python
from repair.experiment import evaluate
evaluate(model, labeled_data) -> dict
```

`labeled_data` yields `(inputs, class_indices)`. It returns `loss`, `accuracy`,
and `samples`. Loss is sample-averaged cross entropy; accuracy is a fraction in
`[0, 1]`. The helper restores the model's prior training flags and rejects empty
data. Labels must be integer class indices suitable for cross entropy.

## Experiment artifacts

Both CLI commands write these files to `--output`:

| File | Contract |
| --- | --- |
| `endpoint_a.pt`, `endpoint_b.pt` | CPU state dictionaries of the independently trained or loaded endpoints |
| `aligned_b.pt` | Candidate state dictionary after function-preserving permutation |
| `repaired_midpoint.pt` | Fused model at `alpha=0.5`, saved even if the requested curve omits 0.5 |
| `report.json` | UTF-8 JSON, `schema_version: 1`, no NaN/Infinity |

The report records `dataset`, `architecture`, `seed`, `epochs`, `batch_size`,
`device`, `torch_version`, `method`, `layer_names`, `calibration_batches`,
`endpoints`, `curve`, `barriers` and a `note`. New reports also include `gpu`,
`peak_cuda_memory_bytes`, and `timings`. `epochs` records the requested
training setting; supplied endpoints are loaded without retraining.

Each curve row has `alpha` and metric objects under `unaligned`, `aligned`,
and `repaired`. Each metric object has:

```json
{"loss": 0.7, "accuracy": 0.8, "samples": 256}
```

For each variant, `barriers` contains `loss_barrier` and `error_barrier`. At each
sampled alpha subtract the linearly interpolated endpoint loss/error, then take
the maximum with zero. Error barriers use fractions, so `0.05` is five percentage
points. Coefficients are deduplicated and sorted. A sparse curve can miss the true
maximum. The synthetic dataset is a functional demonstration, not a benchmark.

Existing experiment artifacts cause the CLI to reject the output directory.
Checkpoints contain model state only. They do not resume optimizer/scheduler state
and do not include an executable model class. Reconstruct the architecture from
the report before `load_state_dict`.

## Plotting and timing

```python
from repair.plotting import plot_report
plot_report(report_path, output_dir=None) -> list[pathlib.Path]
```

With the `plots` extra installed, this reads a schema-v1 report and writes
`interpolation.png` at 180 DPI and `interpolation.pdf`. The default output directory
is the report's parent. It returns the two paths and does not modify the report.
Existing plot files of those names are replaced. It uses the headless Agg backend.
Malformed reports raise `ValueError`. The CLI is
`python -m repair.plotting REPORT_JSON [--output-dir DIR]`.

`timings.training.a` and `.b` contain `epoch_seconds` and their `total_seconds`
sum for trained endpoints; loaded endpoints have no training entry. Top-level
`timings` also records `alignment_seconds`, `curve_seconds`, and `total_seconds`.
Total time starts at CLI entry and ends before report serialization. GPU reports
include the model name and peak allocated CUDA tensor memory in bytes; CPU reports
set those two fields to null. Slurm queue time is outside these timings.

The CIFAR CLI uses persistent data-loader workers when `--num-workers > 0`.
Calibration/test sample ordering and transforms remain fixed. Random training
augmentation streams persist across epochs; changing worker configuration can
change an individual training run's random augmentation sequence.

## Compatibility and extension limits

The current version covers sequential MLPs and CIFAR VGG11. A graph integration
needs separate input and permutation adapters, train-node masking for statistics,
and proof that any permitted permutation leaves endpoint predictions unchanged.
The correction equations can be reused after those contracts hold. It is not
safe to apply `align_models` directly to a graph model or a residual network.
