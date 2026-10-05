# Onboarding

## What this implements

[REPAIR](https://arxiv.org/abs/2211.08403) fixes activation variance collapse when
interpolating between neural networks. This package adapts the
[authors' VGG11 notebook](https://github.com/KellerJordan/REPAIR/blob/e90263d7a4d48376091327274ae541d8d6d34743/notebooks/Train-Merge-REPAIR-VGG11.ipynb).
It is a standalone project inside the thesis repository's `repair/` directory.

The pipeline trains or loads two compatible endpoints, aligns B's channels to A,
interpolates parameters, and calibrates each hidden preactivation. Calibration
sets target channel means and standard deviations to their weighted endpoint
averages. Labels are only needed for training and evaluation.

## Install and run

Use Python 3.10 or newer with a working PyTorch installation. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m repair demo --device cpu --output runs/demo
python -m unittest discover -s tests -v
```

The default demo trains two MLPs on 512 synthetic examples, evaluates on separate
examples, and compares five interpolation coefficients. It needs no downloads or
GPU. Choose a new output directory for each experiment; existing artifacts are
not overwritten.

To use an existing environment without installing the package:

```bash
PYTHONPATH=src python -m repair demo --device cpu --output runs/demo
PYTHONPATH=src python -m unittest discover -s tests -v
```

## CIFAR-10 experiment

For this cluster's environment and Slurm submission commands, see
[Slurm benchmark](docs/slurm.md).

```bash
python -m pip install -e '.[cifar]'
python -m repair cifar10 --device cuda --epochs 100 --width 1 \
  --batch-size 500 --calibration-batches 10 --output runs/cifar10
```

This downloads CIFAR-10 to `data/` and trains both VGG11 endpoints. CUDA is practical
for this experiment; a full CPU run is expensive. Default training follows the
upstream 100-epoch SGD recipe with momentum 0.9, weight decay 0.0005, maximum learning
rate 0.08, five warmup epochs, and linear decay. Short runs scale the warmup to fit.
The CLI uses float32; upstream used mixed precision. Fractional `--width` values
allow smaller runs and are an extension to the authors' integer-width models.

Calibration uses the first 10 deterministic training batches, 5,000 images at the
default batch size. Training retains random crop and horizontal flip. This avoids
different calibration samples across repeated passes. The source notebook used
all augmented training batches. Set `--calibration-batches 100` for all 50,000
images at batch size 500. Test images never calibrate the models.

Reuse endpoint state dictionaries to avoid retraining:

```bash
python -m repair cifar10 --device cuda --width 1 \
  --checkpoint-a runs/cifar10/endpoint_a.pt \
  --checkpoint-b runs/cifar10/endpoint_b.pt \
  --method sequential --output runs/cifar10-sequential
```

Width and number of classes must match the checkpoint architecture. Original
notebook VGG11 checkpoints work at matching integer width. The CLI expects plain
state dictionaries, not optimizer bundles or Lightning checkpoint dictionaries.

## Files to read

| File | Responsibility |
| --- | --- |
| `src/repair/models.py` | VGG11 and MLP construction |
| `src/repair/alignment.py` | Hidden layer names, correlations, Hungarian assignment, channel permutations |
| `src/repair/core.py` | Parameter interpolation, endpoint statistics, calibration, fusion |
| `src/repair/experiment.py` | Training, data loading, evaluation, CLI and saved artifacts |
| `tests/` | Mathematical checks and a training-to-checkpoint CPU experiment |
| `upstream/` | Original notebook and source provenance |

Start with the [SDK example](API_CONTRACTS.md#sdk-example), then the core and
alignment tests. `repair()` takes already aligned endpoints and performs its own
interpolation. It does not call alignment implicitly.

`method="batchnorm"` follows the authors' fast forward-pass implementation. Added
BatchNorm modules calibrate together; native model modules remain in evaluation
mode. Running statistics approximate global dataset moments.
`method="sequential"` measures one interpolated layer at a time after earlier
corrections. It costs additional calibration passes and is useful when checking
moment targets on fixed data. It is distinct from the paper's closed-form variant.

## Read experiment output

`report.json` stores endpoint metrics, the sampled interpolation curve, and loss
and error barriers relative to linearly interpolated endpoint metrics. Accuracy
and error are fractions. A barrier is the maximum measured excess over the sampled
coefficients, not a guarantee about every coefficient between them.

The four `.pt` files contain ordinary state dictionaries. The repaired midpoint
has corrections fused into its hidden affine layers, so load it with the same
model factory. See [artifact contracts](API_CONTRACTS.md#experiment-artifacts).

Install `.[plots]` and run `python -m repair.plotting PATH/report.json` to write
`interpolation.png` and `interpolation.pdf`. Both figures show test loss and
accuracy versus alpha, with three merge methods and the endpoint baseline.
The Slurm script generates them automatically after a successful experiment.

Passing tests establish numerical behavior and a runnable pipeline. Reproducing
the paper's accuracy requires independent full training runs; no accuracy claim
is inferred from the synthetic demo.

## Later graph integration

Keep graph adaptation separate from this baseline. A graph adapter must define
which hidden node features correspond between endpoints, how batches call the
model, and which training nodes contribute calibration statistics. The current
SDK accepts single input tensors and only implements permutations for VGG and
sequential MLPs.

Permutation constraints are the main additional work. Reorder every consumer of
a hidden channel, including message-passing projections, attention projections,
normalization parameters and residual branches. Graph-node permutations and hidden
feature-channel permutations are different operations. Test that each proposed
channel permutation preserves the endpoint's node logits before merging anything.

Once those adapters exist, use training-node feature statistics to calibrate the
same target-mean/target-standard-deviation equations. Keep train/validation/test
mask policy explicit. Document any intentionally transductive calibration policy.

The current thesis connectivity experiments use `LinearSpectralGNN` in
`../src/experiments/common.py`. Its `lin1`/`lin2` hidden-channel pair is a simpler
first adapter than `DiffusedAttention`. Keep its spectral `gamma` coefficient axis
separate from neuron channels. The later insertion point is `barrier_along_path`,
after installing each interpolated parameter vector and before evaluation.
That routine currently saves parameters only; a REPAIR adapter must also preserve
buffers and restore the original state. `DiffusedAttention` additionally requires
attention-head and residual constraints, and `MyModel` must retain its tuple output
contract.

## Deferred work

Graph adapters, residual-network alignment, ImageNet/FFCV reproduction, the
closed-form approximation, distributed training, and a benchmark reproduction
sweep remain outside this first implementation. See [API contracts](API_CONTRACTS.md)
before extending supported architectures.
