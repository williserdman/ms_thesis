# REPAIR implementation research

Research date: 2026-09-15. The source revision inspected was
`e90263d7a4d48376091327274ae541d8d6d34743`.

## Primary sources

- [ICLR 2023 paper](https://openreview.net/forum?id=gU5sJ6ZggcX) and
  [arXiv record](https://arxiv.org/abs/2211.08403)
- [Official author repository](https://github.com/KellerJordan/REPAIR) and
  [pinned source revision](https://github.com/KellerJordan/REPAIR/commit/e90263d7a4d48376091327274ae541d8d6d34743)
- [Self-contained VGG11/CIFAR-10 notebook](https://github.com/KellerJordan/REPAIR/blob/e90263d7a4d48376091327274ae541d8d6d34743/notebooks/Train-Merge-REPAIR-VGG11.ipynb)
- [ResNet50/ImageNet notebook](https://github.com/KellerJordan/REPAIR/blob/e90263d7a4d48376091327274ae541d8d6d34743/notebooks/Merge-ResNet50-ImageNet.ipynb)
- [Repository README](https://raw.githubusercontent.com/KellerJordan/REPAIR/e90263d7a4d48376091327274ae541d8d6d34743/README.md)

The paper names the repository as its code release. The repository README calls the
VGG11 notebook self-contained and says the ResNet50 notebook needs FFCV ImageNet
files and two pretrained checkpoints. The repository is a collection of notebooks
and training scripts. It has no installable package, command-line interface,
requirements file, lockfile, tests, releases, or documented programmatic API.

## Algorithm that the implementation must preserve

REPAIR starts with two independently trained networks of the same architecture.
The channels of endpoint B must first be permuted into correspondence with endpoint
A. For an interpolation coefficient `alpha`, form every interpolated parameter as

```text
theta_alpha = (1 - alpha) * theta_A + alpha * theta_B_aligned
```

For each selected channel, measure the endpoint preactivation random variables
`X_A` and `X_B` over calibration data. REPAIR asks the repaired interpolated channel
to have

```text
goal_mean = (1 - alpha) * mean(X_A) + alpha * mean(X_B)
goal_std  = (1 - alpha) * std(X_A)  + alpha * std(X_B)
```

The paper applies this to every convolutional preactivation in VGG. For ResNets it
also repairs each residual-block output. Repairing only convolution outputs in a
BatchNorm ResNet amounts to resetting the existing BatchNorm statistics; the block
output corrections are the additional part needed for full REPAIR.

### Forward-pass variant used by the main notebook

1. Wrap every chosen endpoint module with a tracker. The official tracker runs the
   original module, feeds its output through a temporary `BatchNorm2d` only to
   collect per-channel running mean and variance, discards the temporary output,
   and returns the original output.
2. Run the same calibration distribution through both aligned endpoint networks.
   The paper reports that about 5,000 examples suffice. The self-contained notebook
   actually iterates over the whole 50,000-example CIFAR-10 training loader with
   random crop and flip.
3. Wrap every chosen interpolated module with a reset layer. Set the temporary
   BatchNorm affine bias to `goal_mean` and affine weight to `goal_std`.
4. Reset its running statistics, use cumulative averaging (`momentum=None` in the
   notebook), and run calibration inputs through the wrapped interpolated network
   in training mode. Each temporary BatchNorm standardizes its incoming channel and
   immediately maps it to the target mean and standard deviation. Downstream layers
   therefore see the corrected upstream activations during the same forward pass.
5. Switch the repaired model to evaluation mode. The temporary BatchNorm modules
   then use their calibrated running statistics. They may remain explicit or be
   fused into the preceding convolution.

For a convolution with weights `W`, bias `b`, and following temporary BatchNorm
parameters `running_mean`, `running_var`, `eps`, affine `weight`, and affine `bias`,
the official notebook fuses them as

```text
scale = weight / sqrt(running_var + eps)
W_fused = scale[:, None, None, None] * W
b_fused = bias + scale * (b - running_mean)
```

This is the paper's "exact" variant. The word means that temporary BatchNorm
normalizes each calibration batch to the requested channel statistics. Finite data,
running-statistic estimation, augmentation, `eps`, and PyTorch's variance estimators
still introduce numerical differences. Calibration is a coupled forward pass with
all reset layers installed, rather than independent corrections computed once from
the unrepaired interpolated network.

The paper says pre-existing BatchNorm layers remain frozen during calibration of
the added layers. The ResNet50 notebook's shared `reset_bn_stats` helper instead
calls `model.train()` and resets every exact `BatchNorm2d`, including native ones.
This source discrepancy does not affect the normalization-free VGG11 MVP. A generic
implementation should follow the paper and leave native normalization modules in
evaluation mode unless it exposes BatchNorm recalibration as a separate operation.

### Closed-form approximate variant

The alternative avoids forward passes through each interpolated model after it has
measured endpoint statistics and covariance. For corresponding endpoint channels,

```text
var_alpha = (1 - alpha)^2 * var_A
          + alpha^2 * var_B
          + 2 * alpha * (1 - alpha) * cov(X_A, X_B)

scale = goal_std / sqrt(var_alpha)
```

The required shift is the one that maps the estimated interpolated mean to
`goal_mean`. The formula is exact for a first-layer unit because its activation is
the linear interpolation of endpoint activations. It is only an approximation in
deeper layers because their inputs differ after interpolation and earlier nonlinear
operations. The paper reports that it works for deep MLP experiments but is too
weak for the ResNet50/ImageNet case. Use the forward-pass variant for the faithful
MVP and defer this method.

## Alignment and permutation requirements

The official VGG implementation aligns layers in forward order. For each convolution:

1. Run endpoint subnetworks through the activation after that convolution and ReLU.
2. Reshape `N x C x H x W` activations to `(N*H*W) x C`.
3. Accumulate channel means, second moments, and the cross-endpoint outer product.
4. Convert these to a `C x C` Pearson-correlation matrix. The notebook adds `1e-4`
   to the denominator.
5. Use SciPy's Hungarian assignment,
   `linear_sum_assignment(correlation, maximize=True)`, to maximize total matched
   correlation.
6. Apply the assignment to B's current layer output dimension, including its bias
   and any attached normalization parameters and buffers. Apply the same assignment
   to the input dimension of the next convolution or final classifier. Verify that
   B's function did not change beyond floating-point noise.

Alignment is not part of the statistical correction itself, but the paper finds it
essential near the midpoint. REPAIR can sit on top of a different valid alignment
method.

Residual networks need constrained permutations. A channel flowing through an
identity residual stream must receive one shared permutation everywhere on that
stream. The shortcut branch, main branch, next block input, and final classifier
must agree. Independent per-convolution permutations can change the represented
function. For a first implementation, a sequential VGG has a much smaller error
surface than generic graph rewriting for residual networks.

## Faithful runnable MVP

Port the official normalization-free VGG11/CIFAR-10 notebook into a small Python
package and command-line demo. Keep the critical path:

- Train or load two same-shape VGG11 endpoints.
- Align B to A by activation correlation and Hungarian assignment.
- Create a midpoint at `alpha=0.5`.
- Apply forward-pass REPAIR to all eight convolution outputs.
- Evaluate endpoints, aligned midpoint before REPAIR, and repaired midpoint.
- Optionally fuse the temporary BatchNorm layers and save a normal VGG11 state dict.

The upstream notebook imports PyTorch, torchvision, NumPy, SciPy, and tqdm. The
official notebook assumes CUDA and mixed precision. A maintained port can support
CPU for a small smoke run while documenting CUDA as the practical full-demo path.
Expose the random seed, device, data directory, endpoint paths, `alpha`, calibration
sample count, and output path. Accept a calibration iterable whose items are either
inputs or `(inputs, labels)` because REPAIR never uses labels.

A possible interface considered during research separated the operations below.
These names are illustrative; the implemented signatures are in
[API_CONTRACTS.md](../API_CONTRACTS.md):

```python
permutations = align_channels(model_a, model_b, calibration_loader)
merged = interpolate(model_a, aligned_b, alpha=0.5)
repaired = repair(merged, model_a, aligned_b, calibration_loader, alpha=0.5)
fused = fuse_repair_layers(repaired)
```

Document that endpoint models must share an architecture, tensor shapes, module
order, preprocessing, and output semantics. `align_channels` should copy its B
input or state clearly that it mutates it. The calibration source should match the
training distribution and preprocessing. A saved permutation map makes the result
auditable and avoids repeating the expensive correlation pass.

The notebook reports expected full-run ranges of 89% to 91% test accuracy for both
endpoints, `67.3% +/- 5%` for the aligned but unrepaired midpoint, and `84.9% +/- 1%`
after REPAIR. These are useful acceptance bands, not deterministic test assertions.
The paper evaluates MLPs, VGG11 through VGG19, ResNet18, ResNet20, ResNet50, and a
double-width ResNet50 on MNIST, FashionMNIST, SVHN, CIFAR-10, CIFAR-100 split data,
and ImageNet. Those experiments establish research coverage; the upstream repo does
not provide a reusable architecture-generic implementation.

## Focused verification

The MVP needs four checks:

1. Compare logits from B before and after permutation on a fixed batch. The maximum
   absolute difference should be at floating-point noise scale, and accuracy should
   remain unchanged.
2. Measure every repaired channel on held-out calibration inputs. Its mean and
   standard deviation should be close to the interpolated endpoint targets.
3. Compare explicit reset-layer logits with fused-convolution logits in evaluation
   mode. Use a tight numeric tolerance, rather than the notebook's looser accuracy
   comparison.
4. Run the end-to-end midpoint demo and record both loss and accuracy. REPAIR does
   not guarantee improvement on an arbitrary synthetic fixture. For a full CIFAR
   reproduction, compare repeated runs with the notebook's accuracy bands above.

Small synthetic tests can verify output/input permutation cancellation and the
convolution plus BatchNorm fusion equation. Full CIFAR-10 training is a slow
reproduction command, not a routine unit test.

## Licensing and provenance

The official REPAIR repository contains no `LICENSE`, `COPYING`, package metadata,
or file-level license notice at the pinned revision. Public visibility alone does
not grant a redistribution license. Treat the repository as a provenance reference,
not a licensed software dependency. Preserve its commit URL and cite the paper.
Implement the published equations and behavior in new code unless the repository
owner supplies reuse terms.

The notebook comments that its VGG definition comes from
[`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py),
whose repository has an [MIT license](https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE).
If code is copied from that source, retain its MIT copyright and license notice.
That MIT license does not license the separate REPAIR notebook code.

The paper should be cited as Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim
Entezari, and Behnam Neyshabur, "REPAIR: REnormalizing Permuted Activations for
Interpolation Repair," ICLR 2023, arXiv:2211.08403.
