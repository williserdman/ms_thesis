# REPAIR implementation design

Implement arXiv:2211.08403 using KellerJordan/REPAIR at commit
`e90263d7a4d48376091327274ae541d8d6d34743`. The user confirmed the paper
implementation comes before graph-model integration.

The standalone Python package provides the authors' CIFAR VGG architecture,
activation-correlation matching with Hungarian assignment, parameter interpolation,
and activation-statistics repair. Copy inputs before changing weights or modes.
Use PyTorch and SciPy, with torchvision optional for CIFAR-10 experiments.

REPAIR sets each selected channel's target mean and standard deviation to the
weighted endpoint means and standard deviations. The fast calibration path uses
the upstream BatchNorm approximation. A sequential path measures each layer after
earlier corrections, as described in the paper. Corrections can be fused into
linear/convolution layers. Explicit layer names keep correction separate from
architecture-specific permutation logic.

The runnable path trains two small classifiers on synthetic data, aligns, merges,
repairs, evaluates a coefficient sweep, and writes JSON. A CIFAR-10 command uses
the upstream VGG11 training recipe and saves reusable state dictionaries. Synthetic
results are software validation, not reproduction of the paper's accuracy numbers.

Deliver onboarding and API/SDK Markdown covering inputs, outputs, state changes,
calibration requirements, checkpoint loading, limitations, and graph integration.
Preserve the original VGG notebook with a pinned provenance note.

Focused verification: known permutations preserve predictions; interpolation has
correct endpoints; calibration matches endpoint target moments; fused and wrapped
predictions agree; the CPU demo completes. Defer full ImageNet/ResNet reproduction,
graph permutation adapters, distributed training, and benchmark accuracy claims.
