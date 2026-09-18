# Source provenance

Paper: [REPAIR: REnormalizing Permuted Activations for Interpolation Repair](https://arxiv.org/abs/2211.08403).

Author repository: [KellerJordan/REPAIR](https://github.com/KellerJordan/REPAIR).
Revision inspected: `e90263d7a4d48376091327274ae541d8d6d34743`, retrieved 2026-09-15.

`Train-Merge-REPAIR-VGG11.ipynb` and `README.original.md` are unchanged copies
from that revision. The notebook is the reference used for this implementation.
Notebook SHA-256: `80cc5392fef18c61ca3f539dc560dcd665d68552cb2d5a8c4fa594a21a214cc3`.

| Notebook cells | Local implementation |
| --- | --- |
| 7 | `src/repair/models.py`: normalization-free CIFAR VGG11 |
| 8 | `src/repair/experiment.py`: SGD, momentum, weight decay and learning-rate schedule |
| 11-16 | `src/repair/alignment.py`: correlations, assignment and channel permutation |
| 18 | `src/repair/core.py`: weight interpolation |
| 21-24 | `src/repair/core.py`: tracking, target moments and calibration |
| 26-27 | `src/repair/core.py`: folding corrections into affine layers |

Local changes include CPU support, a Python SDK and CLI, explicit calibration data,
copies instead of input mutation, Linear support, a sequential calibration option,
fixed calibration transforms, sample-weighted matching moments, and tests.
The synthetic experiment uses an MLP and Adam; it is a software demonstration.
Full CIFAR training uses float32 instead of the notebook's CUDA mixed precision.

The author repository provides no explicit software license at this revision.
This package does not assign a new license to upstream material. The VGG definition
in the notebook credits [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
Its MIT notice is preserved in `LICENSE.pytorch-cifar`.
