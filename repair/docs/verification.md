# Verification

Date: 2026-09-16. Initial verification ran on CPU with Python 3.14.0 and
PyTorch 2.9.0+cu128.
The existing interpreter used on this machine was
`/home/wge3/miniconda3/envs/py312/bin/python`; despite the directory name, it is
Python 3.14. The local `.venv` uses Python 3.14 with PyTorch 2.9.0+cu128 and
torchvision 0.24.0+cu128. No shared environment changed.

## Focused tests

```bash
PYTHONPATH=src /home/wge3/miniconda3/envs/py312/bin/python \
  -m unittest discover -s tests -v
```

17 tests passed after the `persistent_workers=args.num_workers>0` fix. Coverage
includes a known channel permutation, VGG prediction
preservation, interpolation endpoints, target moments, sequential correction,
wrapped/fused equivalence for Linear and Conv2d, biasless fusion, invalid
calibration, and training through saved-checkpoint reload.

## Author-code comparison

Executed model/tracking/fusion definitions from cells 7, 21 and 26 of the preserved
VGG notebook. Replaced `.cuda()` with `.cpu()` for that comparison and calibrated
in float32 on two fixed batches of two random 32x32 RGB images. The local full-width
VGG11 REPAIR path produced identical state keys and a maximum absolute logit
difference of `0.0` on a separate random batch. This checks the upstream fast
calibration and fusion behavior; it is not a trained CIFAR accuracy experiment.

## End-to-end demo

```bash
PYTHONPATH=src /home/wge3/miniconda3/envs/py312/bin/python \
  -m repair demo --device cpu --output runs/demo
```

The default 20-epoch run completed and saved all four checkpoints and
`runs/demo/report.json`. On its 256 held-out synthetic examples:

| Model at alpha=0.5 | Cross entropy | Accuracy |
| --- | --- | --- |
| Unaligned interpolation | 0.776196 | 0.714844 |
| Aligned interpolation | 0.221194 | 0.906250 |
| Aligned plus REPAIR | 0.216371 | 0.902344 |

REPAIR reduced loss in this run; it did not improve every accuracy measurement.
This fixture demonstrates the implementation and makes no claim about the paper's
CIFAR-10 or ImageNet benchmark results. Full CIFAR training and CUDA execution
were not run during the initial verification.

The same demo also completed with `--method sequential`, writing
`runs/demo-sequential/report.json`. At the midpoint its cross entropy was
`0.216518` and accuracy was `0.902344`.

## Packaging

`python -m pip wheel --no-deps --no-build-isolation .` built
`repair_interpolation-0.1.0-py3-none-any.whl`. Running `python -m repair --help`
with that wheel on `PYTHONPATH` from outside the source tree succeeded. The wheel
contains the source-provenance and MIT notices. Local links in the README,
onboarding guide and API contracts resolve.

## Slurm GPU verification

Revised pilot job `3826705` completed with exit code 0 in 2m40s on an RTX 4090
(`gput075`). Application time was 137.465s, including 3.42s for warm epochs,
37.907s for alignment, and 9.002s for the three-alpha curve. Peak allocated
CUDA tensor memory was 1,427,842,048 bytes. All three methods at all three alpha
values produced finite metrics across 10,000 test samples, and both PNG and PDF
plots were produced.

Full job `3826719` completed with exit code 0 in 14m34s. It trained each of two
endpoints for 100 epochs, evaluated 21 interpolation coefficients, and calibrated
on 5,000 images. All 63 method/coefficient measurements were finite and used
10,000 test images. All four checkpoints and both plot files were generated;
the PNG was visually inspected. Endpoint accuracy was 89.45% and 89.79%.
Midpoint accuracy was 10.00% unaligned, 68.23% aligned, and 84.30% repaired.
Full metrics, timings, limitations, and artifacts are in the
[benchmark run record](benchmark-run.md).
