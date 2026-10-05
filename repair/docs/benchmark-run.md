# CIFAR-10 benchmark run

Run dates: 2026-09-15 to 2026-09-16. One independently trained pair of full-width VGG11 models.
Goal: compare unaligned interpolation, activation-aligned interpolation and REPAIR
at 21 coefficients, then save loss and accuracy figures.

## Full benchmark results

Job `3826719` completed with exit code 0 in **14 minutes 34 seconds** on one
RTX 4090. Each endpoint trained for 100 epochs. All 21 coefficients produced
finite metrics on the 10,000-image test set for all three methods.

| Model | Test cross entropy | Test accuracy |
| --- | ---: | ---: |
| Endpoint A | 0.514225 | 89.45% |
| Endpoint B | 0.497745 | 89.79% |
| Unaligned midpoint | 2.377994 | 10.00% |
| Aligned midpoint | 0.993516 | 68.23% |
| Aligned midpoint with REPAIR | 0.644068 | 84.30% |

REPAIR improved midpoint accuracy by 16.07 percentage points over alignment
alone. It reduced the sampled loss barrier by 71.68% and error barrier by 75.13%
relative to alignment alone. A residual barrier remains.

| Method | Loss barrier | Error barrier, percentage points |
| --- | ---: | ---: |
| Unaligned | 1.872009 | 79.637 |
| Aligned | 0.487531 | 21.390 |
| Aligned with REPAIR | 0.138083 | 5.320 |

The baseline at each coefficient is the linear interpolation of the original
endpoint metrics. A barrier is the largest loss increase or accuracy decrease
relative to that baseline across the 21 sampled coefficients.

![CIFAR-10 VGG11 interpolation curves](../runs/cifar10-3826719/interpolation.png)

Local artifacts:

- [PNG figure](../runs/cifar10-3826719/interpolation.png)
- [PDF figure](../runs/cifar10-3826719/interpolation.pdf)
- [JSON report](../runs/cifar10-3826719/report.json)
- Checkpoints in `runs/cifar10-3826719/`: `endpoint_a.pt`, `endpoint_b.pt`,
  `aligned_b.pt`, and `repaired_midpoint.pt`.
- [Slurm log](../slurm/logs/repair-full-3826719.out)

Artifacts under `runs/` and `slurm/logs/` are local and ignored by Git.

Application time was 851.60 seconds: endpoint A training 373.33 seconds,
endpoint B training 339.63 seconds, alignment 37.66 seconds, and curve evaluation
61.84 seconds. Remaining application time includes dataset setup, endpoint
evaluation, and saving. Slurm elapsed time also includes setup and plotting.
Peak allocated CUDA tensor memory was 1,427,842,048 bytes, about 1.33 GiB.

This is one independently trained pair with seed 0, not a repeated-seed
reproduction of all paper experiments. The run uses float32 and 5,000 fixed,
unaugmented training images for calibration/alignment. The authors' notebook
uses mixed precision and all 50,000 augmented training images. The measured
accuracies fall within its stated example ranges; see the
[source comparison](research.md). Graph adapters, repeated seeds, and other
paper architectures remain deferred.

## Setup and job record

- Environment and commands: [Slurm instructions](slurm.md).
- CIFAR-10 archive MD5 verified: `c58f30108f718f92721af3b95e74349a`.
- Dataset: 50,000 training and 10,000 test images.
- Focused suite: 17 tests passed, including PNG/PDF generation.
- CUDA preflight: job `3826613`, completed on an RTX 4090, batch size 500.
- Pilot: job `3826645`, two epochs per endpoint and three alpha values,
  submitted with a 30-minute limit.

The initial pilot exposed costly worker startup under Python 3.14's forkserver
method. Its first four training epochs took 38.86, 35.11, 37.08 and 36.04 seconds.
After alignment, repeated calibration/evaluation remained slow with low sampled
GPU utilization. The initial pilot completed successfully in 21 minutes 3 seconds,
with all metrics finite and both plot files generated. Its report recorded
265.06 seconds for alignment and 748.55 seconds for its three-point curve.

Data-only probe `3826698` completed in 4 minutes 28 seconds. On the same CPU node,
nonpersistent passes took 82.005 and 62.104 seconds; persistent passes took
75.235 seconds initially and 0.608 seconds on reuse. All four passes returned
identical 5,000-example input/label checksums. Worker startup accounted for nearly
all the delay. `persistent_workers` is now enabled for positive worker counts;
the REPAIR calculation and fixed calibration transforms are unchanged.

Revised pilot `3826705` completed successfully in 2 minutes 40 seconds on an
RTX 4090. Warm training epochs took about 3.42 seconds, alignment took 37.91
seconds, and the three-point curve took 9.00 seconds. Peak CUDA tensor allocation
was 1,427,842,048 bytes, about 1.33 GiB. Every metric was finite and both figures
were generated. Pilot accuracy is not the benchmark.

Full job `3826719` started on `gput075`, using one RTX 4090 and a one-hour limit:

```bash
REPAIR_RUN_KIND=full sbatch --parsable --job-name=repair-full \
  --constraint=gpu4090 --time=01:00:00 \
  --chdir=/home/wge3/ms_thesis/repair slurm/repair.sbatch
```

It used the 100-epoch optimizer schedule, 5,000 fixed training images for
alignment/calibration, and all 10,000 test images for loss and accuracy at 21
coefficients. Results are in `runs/cifar10-3826719/`, with the log at
`slurm/logs/repair-full-3826719.out`.
