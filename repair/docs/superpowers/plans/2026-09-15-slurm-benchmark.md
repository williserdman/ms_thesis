# Slurm benchmark plan

Goal: run the VGG11/CIFAR-10 baseline on one GPU, then produce loss and accuracy
curves for 21 coefficients. User authorized Slurm submission and a goal that
continues after the introductory checks pass.

- [x] Inspect GPU partitions and existing environment.
- [x] Prepare project-local dependencies and download CIFAR-10.
- [x] Add PNG/PDF plotting from report JSON and a Slurm pilot/full script.
- [x] Add training, alignment, curve timing and GPU memory metadata.
- [x] Verify the focused tests and plots; submit the two-epoch pilot.
- [x] Check pilot CUDA execution, finite metrics, memory and runtime.
- [x] Submit 100 epochs for each of two endpoints and 21 alpha values.
- [x] Retrieve the report and figures, explain measured barriers and limitations.

Use existing float32 training and the authors' optimizer schedule. The pilot uses
the same batch size, width and data size as the full run. It tests execution and
timing, not final accuracy. No broad refactoring or code-review cycle.

The full run remains one independently trained endpoint pair, not a statistical
reproduction across seeds. Queue wait is separate from measured execution time.

Audit update, 2026-09-16: revised pilot job `3826705` completed with exit code 0
in 2m40s on RTX 4090 node `gput075`. Application time was 137.465s, with warm
epochs at 3.42s, alignment at 37.907s, and the three-alpha curve at 9.002s.
Peak allocated CUDA tensor memory was 1,427,842,048 bytes. All three methods at
all three alpha values were finite across 10,000 test samples, and PNG and PDF
plots were produced. Full job `3826719` completed in 14m34s with exit code 0,
submitted with
`REPAIR_RUN_KIND=full sbatch --parsable --job-name=repair-full
--constraint=gpu4090 --time=01:00:00
--chdir=/home/wge3/ms_thesis/repair slurm/repair.sbatch`. It uses 100 epochs
per endpoint and 21 alpha values, with calibration on 5,000 images.

All full-run metrics were finite, the figures were generated and inspected, and
results are recorded in [the benchmark run record](../../benchmark-run.md).
