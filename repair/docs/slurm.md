# Slurm benchmark

## Environment on this cluster

The project `.venv` inherits the existing
`/home/wge3/miniconda3/envs/py312` environment and installs additions locally.
Despite the base directory name, its interpreter is Python 3.14.0.
Verified versions are PyTorch 2.9.0+cu128, torchvision 0.24.0+cu128,
NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.8. No shared environment was changed.

Creation commands, run once from `repair/`:

```bash
uv venv --python /home/wge3/miniconda3/envs/py312/bin/python --system-site-packages .venv
uv pip install --python .venv/bin/python --no-deps torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python --no-deps -e .
```

The package versions match the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/#v290).
On another machine, install matching PyTorch and torchvision builds appropriate
for its GPU driver, plus this package's `cifar` and `plots` extras.

Download CIFAR before requesting a GPU to avoid charging GPU time for downloads:

```bash
.venv/bin/python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True); CIFAR10('data', train=False, download=True)"
```

The official archive checksum is `c58f30108f718f92721af3b95e74349a`.

## Submit and inspect

Create the log directory once, then submit a two-epoch pilot from the repository root:

```bash
mkdir -p slurm/logs
sbatch --chdir=/home/wge3/ms_thesis/repair slurm/repair.sbatch
```

The pilot is the default. It evaluates interpolation coefficients `0`, `0.5`, and `1`. Inspect its runtime and output before choosing a time limit for the full run.

Submit the full 100-epoch benchmark with 21 coefficients from `0` through `1`:

```bash
REPAIR_RUN_KIND=full sbatch --parsable --job-name=repair-full \
  --constraint=gpu4090 --time=01:00:00 \
  --chdir=/home/wge3/ms_thesis/repair slurm/repair.sbatch
```

The default job requests one H100, L40S, or RTX 4090 GPU, four CPUs, 32 GB of RAM,
and four hours. The submitted full job used a one-hour limit. Slurm writes the
job log to `slurm/logs/repair-cifar10-JOB_ID.out`. The benchmark writes artifacts
to `runs/cifar10-JOB_ID/`; the script refuses to reuse that directory. After
training, it runs the plotting module on `report.json`, which writes
`interpolation.png` and `interpolation.pdf` beside the report.

The full run trains one pair of networks with one seed. It does not estimate confidence intervals or variation across independently trained pairs.

The report includes epoch durations for each trained endpoint, alignment and curve
times, total elapsed seconds, GPU model and peak allocated CUDA tensor memory.
The memory number does not include all driver or allocator-reserved memory.

CIFAR loaders keep their worker processes alive when `--num-workers` is positive.
This avoids repeatedly importing Python/PyTorch and copying datasets for every
epoch and calibration pass. The loader probe confirmed identical calibration
inputs with and without worker reuse. Training augmentation still samples random
crops/flips; worker RNG streams continue across epochs.

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
.venv/bin/python -m repair.plotting runs/cifar10-JOB_ID/report.json
```

Audit date: 2026-09-16. Revised pilot job `3826705` completed with exit
code 0 in 2m40s on RTX 4090 node `gput075` (application time 137.465s). Warm
epochs took 3.42s, alignment took 37.907s, and the three-alpha curve took 9.002s.
Peak allocated CUDA tensor memory was 1,427,842,048 bytes. All three methods at
all three alpha values produced finite metrics across 10,000 test samples, and
both PNG and PDF plots were produced.

Full job `3826719` completed in 14m34s with exit code 0. Submission command:

```bash
REPAIR_RUN_KIND=full sbatch --parsable --job-name=repair-full \
  --constraint=gpu4090 --time=01:00:00 \
  --chdir=/home/wge3/ms_thesis/repair slurm/repair.sbatch
```

It trains each of two endpoints for 100 epochs, evaluates 21 interpolation
coefficients, and calibrates on 5,000 images.

See the [run record](benchmark-run.md) for measured accuracy, barriers, timings,
and links to the PNG/PDF figures and JSON report.
