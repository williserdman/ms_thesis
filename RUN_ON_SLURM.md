# Run the spectral × mode-connectivity PoCs on the SLURM machine

Branch: `exp/spectral-mc-pocs`. Everything below is run from the repo root.

## 1. Get the code + data on the cluster
```bash
git fetch && git checkout exp/spectral-mc-pocs      # if pushed; else scp/rsync the repo
# datasets live in data/ (tracked). If missing, the PyG loaders auto-download
# Planetoid/Heterophilous sets; the *filtered* squirrel/chameleon .npz must be present.
```

## 2. One-time environment
```bash
conda create -n ms_thesis python=3.10 -y
conda activate ms_thesis
pip install -r requirements.txt          # torch, torch_geometric, pytorch_lightning, optuna, numpy
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True on a GPU node
```

## 3. Point the launcher at your cluster  — EDIT `slurm/env.sh`
Set these to match the cluster (jobs fail at activation otherwise):
- `CONDA_ENV` (default `ms_thesis`)
- `module load cuda` line — change/remove for your module system
- in each `slurm/*.sbatch`: `--partition=gpu`, `--gres=gpu:1`, `--time`, `--mem`, `--cpus-per-task`

Override env name without editing: `CONDA_ENV=myenv bash slurm/submit_all.sh ...`

## 4. Smoke test FIRST (fast, real torch run)
```bash
bash slurm/submit_all.sh --smoke
```
Tiny config (cora, K=2, ~5 epochs, 2 seeds). Confirms every experiment runs
end-to-end before spending GPU-hours. Watch: `squeue --me`. Check `results/*_smoke.json`
and `slurm/logs/`.

## 5. Full runs
```bash
bash slurm/submit_all.sh
```
Submits 5 jobs **chained sequentially** (`--dependency=afterok`), each ≤1h on 1 GPU,
in order: `b1_barrier → idea06_subspace → idea09_frontier → idea15_steered → idea18_filtercurve`.
Datasets: cora, citeseer, squirrel, Roman-empire.

Run one experiment alone:
```bash
sbatch slurm/idea09_frontier.sbatch                 # full
sbatch slurm/b1_barrier.sbatch --smoke              # smoke
```
Override args (forwarded to the python module), e.g. fewer datasets / seeds:
```bash
sbatch slurm/b1_barrier.sbatch --datasets cora squirrel --seeds 0 1 2 3
```

## 6. Read results
- JSON per experiment → `results/<experiment>.json`
- stdout/stderr → `slurm/logs/<jobname>_<jobid>.out`
- **Read `results/b1_barrier.json` first.** If `coeff_barrier_nontrivial` is `false`
  for a dataset, the γ-axis is already near-linearly connected there — ideas 06/15/18
  largely degenerate to nulls on it; idea-09's weight-vs-filter gap is then the story.

## Common knobs (all experiments, via `slurm/*.sbatch ... <args>`)
`--datasets ...` `--seeds ...` `--K` `--basis cheb|mono` `--domain adj|lap`
`--hidden_dim` `--lr` `--dropout` `--epochs` `--patience` `--n_points` `--gpus` `--out` `--smoke`

## If a job dies
- activation error → `slurm/env.sh` conda/module lines wrong.
- CUDA OOM → lower `--hidden_dim`, or drop the big graph (`--datasets cora citeseer squirrel`).
- >1h on a sweep (idea-09 at K=16) → `--k_sweep 2 4 8` or `--max_pairs 2`.
- import error → ensure the job runs from repo root so `env.sh` can `cd src/`.
