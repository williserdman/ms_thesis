# ms_thesis

Graph neural network experiments on heterophilous node-classification benchmarks,
built around a diffused-attention model with learnable spectral (Laplacian / Arnoldi)
filters, trained with PyTorch Lightning and tuned with Optuna.

## Layout

```
src/
  main.py                  # entrypoint: dispatches one Optuna study per dataset, multi-GPU aware
  optuna_trainer.py        # OptunaTrainer: hyperparameter search + best-model test
  args.py                  # MyArgs / SimplifiedArgs: filter + spectral config
  loading/
    LightningGraphLoader.py # dataset download + LightningDataModule
    DatasetInfo.py          # num_classes / num_features / class_weights container
  models/
    MyModel.py                        # LightningModule wrapper (loss, train/val/test steps)
    heterophily_diffused_attention.py # DiffusedAttention core model
    maxcut.py                         # maxcut / clustering helper
data/                       # benchmark datasets (tracked; some are slow to regenerate)
requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python main.py
```

`main.py` runs an Optuna study per dataset listed in `ALL_DATASETS` (edit that list to
change which datasets run). It auto-detects GPUs and dispatches one job per free GPU,
falling back to sequential CPU. Results are written to `training_results_<timestamp>.json`
(gitignored).

## Datasets

The `ALL_DATASETS` list in `main.py` selects which benchmarks run. Supported set includes:
Questions, Roman-empire, Amazon-ratings, Tolokers, computers, photo, texas, cornell,
Cora, Citeseer, Pubmed, squirrel, chameleon, actor, Minesweeper. Datasets live under
`data/` and are tracked in git (some are painful to re-download/regenerate).

## Starting a new experiment

This branch (`clean-main`) is the clean baseline. For each new experiment:

```bash
git checkout clean-main
git checkout -b my-experiment
```

Most experiments change the model in `src/models/heterophily_diffused_attention.py`
(the `DiffusedAttention` core) and/or `src/models/MyModel.py`. Existing experiment
branches show the pattern:

- `polyattn`     — polynomial attention
- `clusterattn`  — cluster-as-bias attention
- `clusterdiff`  — cluster-based diffusion
- `littleRNN`    — RNN variant (`src/models/littleRNN.py`)
- `bigger_gcn`   — larger GCN baseline (`src/models/big_gcn.py`)
- `arnoldi_init` — Arnoldi-initialized per-cluster filters (also carries extra analysis tooling)

Keep experiment-specific scratch (result JSON, plots, sweeps) out of git — add patterns
to `.gitignore` rather than committing artifacts.
