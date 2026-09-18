# ms_thesis

## Fixed spectral model

`src/models/fixed_spectral.py` applies fixed GArnoldi-style low/high-pass
filters to the original graph features, concatenates the results, and trains a
two-layer MLP. Only the MLP learns. Filtered features are cached for each graph.
The model uses the existing dataset loader, masks, weighted cross-entropy, and
PyTorch Lightning training.

Run commands from the repository root in an environment with `requirements.txt`
dependencies. The available environment uses Python 3.14, torch 2.9.0,
torch-geometric 2.7.0, pytorch-lightning 2.5.5, and Optuna 4.7.0.

### Development using saved hyperparameters

```bash
python src/train_spectral.py --datasets Cora Roman-empire squirrel chameleon
```

This reads `configs/fixed_spectral/<dataset>.json` and trains a fresh predictor.
It does not import or run Optuna. For a short local run:

```bash
python src/train_spectral.py --datasets Cora --epochs 10 --accelerator cpu
```

### Tune and replace the saved hyperparameters

```bash
python src/train_spectral.py --datasets Cora Roman-empire squirrel chameleon --optuna
```

Defaults are 20 trials per dataset, up to 300 epochs per trial, and early stopping
after 50 epochs without validation-loss improvement. Use `--trials`, `--epochs`,
and `--patience` to change the budget. Optuna searches hidden width, learning
rate, dropout, and weight decay. Polynomial degree 10 and the low/high-pass
filter bank stay fixed.

Selection uses the lowest validation loss, and only the selected checkpoint is
evaluated on the test split. Each dataset config records the chosen parameters,
seed, training budget, validation score, test metrics, and source commit.
Running with `--optuna` replaces that dataset's saved config.

TensorBoard logs, checkpoints, trial summaries, and run configs go under
`spectral_runs/<dataset>/<timestamp>/`. Override locations with `--config-dir`
and `--output-dir`. Checkpoints and logs are ignored by Git; selected configs
are checked in.

### Separate low-pass, high-pass, and band-pass models

Use `--filter low-pass`, `--filter high-pass`, or `--filter band-pass` to train
one fixed filter followed by its own predictor. Each dataset/filter pair has an
independent Optuna search. For example:

```bash
python src/train_spectral.py --datasets Cora --filter band-pass --optuna --trials 20 --epochs 500 --patience 500
```

Repeat for the three filters and four datasets to select 12 trained models.
On Slurm, submit each dataset/filter pair as a separate job requesting one GPU.
Patience 500 lets every trial complete the full 500 epochs. Selection still
uses the checkpoint with the lowest validation loss.

Configs live at `configs/fixed_spectral/<filter>/<dataset>.json`; each records
the winning checkpoint in `selection.checkpoint`. Logs and checkpoints live at
`spectral_runs/<filter>/<dataset>/<timestamp>/`, or below `--output-dir`.
To train a fresh predictor with a saved single-filter config, omit `--optuna`:

```bash
python src/train_spectral.py --datasets Cora --filter band-pass
```

### Filter convention

The graph is symmetrized, and the filters use its symmetric normalized
Laplacian. Low-pass targets `exp(-10 * lambda**2)`; high-pass targets its
complement. Band-pass targets `exp(-10 * (lambda - 1)**2)`.
Arnoldi interpolation uses Chebyshev sample nodes on `[0, 2]` and
retains both coefficients and their recurrence. There is no Jackson damping.
This preserves the initialization's polynomial basis rather than copying
coefficients into an unrelated propagation rule.

The existing loader's splits and class weights are unchanged. Roman-empire,
squirrel, and chameleon use split index 1; Cora uses its public split. Class
weights use all node labels. These are single-seed development settings, not
multi-split benchmark estimates. A numerical filter check and the tuning/config
round trip can be run with:

```bash
python -m unittest discover -s tests -v
```

Deferred: broader filter banks, multi-seed/split selection, and persistent Optuna
studies. The older attention experiment remains available through `src/main.py`.
