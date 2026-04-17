# PolyFormer Mode Connectivity

This folder hooks up mode-connectivity analysis to checkpoints produced by two independent training runs.

## What it does

- Loads checkpoint pairs for run1 and run2 from `PolyFormer/node_classification/saved_models/<dataset>`.
- Reconstructs the selected model and dataset pipeline using the same data-loading code.
- Evaluates linear interpolation and trained Bezier interpolation in weight space.
- Saves three plots per run:
  - `linear_path.png`
  - `bezier_path.png`
  - `linear_vs_bezier.png`

## Supported datasets (initial scope)

- cora
- citeseer
- pubmed
- computers

## Run

From repository root:

```bash
python -m mode_connectivity.run_mode_connectivity --dataset cora --steps 21 --bezier_epochs 120
```

For GCN checkpoints:

```bash
python -m mode_connectivity.run_mode_connectivity --dataset cora --model gcn --steps 21 --bezier_epochs 120
```

Example with explicit checkpoints:

```bash
python -m mode_connectivity.run_mode_connectivity \
  --dataset citeseer \
  --checkpoint_a PolyFormer/node_classification/saved_models/citeseer/mono_run1_seed1941488137.pt \
  --checkpoint_b PolyFormer/node_classification/saved_models/citeseer/mono_run2_seed4198936517.pt
```

## Output

Plots are saved under:

- `mode_connectivity/outputs/<dataset>/`

## Notes

- If CUDA is not available, the script falls back to CPU automatically.
- Current hyperparameter reconstruction is based on `train_all_supported_two_seeds.sh` values for the local supported datasets.

## Checkpoint naming

- PolyFormer: `<base>_run{1|2}_seed{seed}.pt`
- GCN: `gcn_run{1|2}_seed{seed}.pt`
