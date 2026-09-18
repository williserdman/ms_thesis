# GCN mode connectivity

Core experimental procedure from [Unveiling Mode Connectivity in Graph Neural
Networks](https://arxiv.org/abs/2502.12608), using the existing `ms_thesis` data
pipeline. Train GCN endpoints, fit quadratic Bézier controls, and compare loss
and accuracy along straight and curved parameter paths.

This is a procedure replication. The paper omits training hyperparameters and
split details, so the implementation records its choices. The broader paper's
architecture, synthetic-graph, and generalization experiments remain deferred.

From this directory, use a Python environment containing PyTorch, PyG,
PyTorch Lightning, NumPy, and matplotlib:

```bash
python -m gcn_mc run --smoke --output runs/smoke
python -m gcn_mc run --datasets Cora --output runs/cora
python -m gcn_mc plot runs/cora/report.json
python -m unittest discover -s tests -v
```

The default run uses independent seed pairs `0:1 2:3 4:5`, 200 endpoint epochs,
200 curve steps per pair, and a 21-point evaluation grid. Use `--pairs 0:1` for
one pair, or `--device cuda` inside a GPU allocation. Each run requires a new or
empty output directory. `--help` lists all settings.

Results include endpoint and control checkpoints, exact split masks, JSON
metrics and training histories, and PNG/PDF plots. The loader preserves the
thesis preprocessing and masks. It does not implement the paper's data pipeline.

- [Developer onboarding](ONBOARDING.md)
- [Next-session handoff](handoff.md)
- [Paper protocol and gaps](docs/paper_protocol.md)
- [Verification record](docs/verification.md)
- [GCN REPAIR integration](docs/repair_integration.md)
- [REPAIR verification and results](docs/repair_verification.md)
- [Four-dataset GCN comparison](docs/dataset_sweep.md)
- [Saved figures and reports](results/gcn-four-datasets/README.md)

The REPAIR extension lives on branch `exp/gnn-repair` in
`/home/wge3/ms_thesis/.worktrees/gnn-repair`. Reuse the existing Cora checkpoints:

```bash
cd /home/wge3/ms_thesis/.worktrees/gnn-repair/gcn_bezier_conn
/home/wge3/miniconda3/envs/py312/bin/python -m gcn_mc repair \
  --source /home/wge3/ms_thesis/gcn_bezier_conn/runs/cora/report.json \
  --thesis-root /home/wge3/ms_thesis \
  --output runs/cora-repair
```

This compares raw linear, aligned linear, aligned plus REPAIR, and the existing
Bézier path. It calibrates hidden activations using training-node statistics
after graph aggregation. No endpoints or curves are trained by this command.
The sibling `repair/src/repair` source must remain available. In this worktree,
also pass `--thesis-root /home/wge3/ms_thesis` to ordinary `run` commands so they
use the current thesis loader.

For a separate environment, install this project with `python -m pip install -e .`.
The thesis checkout is still required; pass `--thesis-root /path/to/ms_thesis`
if it is not this directory's parent. No installation is needed to run from
this source directory in an existing compatible environment.
