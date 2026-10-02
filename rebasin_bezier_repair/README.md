# Re-Basin, Bezier, and REPAIR

This package connects two trained PyTorch models in three stages:

1. Git Re-Basin aligns the second endpoint with the first.
2. A quadratic Bezier path fits one control state while the endpoints stay fixed.
3. REPAIR calibrates any requested interior model and folds its corrections into
   that model's parameters.

Architecture adapters keep model-specific tensor axes, execution, repair sites,
and correction rules out of the shared pipeline. The included adapters support
sequential ReLU MLPs and plain PyTorch Geometric GCNs.

## Install

The project depends on the sibling `git_re_basin` checkout. From this directory,
install both packages in editable mode:

```bash
python -m pip install -e ../git_re_basin -e .
```

Install the optional GCN dependencies with:

```bash
python -m pip install -e ../git_re_basin -e '.[gcn]'
```

Importing `rebasin_bezier_repair` and using the MLP adapter do not require
PyTorch Geometric. Import `GCNAdapter` from
`rebasin_bezier_repair.adapters.gcn` only when the optional dependency is
installed.

## Connect two models

`train_data` and `calibration_data` must be finite, nonempty, re-iterable batch
sources. The loss callback receives `(logits, batch)` and returns a scalar loss.

```python
from rebasin_bezier_repair import MLPAdapter, connect

path = connect(
    model_a,
    model_b,
    adapter=MLPAdapter(),
    train_data=train_batches,
    loss_fn=loss_fn,
    calibration_data=calibration_batches,
)

midpoint = path.model_at(
    0.5,
    repair=True,
    calibration_data=calibration_batches,
)
```

`model_at` returns an independent model in evaluation mode. Exact endpoints do
not need recalibration. Each repaired interior sample measures its own moments
after each preceding correction. The resulting repaired family is generally not
a quadratic Bezier parameter curve. Save a returned model with ordinary
`state_dict` operations.

The pipeline interpolates floating parameters only. Other state is fixed and
must match between aligned endpoints. Native BatchNorm models are not supported.
The supplied adapters reject unsupported structures instead of inferring tensor
couplings from names.

## Run from a source checkout

These commands run both synthetic CPU examples and the focused tests in the thesis environment:

```bash
PYTHONPATH=src:../git_re_basin/src /home/wge3/ms_thesis/repair/.venv/bin/python examples/mlp.py
PYTHONPATH=src:../git_re_basin/src /home/wge3/ms_thesis/repair/.venv/bin/python examples/gcn.py
PYTHONPATH=src:../git_re_basin/src /home/wge3/ms_thesis/repair/.venv/bin/python -m unittest discover -s tests -v
```

The examples verify the software flow; their metrics are not paper-replication benchmarks.

See [Adding an architecture](docs/adding-an-architecture.md) for the adapter
contract and focused completion checks.

## Source provenance and scope

Matching uses the public API from the sibling `git_re_basin` project. Quadratic
interpolation and control fitting are adapted from
`../gcn_bezier_conn/gcn_mc/paths.py` and its experiment code. Sequential REPAIR
and activation collection are adapted from that project's `repair.py` and
`activations.py`. Population-moment accumulation and affine folding follow
`../repair/src/repair/core.py`. Existing notices remain with their source
material.

Deferred work includes more architecture adapters, native BatchNorm policies,
other matching methods, REPAIR during curve optimization, full-path
serialization, benchmark reporting, and performance optimization.
