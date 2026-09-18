# REPAIR

PyTorch implementation of [REPAIR](https://arxiv.org/abs/2211.08403), based on the
[authors' source](https://github.com/KellerJordan/REPAIR) at
`e90263d7a4d48376091327274ae541d8d6d34743`.

Train two networks, align their hidden channels, interpolate their weights, then
restore hidden activation means and standard deviations. Includes VGG11/CIFAR-10,
a small MLP demo, and a reusable calibration API.

```bash
python -m pip install -e .
python -m repair demo --device cpu --output runs/demo
```

The command writes endpoint, aligned, and repaired checkpoints plus `report.json`
with unaligned/aligned/repaired interpolation curves. Synthetic results test the
pipeline; they do not reproduce the paper's benchmark accuracy.

- [Onboarding](ONBOARDING.md): setup, commands, code map, and graph integration path.
- [API contracts and SDK](API_CONTRACTS.md): signatures, tensors, state and artifacts.
- [Research notes](docs/research.md): equations and paper/source details.
- [Source provenance](upstream/PROVENANCE.md): pinned notebook and attribution.
- [Verification](docs/verification.md): checks performed and measured demo results.
- [Slurm benchmark](docs/slurm.md): cluster environment, pilot/full jobs and figures.
- [Benchmark results](docs/benchmark-run.md): completed VGG11/CIFAR-10 run and graphics.

```bash
python -m unittest discover -s tests -v
```

Generate loss and accuracy figures from a saved report:

```bash
python -m pip install -e '.[plots]'
python -m repair.plotting runs/demo/report.json
```

Implemented experiment scope is sequential MLPs and normalization-free CIFAR VGG11.
ResNet/ImageNet experiments, graph-model adapters, and closed-form REPAIR remain
deferred. The original VGG notebook is included in `upstream/`.
