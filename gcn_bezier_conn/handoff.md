# Handoff

Updated: 2026-09-18. Branch: `exp/gnn-repair`.

## Publishing this branch

The implementation, result snapshots, and initial handoff were committed as
`cbfab3f`. The initial HTTPS push failed. The user subsequently configured SSH,
and GitHub SSH authentication is now verified. `origin` still uses HTTPS; use
this command to push over SSH without changing the stored remote configuration:

```bash
git -c 'core.sshCommand=ssh -o StrictHostKeyChecking=yes -o CheckHostIP=no -o BatchMode=yes' \
  -c remote.origin.pushurl=git@github.com:williserdman/ms_thesis.git \
  push -u origin exp/gnn-repair
```

Verify the remote branch points to local `HEAD` before reporting it as pushed.
Do not put credentials into this handoff, repository files, or chat.

## Next session

The current task extends the original linear and quadratic Bézier baseline to
GCN, MLP, GraphSAGE, and GAT on Cora, Roman-empire, squirrel, and chameleon. Keep
the completed legacy GCN and REPAIR artifacts valid. Reevaluate after the
baseline runs exist. Extending REPAIR to reference models or other architectures
remains deferred. REPAIR still applies only to legacy GCN linear paths, with the
saved Bézier path as comparator.

All 16 fixed reference-preset configurations have now completed full endpoint
budgets and three seed pairs each (Slurm 3834822). See
[full reference results](docs/reference_full_results.md). The earlier Cora pilots
and smoke checks remain archived separately.

The user requested Optuna and explicitly chose endpoint models only. The existing
thesis hook now supports the four architectures, persistent studies, and cached
best parameters. All 34 tests pass. A real Cora repeat reused its two-trial study
with zero new trials. The full tuned matrix is running as Slurm array 3834945,
under `runs/tuned-endpoints-20260918/`, with 20 trials/configuration and three final
endpoint pairs. Check its state before claiming completion. Bézier settings stay
fixed; tuning uses validation accuracy and never computes test metrics.

Cache location: `runs/optuna-cache/<dataset>/<architecture>/<context-hash>/`.
Keep `study.sqlite3` and `best.json`; deleting these discards reusable searches.
Identical completed budgets are reused; raising the trial budget extends the
study. Data, training/search settings, source implementation, runtime, and tuning
seed changes create new contexts. Do not alter tuning implementation while the
matrix is running; source hashes are part of cache identity.

The Cora Bézier paths have zero sampled training-loss barriers and relatively
flat test accuracy, but substantially higher validation/test cross-entropy than
their endpoints. Reevaluate curve training using training/validation data before
extending REPAIR. No settings were selected using test metrics. The labeled
[comparison](results/reference-architectures/README.md) records this result.

## Read first

- [ONBOARDING.md](ONBOARDING.md): environment, commands, code map, loader behavior.
- [Paper protocol](docs/paper_protocol.md): paper facts, missing details, current
  implementation choices. The scope is a procedure replication, not confirmed
  numerical reproduction.
- [Architecture design](docs/superpowers/specs/2026-09-18-architecture-baselines-design.md)
  and [implementation plan](docs/superpowers/plans/2026-09-18-architecture-baselines.md):
  agreed behavior and bounded work.
- [Architecture source check](docs/architecture_sources.md): official reference
  code, pinned commit, exact profiles, licensing, and source differences.
- [Dataset sweep](docs/dataset_sweep.md): completed four-dataset results and limits.
- [REPAIR integration](docs/repair_integration.md): current calibration and
  alignment semantics. The sibling [REPAIR handoff](../repair/handoff.md) covers
  that package's earlier work.
- [Saved results](results/gcn-four-datasets/README.md): tracked figures and reports.
- [Cached tuning design](docs/superpowers/specs/2026-09-18-endpoint-tuning.md):
  endpoint-only scope, search dimensions, objective, and cache identity.

A bounded first-party search found no public repository from the
mode-connectivity authors. This is not proof that none exists. Appendix B refers
to Luo et al., *Classic GNNs are Strong Baselines*, for architecture and
hyperparameter choices. Its official `tunedGNN` source is pinned at commit
`23f9604e8b13a9a6d3faa2f691cd844006979153` under `upstream/tunedGNN`, together
with its MIT License and provenance. The implementation adapts its models and
profiles to the existing PyTorch/PyG runner and thesis loader. It does not adopt
the upstream loader or graph preprocessing.

## Implemented behavior

1. `run --preset reference --architecture {gcn,mlp,graphsage,gat}` is available.
   Reference settings resolve per dataset. Explicit CLI overrides apply after
   the preset and the effective configuration is saved.
2. The existing default command and legacy GCN checkpoint replay are preserved.
   `--smoke` applies after preset resolution and caps width at 8, endpoint and
   curve training at 5 steps, evaluation at 5 points, and pairs at 1.
3. The thesis loader, graph edges, features, and masks are unchanged. The MLP
   substitutes linear operators into the reference GCN profile and ignores
   edges. This is a local choice because tunedGNN has no MLP recipe.
4. BatchNorm models calibrate each endpoint and path point with one
   full-graph, label-free forward, dropout disabled, fresh statistics, and no
   state leakage. This is transductive BatchNorm calibration, not REPAIR.
5. REPAIR rejects reference and non-GCN reports before creating output. Tests
   cover pinned upstream equivalence, MLP edge independence, path endpoints,
   control gradients, BatchNorm state isolation, and checkpoint replay.

Avoid silently carrying every GCN default into every architecture. The completed
sweep documents weak Roman-empire endpoints and Bézier overfitting on filtered
squirrel/chameleon. Any setting selection must use training/validation data;
test metrics remain evaluation-only.

## Workspace and checkpoint

Work in the existing linked worktree `<thesis-checkout>/.worktrees/gnn-repair`,
under `gcn_bezier_conn`. Exact machine paths and interpreter are in onboarding.
The original checkout remains on `exp/spectral-mc-pocs` with unrelated dirty
files. Do not switch, clean, reset, or commit that checkout's unrelated work.

This checkpoint includes the existing loader path/cache fallback fixes that
were already used by all recorded runs. They were copied unchanged into the
worktree so the branch contains the loader expected by `gcn_mc.data`; no data
pipeline was reimplemented. Saved-run replay must use the original loader/cache
location recorded in the report, or an explicit compatible `--thesis-root`.

Raw `runs/` artifacts and model checkpoints remain local and ignored. Compact
figures and JSON reports are tracked under `results/gcn-four-datasets/` and
`results/reference-architectures/`; those copies do not include model weights.
The tuned matrix is currently running; inspect Slurm array 3834945. Other user
jobs may share the account and must be left alone.

Before this extension, all 11 legacy GCN/REPAIR tests passed in 0.377 seconds
after imports. The dataset-sweep record documents successful Slurm jobs and
replay of all 12 repaired midpoint checkpoints. These are historical legacy
results. New runs are in `runs/reference-smoke-20260918/` (Slurm array 3834723)
and `runs/reference-cora-20260918/` (array 3834752). The latter uses seeds 0:1,
500 endpoint epochs, 200 curve steps, and 21 points per model. No packages or
machine configuration changed.

The intended reference command form is:

```bash
$PY -m gcn_mc run \
  --thesis-root /home/wge3/ms_thesis \
  --preset reference \
  --architecture graphsage \
  --datasets Roman-empire \
  --output runs/reference-roman-graphsage
```

Supported overrides cover width, depth, dropout, endpoint epochs and optimizer,
normalization, residual connections, input projection, GAT heads, and endpoint
selection. Only one GAT head is supported. `scripts/reference_baselines.sbatch`
maps array tasks 0–15 to four datasets by four architectures; tasks 0–3 are Cora.
See onboarding for commands and profile override rules.

## User preferences

Use the thesis data pipeline. Keep changes in a worktree. Prefer the smallest
runnable implementation, focused checks, and concise updates. Do not initiate
review cycles or broad hardening. Do not add AI-authorship attribution. The user
explicitly authorized committing and pushing this branch and requested this
handoff in the project folder rather than the skill's default temporary folder.

## Suggested skills

- `superpowers:using-superpowers` for session setup.
- `superpowers:brainstorming` and `superpowers:writing-plans` for the bounded
  architecture extension, using the already agreed baseline-first scope.
- `research` for author code and reference architecture settings.
- `superpowers:test-driven-development` for focused model/path behavior checks.
- `superpowers:verification-before-completion` before reporting success or pushing.
- `unslop` for concise documentation. Apply the user's MVP and no-review
  preferences over broader skill workflows.
