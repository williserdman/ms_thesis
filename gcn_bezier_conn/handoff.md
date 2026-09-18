# Handoff

Updated: 2026-09-18. Branch: `exp/gnn-repair`.

## Push status

The implementation, result snapshots, and initial handoff were committed as
`cbfab3f`. The push to `origin` failed because HTTPS Git authentication is absent.
SSH authentication also failed; this session has no GitHub token, credential
helper/store, or SSH identity. The code is committed locally, not confirmed on
the remote. After GitHub authentication is available, run from the worktree:

```bash
git push -u origin exp/gnn-repair
```

Verify the remote branch points to local `HEAD` before reporting it as pushed.
Do not put credentials into this handoff, repository files, or chat.

## Next session

The user wants to preserve the current GCN/REPAIR work, then first reproduce the
original paper's linear and quadratic Bézier experiments with **GCN, MLP,
GraphSAGE, and GAT** on **Cora, Roman-empire, squirrel, and chameleon**. Reevaluate
after those baselines exist. Extending REPAIR to the other architectures is
deferred. The earlier choice still stands: REPAIR on linear paths only, with
Bézier as the comparator.

Only GCN is implemented today. No model-extension code was written before this
handoff. The next step is implementation, not another effort estimate.

## Read first

- [ONBOARDING.md](ONBOARDING.md): environment, commands, code map, loader behavior.
- [Paper protocol](docs/paper_protocol.md): paper facts, missing details, current
  implementation choices. The scope is a procedure replication, not confirmed
  numerical reproduction.
- [Architecture source check](docs/architecture_sources.md): newly identified
  official reference code, pinned commit, preliminary settings, and open checks.
- [Dataset sweep](docs/dataset_sweep.md): completed four-dataset results and limits.
- [REPAIR integration](docs/repair_integration.md): current calibration and
  alignment semantics. The sibling [REPAIR handoff](../repair/handoff.md) covers
  that package's earlier work.
- [Saved results](results/gcn-four-datasets/README.md): tracked figures and reports.

The previous paper search did not identify an official mode-connectivity code
repository. Appendix B refers to Luo et al., *Classic GNNs are Strong Baselines*,
for architecture and hyperparameter choices. The fresh check identified that
reference's official `tunedGNN` code. It uses deeper models and optional
normalization/residuals; see the source-check note before choosing the design.
The refreshed search for the mode-connectivity authors' own code is unfinished.
Preserve citations and licensing for reused source. The existing implementation
uses PyTorch/PyG operators and the thesis loader; it is not a checkout of the
mode-connectivity authors' code.

## Implementation starting point

1. Confirm available author/reference code and recover the model settings that
   are actually specified. Record any remaining choices explicitly.
2. Add the three model families with a common `forward(x, edge_index)` interface;
   MLP ignores edges. Add architecture selection to the CLI and saved model
   configuration. Preserve replay of existing GCN checkpoints.
3. Replace direct GCN construction in the baseline runner with model selection.
   Reuse the existing training, parameter interpolation, metrics, and thesis
   loader. Label reports and graphics by architecture.
4. Keep the REPAIR command GCN-only and reject other architectures clearly until
   a later task defines their alignment/calibration rules.
5. Verify path endpoints, control gradients, unchanged endpoint parameters, and
   checkpoint replay for each new architecture. Run small end-to-end checks
   before the three-pair dataset sweep. Update onboarding and report measured
   results before reevaluating REPAIR.

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
figures and JSON reports are tracked under `results/gcn-four-datasets/`; those
copies do not include model weights. No experiment jobs remain running.

Before this checkpoint, all 11 GCN/REPAIR tests passed again in 0.377 seconds
after imports. The dataset-sweep record documents successful Slurm jobs and
replay of all 12 repaired midpoint checkpoints. No packages or machine
configuration changed. No architecture-extension implementation plan has yet
been written; the existing plans describe the completed GCN work.

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
