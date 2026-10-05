# Handoff

Updated 2026-10-05. The approved framework MVP is complete. No additional
implementation is pending; the next session can use it for experiments or adapt
another architecture when requested. Keep changes small and runnable. Start
review or hardening work only when requested.

## Read first

- [README](README.md): usage, installation, verified run commands, source
  provenance, and deferred work.
- [Architecture guide](docs/adding-an-architecture.md): the adapter contract,
  working templates, and completion checks for a new model family.
- [Approved design](docs/superpowers/specs/2026-09-29-rebasin-bezier-repair-design.md):
  numerical behavior and scope boundaries.
- [Completed plan](docs/superpowers/plans/2026-10-02-rebasin-bezier-repair.md):
  implementation tasks and original verification evidence.

The MLP and GCN examples are software flow checks, not paper benchmarks. For
experiments, replace their endpoints and data callbacks through the same public
pipeline. Consult the README and design before assuming support for another
architecture or normalization scheme.

## Repository state

This directory belongs to the parent thesis monorepo at `..`. The framework and
this handoff were integrated locally into `main` through a project-only branch;
no push was requested. Inspect `git log main -- rebasin_bezier_repair` from the
repo root for the integrated history.

The original checkout remains on `exp/spectral-mc-pocs`, which contains unrelated
work and uncommitted changes. Preserve those files and the sibling projects.
The temporary integration branch and worktree were removed. Other registered
worktrees belong to separate tasks.

The sibling source dependencies are local and absent from fresh worktrees.
When testing elsewhere, use the existing interpreter from the README and an
absolute path to the sibling Git Re-Basin source in `PYTHONPATH`.

## Suggested skills

- New architecture or behavior: `superpowers:brainstorming`, then
  `superpowers:writing-plans` and `superpowers:test-driven-development`.
- Adapter interface changes: `codebase-design`.
- Test failures or numerical discrepancies: `superpowers:systematic-debugging`.
- Completion or integration claims: `superpowers:verification-before-completion`.
