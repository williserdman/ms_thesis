# REPAIR handoff

Updated 2026-09-17. Workspace: `~/ms_thesis/repair`. Paths below are relative to
that workspace unless stated otherwise.

## Current state

The requested implementation, onboarding/API documentation, and Slurm CIFAR-10
benchmark are complete. The tracked goal is marked complete. No job or coding
task remains active. The user now requested this handoff; no next-session task
was specified.

Original request: implement <https://arxiv.org/abs/2211.08403>, use the authors'
source when available, and provide onboarding and API/SDK Markdown. The user
then authorized the CIFAR benchmark on Slurm, including plots. Authors' source:
<https://github.com/KellerJordan/REPAIR>.

The user eventually wants integration with thesis graph models, but explicitly
chose to implement and validate the paper baseline first. Graph integration is
future work, not an unfinished part of the completed benchmark.

## Read these artifacts

| Need | Source of truth |
| --- | --- |
| Package overview and commands | `README.md` |
| Setup, code map, graph integration considerations | `ONBOARDING.md` |
| Python SDK signatures, tensor assumptions, artifact contracts | `API_CONTRACTS.md` |
| Completed benchmark, figures, metrics, timings, caveats, pilot history | `docs/benchmark-run.md` |
| Cluster environment and submission commands | `docs/slurm.md` |
| Tests, upstream comparison, packaging and GPU verification | `docs/verification.md` |
| Paper equations, upstream differences and licensing findings | `docs/research.md` |
| Pinned authors' source and notebook | `upstream/PROVENANCE.md`, `upstream/` |
| Accepted design | `docs/superpowers/specs/2026-09-15-repair-design.md` |
| Completed implementation and benchmark plans | `docs/superpowers/plans/2026-09-15-repair.md`, `docs/superpowers/plans/2026-09-15-slurm-benchmark.md` |

Use those records rather than reconstructing the implementation or rerunning the
benchmark. The package source is `src/repair/`; tests are in `tests/`.

## Latest conversation

The user asked what the other `runs/cifar10-*` directories were. We explained
that their suffixes are Slurm job IDs, the earlier two are pilots, and the last
is the final benchmark. Their settings and history are already recorded in
`docs/benchmark-run.md`.

Final artifacts are in `runs/cifar10-3826719/`, including `report.json`,
`interpolation.png`, `interpolation.pdf`, and four checkpoints. Pilot directories
are `runs/cifar10-3826645/` and `runs/cifar10-3826705/`. We said the pilots can be
deleted without affecting the final run; the user has **not requested deletion**.

The final plot and results were delivered. Graph adapters and repeated-seed
experiments remain deferred; see the existing docs for scope and limitations.

## Working constraints

- Keep responses extremely concise. Prefer a runnable MVP, small reversible
  changes, and focused verification. Do not start review cycles or hardening
  unless requested.
- Do not add AI authorship or co-authorship attribution.
- The parent thesis repository has unrelated changes. `repair/` currently appears
  as untracked in Git. No commits were created. Preserve unrelated work.
- Run outputs, datasets, the local environment, and Slurm logs are Git-ignored;
  they exist on this machine, not in a fresh clone.
- Use `.venv/bin/python` for this project. Environment details and reproducible
  setup are in `docs/slurm.md`; do not change the shared base environment.
- The user said to ignore the earlier cross-account `SETUP_REPLICATION.md`
  question. Do not reopen that resolved question for this project work.
- Delegate independent bounded tasks when useful. User prefers Sol for ambiguous
  work and Luna only for fully specified tasks with objective checks.
- During this handoff, login-shell tool calls were slow and `rg` was unavailable.
  `exec_command` with `login: false` and standard `find`/`head` worked. This did
  not require configuration changes.

## Suggested skills

- `superpowers:using-superpowers` and `unslop` for the next session's workflow
  and concise writing.
- For a future graph integration request: `superpowers:brainstorming` and
  `codebase-design`, after reading the graph section in `ONBOARDING.md`.
- `superpowers:writing-plans` if the next approved change needs multiple steps.
- `superpowers:verification-before-completion` before claiming new work passes.
- `diagnosing-bugs` or `superpowers:systematic-debugging` only if a new failure
  occurs. The previous loader startup issue is resolved and documented.

Do not restart completed introductory steps or create another goal without a new
request. Await the user's next concrete task.
