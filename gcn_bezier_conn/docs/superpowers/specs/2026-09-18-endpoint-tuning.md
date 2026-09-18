# Cached endpoint tuning

The user requested Optuna tuning using existing thesis hooks, with reusable
cached hyperparameters. They explicitly selected endpoint models only. Keep
Bézier fitting fixed. Continue the already launched reference-preset matrix as
the untuned comparator; then run the same matrix with tuned endpoints.

Extend the worktree's `src/optuna_trainer.py`, preserving its legacy MyModel
interface. Reuse its common learning-rate, width, and dropout suggestions and
MedianPruner. Add a custom-objective/storage interface instead of another Optuna
study runner. The GNN adapter supplies the existing factory and training loop.

Search learning rate, width, dropout, depth (2/3/4 plus the reference depth), and
weight decay (0, .00005, .0005, .001, .005). Hold normalization, residuals, input
projection, and attention heads at the reference profile. Enqueue that profile
as the first trial. Default to 20 trials and tuning seed 42; retain the dataset's
full endpoint epoch budget. Optimize calibrated selected-endpoint validation
accuracy; train on the train mask only. Intermediate validation accuracy supports
pruning every ten epochs and at the final epoch, limiting database writes on the
shared filesystem. Do not compute test metrics inside tuning. Final experiments retrain
independent endpoints with seeds 0–5 using one selected configuration per dataset
and architecture.

Store one SQLite study and readable `best.json` per cache identity, defaulting
to ignored `runs/optuna-cache/`. Identity includes graph/features, train/validation
labels and masks, architecture/fixed settings, epoch budget, search definition,
tuning seed, loader and implementation hashes, and relevant runtime versions and
device type. Test labels, final endpoint seeds, Bézier settings, output paths,
and requested trial count do not define endpoint tuning. An identical completed
budget performs zero new trials. Increasing the trial count resumes the study.
Use a per-study filesystem lock to avoid duplicate simultaneous searches; mark
interrupted running trials failed when the next exclusive owner resumes.

CLI: `run --preset reference --architecture NAME --tune-endpoints`, with
`--tuning-trials`, `--tuning-seed`, and `--tuning-cache`. Save chosen parameters,
objective, cache identity/path, trial count, and hit/miss in the report. Explicit
overrides of searched parameters define fixed values (remove those dimensions
from the search); smoke width caps restrict available widths and use a distinct
epoch/search identity. Reject tuning for the legacy model family.

Verify persistent reuse/resume with real cheap Optuna objectives, no test-label
influence on identity/objective, correct override precedence, and one complete
tiny-graph tuning-to-connectivity run. Reuse existing pipeline tests. Verify a
second real run reports a cache hit without new trials. Archive full fixed and
tuned results and update onboarding/handoff; commit and push the authorized
branch. No new REPAIR adapters or machine configuration changes.
