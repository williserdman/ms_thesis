# Endpoint tuning verification

Checkpoint: 2026-09-18. Tuning reuses the thesis `src/optuna_trainer.py` hook;
the original MyModel entry point retains its positional interface and default
search. Legacy-specific Lightning/model imports are deferred until needed.

The full suite passed: **34 tests in 6.467 seconds**, after imports. The saved
log is `runs/endpoint-tuning-tests.log`. New tests use real Optuna SQLite studies
and verify completed-budget reuse with no objective calls, budget extension,
reopening the same best trial, independent study names, baseline enqueueing,
fixed dimensions, test-target exclusion, and a complete tuning-to-path run.
Existing model, BatchNorm, endpoint isolation, and REPAIR tests also pass.

A real Cora smoke run then exercised the original thesis loader twice:

- `runs/tuning-cache-smoke-first/report.json`: two new trials.
- `runs/tuning-cache-smoke-cached/report.json`: zero new trials, cache hit, same
  key and best parameters, same total of two finished trials.

The smoke cache is separate from full-budget GPU searches. Each full search uses
20 total trials, tuning seed 42, the reference dataset's endpoint epoch budget,
and the original masks. The objective maximizes calibrated selected-endpoint
validation accuracy. It does not compute test metrics. Final endpoint seeds
0–5 are trained independently after selection. Bézier fitting stays at its
existing 200 steps and learning rate 0.01.

All four completed Cora full-budget searches were also reopened through the GNN
adapter, with endpoint training patched to fail if called. Every search returned
the same parameters with `cache_hit=true`, `new_trials=0`, and 20 finished trials.
The record is `results/tuned-endpoints-20260918/cache_reuse.json`.

All 16 tasks in array **3834945** completed with exit code 0. Each configuration
used 20 trials including pruned trials, six final endpoint seeds, three pairs,
200 Bézier steps and 21 evaluation points. See the
[tuned reports, parameters and graphics](../results/tuned-endpoints-20260918/README.md).
The separate fixed-reference matrix, array 3834822, is also complete; see
[its results](reference_full_results.md).

The archive audit passed for all 16 reports against their persisted studies,
source epoch budgets, effective model settings, unchanged dataset identities,
and reported endpoint/path boundaries. Across 320 trials, 173 completed and
147 were pruned; none failed. Maximum reported boundary loss difference was
4.77e-7, with no accuracy difference. This audit does not replay every checkpoint.
The endpoint comparison and labeled path graphics were generated from these
reports. Cache databases and model weights remain local; the branch includes
selected parameters, reports, PNGs, and PDFs.
