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

The full tuned matrix was submitted as array **3834945**. Completion and resulting
metrics will be recorded after all 16 reports exist. The separate fixed-reference
matrix (3834822) is complete; see [its results](reference_full_results.md).
