# Endpoint tuning implementation plan

- [ ] Extend the existing thesis Optuna hook with reusable suggestions, custom
  objectives, and persistent total-budget studies; preserve the old interface.
- [ ] Add endpoint-only validation callbacks and a GNN tuning/cache adapter.
- [ ] Wire CLI, effective model settings, provenance, and fixed search overrides.
- [ ] Verify objective isolation, cache reuse/resume/invalidation, and end-to-end
  tuning with focused tests and a real cached repeat.
- [ ] Complete the reference-preset matrix already running; run tuned endpoints
  for all four architectures/datasets with three independent endpoint pairs.
- [ ] Save comparison artifacts, update onboarding/handoff, commit and push.

See the adjacent spec for the cache identity and user-approved endpoint-only scope.
The existing thesis data loader and Bézier optimizer are unchanged.
