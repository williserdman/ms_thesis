# Re-Basin, Bézier, and REPAIR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a reusable alignment → quadratic Bézier → post-hoc REPAIR pipeline with architecture adapters and runnable MLP/GCN examples.

**Architecture:** A shared pipeline imports Git Re-Basin matching and delegates model execution, permutation descriptions, and correction locations to explicit adapters. Generic path fitting and calibration operate on full named states. Task/data loss and calibration selection are caller supplied.

**Tech Stack:** Python >=3.10, PyTorch >=2.2, sibling `git-re-basin-thesis` >=0.1.0, standard-library unittest; optional torch-geometric >=2.6.

**Spec:** [Approved design](../specs/2026-09-29-rebasin-bezier-repair-design.md)

## Global Constraints

- Distribution `rebasin-bezier-repair`; import package `rebasin_bezier_repair`; conventional `src/` layout.
- Import public matching/state validation from the sibling package. Adapt only needed curve/calibration functions; preserve local source attribution and notices.
- Keep PyG optional. The package root and MLP adapter must import without it.
- Interpolate floating parameters only. Non-parameter state stays fixed and must match A and aligned B. Native BatchNorm models are unsupported.
- Caller endpoints' parameters, buffers, gradients, and training flags remain unchanged. Work on owned copies and detached endpoint states.
- Adam curve defaults: 100 steps, learning rate 0.01, one uniform t and one training batch per step, final control retained.
- Sequential population-moment REPAIR uses epsilon 1e-8; exact endpoints bypass correction; repaired samples are not an exact quadratic curve.
- Deliver package, two CPU synthetic examples, README, and adapter guide. Keep benchmark runners, plotting, Slurm, additional matching methods, and full-path serialization deferred.
- Follow the user's MVP instruction: focused verification; no review-agent cycles or hardening work. Stage and commit only this project's files; preserve parent/sibling changes. No AI-authorship attribution.

## Review Focus

These numerical/interface cases are covered by the named tasks below, rather than deferred to a review cycle:

- PyG unavailable: core imports and MLP execution still work. Task 1.
- Uneven calibration batches, including single-observation batches: pooled population moments weight observations correctly. Task 3.
- Constant channels: the denominator floor produces finite corrections without promising recovered variance. Task 3.
- Fixed buffers: unequal endpoint buffers fail clearly and equal buffers remain unchanged. Tasks 2 and 5.
- Empty or one-shot data sources: reject rather than consume silently or loop forever. Tasks 2, 3, and 5.

## Files and verification environment

| File | Responsibility |
| --- | --- |
| `pyproject.toml`, `.gitignore` | Package metadata, optional GCN dependency, generated-file exclusions. |
| `src/rebasin_bezier_repair/adapters/base.py` | Adapter protocol, repair-site descriptor, shared affine correction helper. |
| `src/rebasin_bezier_repair/adapters/mlp.py` | Sequential biased Linear/ReLU adapter. |
| `src/rebasin_bezier_repair/adapters/gcn.py` | Plain biased uncached GCNConv-stack adapter; optional import. |
| `src/rebasin_bezier_repair/paths.py` | Quadratic state interpolation and control-only fitting. |
| `src/rebasin_bezier_repair/repair.py` | Temporary hooks, pooled moments, sequential sample correction. |
| `src/rebasin_bezier_repair/pipeline.py`, `__init__.py` | Public `connect` and `ConnectivityPath`; compose stages. |
| `examples/mlp.py`, `examples/gcn.py` | Small independent endpoint training and shared-interface demonstration. |
| `README.md`, `docs/adding-an-architecture.md` | Setup, examples, extension contract and checks. |
| `tests/test_adapters.py`, `test_paths.py`, `test_repair.py`, `test_gcn.py`, `test_pipeline.py` | Focused mathematical and interface contracts. |
| `tests/__init__.py`, `tests/support.py` | Small deterministic MLP/data fixtures reused by core tests. |

Use `/home/wge3/ms_thesis/repair/.venv/bin/python`, verified to have Python 3.14, torch 2.9, SciPy 1.16, and PyG 2.7. Source development needs no installation or configuration change. Prefix test/example commands with:

```bash
PYTHONPATH=src:../git_re_basin/src /home/wge3/ms_thesis/repair/.venv/bin/python
```

Below, `$PY` means that interpreter and the same `PYTHONPATH` is set for each command. This is a development launch setting; product code must not modify `sys.path`. README installation instructions use editable sibling/project packages in a user's chosen environment.

## Task 1: MLP adapter and package foundation

**Files:** Create `pyproject.toml`, `.gitignore`, `src/rebasin_bezier_repair/__init__.py`, `src/rebasin_bezier_repair/adapters/__init__.py`, `src/rebasin_bezier_repair/adapters/base.py`, `src/rebasin_bezier_repair/adapters/mlp.py`, `tests/__init__.py`, `tests/support.py`, `tests/test_adapters.py`.

**Interfaces:**

- `State = Mapping[str, Tensor]`; `PermutationSpec = Mapping[str, tuple[str | None, ...]]`.
- Frozen `RepairSite(name: str, module_path: str, channel_axis: int)`.
- `ObservationSelector = Callable[[RepairSite, Tensor, object], Tensor]`. Selection preserves rank/channel axis; shared calibration subsequently moves the channel axis last and flattens other axes.
- `ArchitectureAdapter` protocol methods:
  - `permutation_spec(model: nn.Module, state: State) -> PermutationSpec`
  - `forward(model: nn.Module, state: State, batch: object) -> Tensor`
  - `repair_sites(model: nn.Module) -> tuple[RepairSite, ...]`
  - `apply_correction(state: State, site: RepairSite, scale: Tensor, shift: Tensor) -> dict[str, Tensor]`
- `correct_affine_rows(state: State, weight_name: str, bias_name: str, scale: Tensor, shift: Tensor) -> dict[str, Tensor]` returns a copied state, with `W'=scale*W` on output rows and `b'=scale*b+shift`.
- `MLPAdapter()` implements the protocol. Batches are a tensor or tuple/list whose first item is the input tensor. Execution uses `torch.func.functional_call(..., strict=True)`.

- [x] Write `test_mlp_permutation_preserves_logits`, `test_mlp_rejects_unsupported_structure`, `test_affine_correction_copies_state`, and `test_core_import_without_pyg` before product code. Fixtures use a biased `4→8→6→3` MLP and 24 deterministic examples. The first test uses a reversed permutation for each hidden group and asserts:

  ```python
  self.assertEqual(set(spec), set(state))
  torch.testing.assert_close(original_logits, permuted_logits, rtol=1e-4, atol=1e-5)
  self.assertEqual([site.module_path for site in sites], ["0", "2"])
  ```

  Also assert caller state is byte-equal before/after correction; unsupported biasless Linear and intervening normalization raise clear errors. The import test uses a subprocess import blocker for `torch_geometric` and successfully imports the package and MLP adapter.
- [x] Run `$PY -m unittest tests.test_adapters -v`. Expected: failure because the new package/adapter does not exist.
- [x] Implement the base protocol/helper and MLP adapter. Adapt the sibling MLP axis-description algorithm; include all buffers with fixed axes. Reject structure outside biased Linear/ReLU stacks. The public helper from the sibling validates exact state coverage, so build a complete spec before calling validation.
- [x] Add metadata: torch and `git-re-basin-thesis` required, `gcn` optional extra containing torch-geometric. Root exports only PyG-free symbols at this stage. Add local exclusions for bytecode, egg-info, build/dist, and environments.
- [x] Run `$PY -m unittest tests.test_adapters -v`. Expected: all four contracts pass, including blocked-PyG import.
- [x] Commit exact Task 1 files: `feat: add architecture adapter interface and MLP adapter`.

## Task 2: Differentiable Bézier fitting

**Files:** Create `src/rebasin_bezier_repair/paths.py`, `tests/test_paths.py`.

**Interfaces:** Consumes Task 1's adapter and state types. Produces:

- `bezier_state(a: State, b: State, control: State, t: float, *, parameter_names: Collection[str]) -> dict[str, Tensor]`.
- `fit_curve(model: nn.Module, adapter: ArchitectureAdapter, a: State, b: State, train_data: Iterable[object], loss_fn: Callable[[Tensor, object], Tensor], *, steps: int = 100, lr: float = 0.01, seed: int = 0) -> dict[str, Tensor]`.
- Control contains named floating parameters only; evaluated states contain all parameters and fixed buffers.

- [x] Write `test_endpoints_and_control_gradient`, `test_fitting_changes_only_control`, `test_fixed_buffers`, and `test_invalid_curve_inputs`. Pin the formula and gradient with:

  ```python
  torch.testing.assert_close(bezier_state(a, b, c, 0, parameter_names=names)[key], a[key], rtol=0, atol=0)
  torch.testing.assert_close(bezier_state(a, b, c, 1, parameter_names=names)[key], b[key], rtol=0, atol=0)
  midpoint[key].sum().backward()
  torch.testing.assert_close(c[key].grad, torch.full_like(c[key], 0.5))
  ```

  Fit three steps of training CE; at least one control differs from the arithmetic midpoint, and caller state/gradients/flags remain unchanged. Register an equal constant buffer and assert it is copied exactly; unequal buffers fail. Reject nonfinite/out-of-range t, empty or one-shot training sources, nonpositive steps/lr, and nonfinite loss.
- [x] Run `$PY -m unittest tests.test_paths -v`. Expected: missing path module/functions.
- [x] Implement the quadratic formula from the GCN source, preserving autograd through control tensors. Validate fixed-state equality and separate parameters via `named_parameters()`.
- [x] Implement fitting on an owned copy: detached endpoints, midpoint control, Adam defaults, finite data cycling, training mode, one uniform t per optimizer step, final detached control. Validate sources/settings before optimization; do not retain batches after return.
- [x] Run `$PY -m unittest tests.test_adapters tests.test_paths -v`. Expected: adapter and path contracts pass.
- [x] Commit Task 2 files: `feat: fit quadratic Bezier controls through architecture adapters`.

## Task 3: Generic sequential REPAIR

**Files:** Create `src/rebasin_bezier_repair/repair.py`, `tests/test_repair.py`.

**Interfaces:** Consumes Task 1's adapter/sites/selector/helper. Produces:

- Frozen `ChannelMoments(mean: Tensor, std: Tensor)`; `Moments = Mapping[str, ChannelMoments]`, keyed by `RepairSite.name`.
- `collect_moments(model: nn.Module, adapter: ArchitectureAdapter, state: State, calibration_data: Iterable[object], *, selector: ObservationSelector | None = None, sites: tuple[RepairSite, ...] | None = None) -> dict[str, ChannelMoments]`.
- `repair_state(model: nn.Module, adapter: ArchitectureAdapter, state: State, t: float, endpoint_moments: tuple[Moments, Moments], calibration_data: Iterable[object] | None, *, selector: ObservationSelector | None = None, eps: float = 1e-8) -> dict[str, Tensor]`.

- [x] Write `test_pooled_population_moments`, `test_sequential_target_moments`, `test_constant_channel_is_finite`, and `test_calibration_sources_and_cleanup`. Pin uneven-batch pooling with selected values `[[1,2],[3,6],[8,9]]` divided into batches of size 2 and 1:

  ```python
  torch.testing.assert_close(measured.mean, values.mean(0))
  torch.testing.assert_close(measured.std, values.std(0, unbiased=False))
  ```

  For a deterministic two-hidden-layer MLP with nonzero channel variances, repair at `t=0.5`, collect both sites again, and compare each mean/std to endpoint targets with `rtol=1e-4, atol=1e-5`. Constant-channel corrected state must remain finite. Empty/one-shot data and fewer than two selected observations fail. Temporary hooks and model flags are restored even when the selector raises.
- [x] Run `$PY -m unittest tests.test_repair -v`. Expected: missing repair module/functions.
- [x] Implement temporary output hooks and float64 pooled sums/squared sums/counts. Return population moments in the activation dtype/device. Selection runs before axis movement/flattening; labels are not passed to statistical calculations.
- [x] Implement sequential correction from the existing GCN REPAIR formula. Clone/detach the sampled state; bypass exact endpoints; recollect each site after preceding corrections; call adapter correction with blended targets and the denominator floor. Restore hooks/modes in `finally`; use evaluation mode and no gradients.
- [x] Run `$PY -m unittest tests.test_adapters tests.test_paths tests.test_repair -v`. Expected: all focused core contracts pass.
- [x] Commit Task 3 files: `feat: calibrate arbitrary path samples with sequential REPAIR`.

## Task 4: GCN adapter proves the same interface

**Files:** Create `src/rebasin_bezier_repair/adapters/gcn.py`, `tests/test_gcn.py`.

**Interfaces:** Consumes Tasks 1–3 unchanged. Produces `GCNAdapter()` for a model exposing a plain `convs: nn.ModuleList` of biased uncached GCNConv layers. Batch supplies `x` and `edge_index`; calibration selector and loss callback own masks.

- [x] Write `test_gcn_permutation_and_shared_curve`, `test_gcn_training_node_calibration`, and `test_gcn_repair_and_replay`. Use a deterministic synthetic graph and a three-convolution test model to exercise two repair sites. Assertions:

  ```python
  self.assertEqual(set(spec), set(state))
  torch.testing.assert_close(original_logits, permuted_logits, rtol=1e-4, atol=1e-5)
  self.assertEqual([site.module_path for site in sites], ["convs.0", "convs.1"])
  ```

  Fit through the same `fit_curve`. Compare selected moments to direct preactivation moments on `train_mask`. REPAIR reaches nondegenerate targets and a reloaded ordinary model reproduces logits. Reject cached/biasless convolutions and fixed unsupported buffers. GCN tests skip only if PyG is unavailable; it is available in the selected environment.
- [x] Run `$PY -m unittest tests.test_gcn -v`. Expected: missing GCN adapter, with no optional-dependency skip here.
- [x] Implement the spec: `convs.i.lin.weight` uses `(hidden_i, hidden_(i-1))` with fixed input/output axes at the ends; outer biases use the output group. Hook complete GCNConv outputs; fold into internal weight rows and outer bias. Forward uses strict functional execution on `(batch.x, batch.edge_index)`.
- [x] Run `$PY -m unittest discover -s tests -v`. Expected: both adapters and generic path/calibration contracts pass.
- [x] Commit Task 4 files: `feat: add GCN architecture adapter`.

## Task 5: Public pipeline, examples, and extension guide

**Files:** Create `src/rebasin_bezier_repair/pipeline.py`, `examples/mlp.py`, `examples/gcn.py`, `README.md`, `docs/adding-an-architecture.md`, `tests/test_pipeline.py`; modify package `__init__.py`.

**Interfaces:** Consumes all prior interfaces. Produces:

```python
connect(model_a: nn.Module, model_b: nn.Module, *,
        adapter: ArchitectureAdapter,
        train_data: Iterable[object],
        loss_fn: Callable[[Tensor, object], Tensor],
        calibration_data: Iterable[object],
        select_observations: ObservationSelector | None = None,
        curve_steps: int = 100, curve_lr: float = 0.01,
        seed: int = 0, matching_max_iter: int = 100,
        repair_eps: float = 1e-8,
        invariance_rtol: float = 1e-4,
        invariance_atol: float = 1e-5) -> ConnectivityPath

ConnectivityPath.model_at(self, t: float, repair: bool = False,
                          calibration_data: Iterable[object] | None = None) -> nn.Module
```

`ConnectivityPath` exposes detached `endpoint_a`, original `endpoint_b`, `aligned_b`, `control`, `permutation`, and `endpoint_moments`. It privately owns template/adapter/selector and repair epsilon; it retains no training/calibration data source.

- [x] Write `test_connect_preserves_endpoints_and_replays_models`, `test_connect_rejects_incompatible_inputs`, and `test_gcn_held_out_labels_do_not_affect_path`. The first connects the MLP pair, checks original models' full states/gradients/flags, checks both exact endpoints, repairs the midpoint with explicit calibration, and reloads its state dictionary into the original architecture for equal predictions. Negative cases include incompatible endpoint shape/dtype, invalid t, missing interior calibration, and unequal fixed buffers. In the GCN test, changing only held-out labels with the same seed leaves control and repaired state equal when loss and selector use training rows.
- [x] Run `$PY -m unittest tests.test_pipeline -v`. Expected: missing pipeline/public operation.
- [x] Implement composition: validate states, adapter support/specs and finite re-iterable sources; reject native BatchNorm; import `weight_matching`/`apply_permutation`; verify B's logits on calibration batches in eval mode; reject unequal fixed buffers; call fitting and cached endpoint moment collection. `model_at` validates t, optionally repairs the sample, then loads a copied template and returns eval mode. Root exports `connect`, `ConnectivityPath`, `MLPAdapter`, and base interface types; import GCN explicitly from `adapters.gcn`.
- [x] Run `$PY -m unittest discover -s tests -v`. Expected: all focused contracts pass, including GCN tests in this environment.
- [x] Write both examples with `main()` guards. MLP uses 96 synthetic 4D samples, a biased `4→8→6→3` model, and independent endpoint seeds 0/1. GCN uses 60 nodes, 4 features, 3 classes, deterministic edges, masks of 36/12/12 nodes, and a biased uncached two-layer width-8 model adapted from the existing GCN class. Use CPU, one torch thread, 50 small endpoint-training epochs, and the shared curve defaults. Print `endpoint_a`, `endpoint_b`, `bezier_midpoint`, and `repaired_midpoint` loss/accuracy; require finite results without an accuracy-improvement threshold.
- [x] Write README installation/run commands and the adapter guide. Installation uses `python -m pip install -e ../git_re_basin -e .` or `-e '.[gcn]'`; identify the sibling dependency. Guide describes the four responsibilities, tensor-axis coupling, observation selection, correction-site mapping, explicit rejection of unsupported structures, and focused invariance/gradient/moment/replay checks. Include local source provenance and deferred work, using working adapters as the template.
- [x] Run `$PY examples/mlp.py` and `$PY examples/gcn.py`. Expected: exit 0, four named finite metric rows each, ordinary repaired models materialized through the same interface. Run `$PY -m unittest discover -s tests -v` after integration. Expected: all tests pass. Inspect dependency metadata and README commands against actual files; run `git diff --check`.
- [x] Commit exact Task 5 files: `feat: expose connectivity pipeline with MLP and GCN examples`.

## Execution handoff

Native execution is recommended because the five tasks share interfaces and are small. Preserve the user's explicit request to skip review cycles regardless of execution method. Independent bounded implementation may be delegated if requested or useful, with file ownership established first.

Execute in the intended `rebasin_bezier_repair` project directory. Its siblings contain local untracked source/artifacts, so assess worktree isolation before execution rather than assuming a new checkout contains those dependencies. Keep any execution ledger project-local and record tests/deviations as tasks complete.

User approved this plan. All five tasks are implemented and verified in the intended project directory. The full suite passes 18 tests; both CPU examples exit successfully with four finite metric rows each.
