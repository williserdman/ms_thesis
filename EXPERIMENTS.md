# Spectral Filtering × Mode Connectivity — PoC Experiments

Five proof-of-concept experiments probing the bridge between **learnable spectral
graph filters** and **mode connectivity** (linear / Bézier loss-barrier paths).
This branch (`exp/spectral-mc-pocs`) builds on `clean-main`.

## Why a new substrate model

The portfolio review found the repo's headline model `DiffusedAttention` exposes
**no clean linear spectral-coefficient axis** (its filter coefficients are buried
in a nonlinear 4-block PolyAttn stack). ~12/20 ideas assume exactly such an axis.
So these PoCs run on an explicit **linear-γ GPR-GNN head**:

`src/models/linear_spectral.py` → `LinearSpectralGNN`

    Z = Σ_{k=0..K} γ_k · φ_k(op) · MLP(X)        # linear in γ

- `γ` is an explicit `(K+1,)` coefficient axis → interpolating γ is a genuine
  morph of the filter response `g(λ) = Σ_k γ_k φ_k(λ)`.
- `basis ∈ {cheb, mono}`, `domain ∈ {adj, lap}`, `gamma_init ∈ {ppr, ones, random}`.
- Same repo contract: `forward(batch) -> (logits, inner_loss)`, a LightningModule,
  built from `DatasetInfo`, fed by the existing `LightningGraphLoader`.

Shared math (framework-agnostic, numpy/torch both):
- `src/mode_connectivity/paths.py` — linear/Bézier interpolation, loss barrier.
- `src/mode_connectivity/spectral.py` — basis matrix, filter response, condition number κ.
- `src/experiments/common.py` — dataset prep, training, param vectorization
  (whole-net vs `γ`-only subset), masked eval, `barrier_along_path`.

## The experiments

| # | Module | Question | Headline metric |
|---|--------|----------|-----------------|
| **B1** | `b1_barrier` | Does a barrier even *exist* in the low-dim γ axis? (gates the rest) | `coeff_barrier_nontrivial` (mean γ-linear barrier > 0.02) |
| 06 | `idea06_subspace` | Is the barrier removable from the *filter axis* or the *mixer*? | `ρ_coef` vs `ρ_bulk` (barrier fraction removed by bending γ vs MLP) |
| 09 | `idea09_frontier` | Does filter order K trade accuracy for barrier height? Does κ predict it? | acc(K), weight-barrier(K), response-barrier(K), corr(barrier, log κ_K), elbow K* |
| 15 | `idea15_steered` | Can a tiny net steer the Bézier control from the graph spectrum, and transfer? | `steered_below_midpoint`; held-out `transfer_below_midpoint` |
| 18 | `idea18_filtercurve` | Does a per-node Bézier *curve* of filters beat one shared filter? | `curve_beats_baseline`; corr(per-node t_i, degree) |

**B1 runs first by design** — if the ~(K+1)-dim coefficient chart is already
near-linearly connected (`coeff_barrier_nontrivial == false`), ideas 06/15/18
largely degenerate to nulls and 09's weight-vs-filter gap is the story. Read B1's
output before trusting the rest.

## Running

### Local (this repo, CPU/torch-less box)
Only the framework-agnostic math runs locally; torch is not installed here.
```bash
python3 tests/test_math.py                 # 12 unit tests on path + spectral math
cd src && python3 -m py_compile experiments/*.py   # structural check
```

### SLURM (1 GPU per job, ≤1h wall, sequential chain)
Edit `slurm/env.sh` for your cluster (set `CONDA_ENV`, module loads, partition).
```bash
bash slurm/submit_all.sh --smoke     # FAST smoke first: tiny config, real torch run
bash slurm/submit_all.sh             # full runs (cora, citeseer, squirrel, Roman-empire)
```
`submit_all.sh` submits each experiment as its own job chained with
`--dependency=afterok`, so they run **sequentially** and every job stays under the
1-hour limit. Per-experiment scripts also exist: `sbatch slurm/<name>.sbatch [--smoke]`.

Results → `results/<experiment>.json`; logs → `slurm/logs/`.

### The `--smoke` config
1 dataset (cora), K=2, ~5 epochs, 2 seeds, CPU-ok. Verifies every experiment runs
end-to-end in minutes — make this your first SLURM job before committing GPU-hours.

## Layout
```
src/models/linear_spectral.py        # GPR-GNN linear-γ substrate
src/mode_connectivity/{paths,spectral}.py
src/experiments/common.py            # shared harness
src/experiments/{b1_barrier,idea06_subspace,idea09_frontier,idea15_steered,idea18_filtercurve}.py
slurm/{env.sh,<name>.sbatch,submit_all.sh}
tests/test_math.py
results/                             # json outputs (gitignored)
```

## Status / caveats
- All 5 modules + the substrate `py_compile` clean; the 12 math unit tests pass.
- **Not yet run on GPU** — no torch on the build box. First SLURM `--smoke` job is
  the real end-to-end smoke test.
- idea-06/15 use forward-only barrier evaluation + a cheap control-point search/fit
  (no GNN retraining) — documented PoC fits, not full optimization.
- Each experiment caps its sweep/pairs and logs what was capped to respect the 1h limit.
