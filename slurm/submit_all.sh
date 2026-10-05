#!/bin/bash
# Submit all spectral x mode-connectivity PoC experiments to SLURM, SEQUENTIALLY.
#
# Each experiment is its own job (<=1hr wall, 1 GPU). They are chained with
# --dependency=afterok so job N+1 only starts after job N succeeds -- this keeps
# every job under a 1-hour limit while running the suite in order. B1 (the
# barrier-existence gate) runs first by design.
#
# Usage:
#   bash slurm/submit_all.sh            # full runs
#   bash slurm/submit_all.sh --smoke    # tiny smoke config (fast first-run check)
#
# Override env (optional):
#   CONDA_ENV=myenv PARTITION=gpu-a100 bash slurm/submit_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root
mkdir -p slurm/logs results

EXTRA_ARGS="$*"                  # forwarded to every experiment (e.g. --smoke)

# Order matters: B1 is the gating pilot; the rest follow.
EXPERIMENTS=(
  b1_barrier
  idea06_subspace
  idea09_frontier
  idea15_steered
  idea18_filtercurve
)

dep=""
for exp in "${EXPERIMENTS[@]}"; do
  script="slurm/${exp}.sbatch"
  if [[ ! -f "$script" ]]; then
    echo "ERROR: missing $script" >&2
    exit 1
  fi
  if [[ -z "$dep" ]]; then
    jid=$(sbatch --parsable "$script" $EXTRA_ARGS)
  else
    jid=$(sbatch --parsable --dependency=afterok:"$dep" "$script" $EXTRA_ARGS)
  fi
  echo "submitted $exp -> job $jid${dep:+ (after $dep)}"
  dep="$jid"
done

echo
echo "All ${#EXPERIMENTS[@]} jobs submitted as a sequential chain. Watch: squeue --me"
echo "Results land in results/<experiment>.json ; logs in slurm/logs/."
