#!/bin/bash
# Shared SLURM environment setup for the spectral x mode-connectivity PoCs.
# GENERIC conda+gpu assumptions -- edit CONDA_ENV / module loads for your cluster.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-ms_thesis}"

# --- activate environment (edit to match your cluster) ---
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# run from src/ so repo package imports (loading.*, models.*) resolve
cd "$(dirname "${BASH_SOURCE[0]}")/../src"
export PYTHONUNBUFFERED=1
echo "[env] python=$(which python) cuda_visible=${CUDA_VISIBLE_DEVICES:-none} cwd=$(pwd)"
