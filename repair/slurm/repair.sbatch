#!/bin/bash
#SBATCH --job-name=repair-cifar10
#SBATCH --partition=gpu
#SBATCH --constraint="gpu2h100|gpul40s|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

if [[ -f "$PWD/pyproject.toml" && -d "$PWD/src/repair" ]]; then
    PROJECT_ROOT="$PWD"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/pyproject.toml" && -d "$SLURM_SUBMIT_DIR/src/repair" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    echo "Run sbatch with --chdir=/home/wge3/ms_thesis/repair" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
RUN_DIR="$PROJECT_ROOT/runs/cifar10-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
RUN_KIND="${REPAIR_RUN_KIND:-pilot}"

case "$RUN_KIND" in
    pilot)
        EPOCHS=2
        ALPHAS=(0 0.5 1)
        ;;
    full)
        EPOCHS=100
        ALPHAS=(0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1)
        ;;
    *)
        echo "REPAIR_RUN_KIND must be pilot or full, got: $RUN_KIND" >&2
        exit 2
        ;;
esac

[[ -x "$PYTHON" ]] || { echo "Missing Python executable: $PYTHON" >&2; exit 1; }
mkdir -p "$PROJECT_ROOT/runs"
mkdir "$RUN_DIR"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

echo "run_kind=$RUN_KIND"
echo "run_dir=$RUN_DIR"
echo "hostname=$(hostname)"
nvidia-smi
"$PYTHON" - <<'PY'
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

"$PYTHON" -m repair cifar10 \
    --epochs "$EPOCHS" \
    --width 1 \
    --batch-size 500 \
    --calibration-batches 10 \
    --device cuda \
    --threads 4 \
    --num-workers 4 \
    --alphas "${ALPHAS[@]}" \
    --output "$RUN_DIR"

"$PYTHON" -m repair.plotting "$RUN_DIR/report.json"
