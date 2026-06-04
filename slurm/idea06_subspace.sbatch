#!/bin/bash
#SBATCH --job-name=smc_idea06_subspace
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
source "$(dirname "$0")/env.sh"          # this cd's into src/ and activates conda
python -m experiments.idea06_subspace --gpus 1 --out "../results/idea06_subspace.json" "$@"
