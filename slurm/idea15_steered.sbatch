#!/bin/bash
#SBATCH --job-name=smc_idea15_steered
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
source "$(dirname "$0")/env.sh"          # this cd's into src/ and activates conda
python -m experiments.idea15_steered --gpus 1 --out "../results/idea15_steered.json" "$@"
