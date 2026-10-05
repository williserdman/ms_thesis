#!/bin/bash
# Submit from gcn_bezier_conn: sbatch --array=0-15%4 scripts/repair_matrix.sbatch SOURCE_ROOT OUTPUT_ROOT
#SBATCH --job-name=gnn-repair
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
source_root=${1:?Supply the completed baseline matrix root}
output_root=${2:?Supply a new repair output root}
datasets=(Cora Roman-empire squirrel chameleon)
architectures=(gcn mlp graphsage gat)
task=${SLURM_ARRAY_TASK_ID:?Submit with --array=0-15%4}
dataset=${datasets[$((task / 4))]}
architecture=${architectures[$((task % 4))]}
"${GNN_PYTHON:-/home/wge3/miniconda3/envs/py312/bin/python}" -u -m gcn_mc repair \
    --source "$source_root/$dataset/$architecture/report.json" \
    --thesis-root "${GNN_THESIS_ROOT:-/home/wge3/ms_thesis}" \
    --device cuda --threads 2 --output "$output_root/$dataset/$architecture"
