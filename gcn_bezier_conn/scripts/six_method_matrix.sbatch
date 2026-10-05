#!/bin/bash
# From gcn_bezier_conn: sbatch --array=0-59%4 scripts/six_method_matrix.sbatch OUTPUT_ROOT [TUNED_SOURCE_ROOT]
#SBATCH --job-name=gnn-six-methods
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpul40s|gpu2h100|dgx
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
output_root=${1:?Supply a new matrix output root}
source_root=${2:-runs/tuned-endpoints-20260918}
datasets=(Cora Roman-empire squirrel chameleon Questions computers photo Citeseer Pubmed actor texas cornell Amazon-ratings Minesweeper Tolokers)
architectures=(gcn mlp graphsage gat)
task=${SLURM_ARRAY_TASK_ID:?Submit with --array=0-59%4}
if (( task < 0 || task >= 60 )); then
    echo 'Array index must be between 0 and 59' >&2
    exit 2
fi
dataset=${datasets[$((task / 4))]}
architecture=${architectures[$((task % 4))]}
python=${GNN_PYTHON:-/home/wge3/miniconda3/envs/py312/bin/python}
thesis_root=${GNN_THESIS_ROOT:-/home/wge3/ms_thesis}
if (( task < 16 )); then
    baseline="$source_root/$dataset/$architecture/report.json"
    test -s "$baseline"
else
    baseline="$output_root/baselines/$dataset/$architecture/report.json"
    if [[ ! -s "$baseline" ]]; then
        "$python" -u -B -m gcn_mc run \
            --preset reference --architecture "$architecture" --datasets "$dataset" \
            --thesis-root "$thesis_root" --device cuda --threads 2 \
            --pairs 0:1 2:3 4:5 --curve-epochs 200 --points 21 \
            --output "$output_root/baselines/$dataset/$architecture"
    fi
fi
analysis="$output_root/analysis/$dataset/$architecture"
if [[ ! -s "$analysis/report.json" ]]; then
    "$python" -u -B -m gcn_mc repair --include-bezier \
        --source "$baseline" --thesis-root "$thesis_root" \
        --device cuda --threads 2 --output "$analysis"
else
    "$python" -u -B -m gcn_mc plot "$analysis/report.json"
fi
