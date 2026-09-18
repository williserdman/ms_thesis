#!/bin/bash
# Submit from gcn_bezier_conn: sbatch --array=0-15%4 scripts/reference_baselines.sbatch OUTPUT_ROOT smoke
# Array 0-3: Cora; 4-7: Roman-empire; 8-11: squirrel; 12-15: chameleon.
# Each group orders gcn, mlp, graphsage, gat. Omit 'smoke' for preset budgets.
# Optional arguments after the mode are forwarded to `gcn_mc run`.
#SBATCH --job-name=gnn-reference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
output_root=${1:?Supply a new output root}
mode=${2:-full}
shift "$(( $# > 1 ? 2 : 1 ))"
datasets=(Cora Roman-empire squirrel chameleon)
architectures=(gcn mlp graphsage gat)
task=${SLURM_ARRAY_TASK_ID:?Submit with --array=0-15%4}
dataset=${datasets[$((task / 4))]}
architecture=${architectures[$((task % 4))]}
extra=()
if [[ "$mode" == smoke ]]; then
    extra+=(--smoke)
elif [[ "$mode" != full ]]; then
    echo "Mode must be smoke or full" >&2
    exit 2
fi
"${GNN_PYTHON:-/home/wge3/miniconda3/envs/py312/bin/python}" -u -m gcn_mc run \
    --preset reference --architecture "$architecture" --datasets "$dataset" \
    --thesis-root "${GNN_THESIS_ROOT:-/home/wge3/ms_thesis}" \
    --device cuda --threads 2 --output "$output_root/$dataset/$architecture" \
    "${extra[@]}" "$@"
