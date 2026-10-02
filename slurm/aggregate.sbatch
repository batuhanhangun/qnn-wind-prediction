#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -q <QOS>
#SBATCH -C cpu
#SBATCH -t <WALLTIME>
#SBATCH -N <NODES>
#SBATCH -J qnnwind-aggregate
#SBATCH -o <QNNWIND_SCRATCH>/logs/%x-%j.out
# Aggregation, tables, figures, and statistics (scripts/analyze.py). Use NODES = 1.
# Render with slurm/render.sh, then sbatch.
set -euo pipefail
export QNNWIND_REPO="<QNNWIND_REPO>"
export QNNWIND_SCRATCH="<QNNWIND_SCRATCH>"
export QNNWIND_EXTRAS="<QNNWIND_EXTRAS>"
CONFIG="${QNNWIND_CONFIG:-configs/experiment.yaml}"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, config $CONFIG"
source "$QNNWIND_REPO/slurm/git_provenance.sh" warn

# One entry point: aggregation (<runs>/results/aggregated), tables, figures, and
# outputs/results_summary.md in the clone.
"$QNNWIND_REPO/slurm/in_container.sh" scripts/analyze.py --config "$CONFIG" --runs "$QNNWIND_SCRATCH"
