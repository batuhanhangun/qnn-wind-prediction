#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -q <QOS>
#SBATCH -C cpu
#SBATCH -t <WALLTIME>
#SBATCH -N <NODES>
#SBATCH -J qnnwind-qnn
#SBATCH -o <QNNWIND_SCRATCH>/logs/%x-%j.out
# Final QNN runs (task group "qnn"); no tuning dependency. All QNN runs use the same
# W = runtime.workers (set from the calibration). Render with slurm/render.sh, then sbatch.
# Optional: export NO_NEW_AFTER_MINUTES=<minutes> to stop starting tasks near the walltime.
set -euo pipefail
export QNNWIND_REPO="<QNNWIND_REPO>"
export QNNWIND_SCRATCH="<QNNWIND_SCRATCH>"
export QNNWIND_EXTRAS="<QNNWIND_EXTRAS>"
CONFIG="${QNNWIND_CONFIG:-configs/experiment.yaml}"
NO_NEW="${NO_NEW_AFTER_MINUTES:+--no-new-after-minutes $NO_NEW_AFTER_MINUTES}"
echo "job $SLURM_JOB_ID: $SLURM_NNODES node(s) $SLURM_JOB_NODELIST, config $CONFIG"
source "$QNNWIND_REPO/slurm/git_provenance.sh" strict

# One container per node. Every node's launcher walks the same longest-first task list and
# claims tasks with atomic locks, so the nodes share the work dynamically.
srun -N "$SLURM_NNODES" --ntasks-per-node=1 -c 256 --cpu-bind=none \
    "$QNNWIND_REPO/slurm/in_container.sh" scripts/launcher.py --config "$CONFIG" --group qnn --cluster $NO_NEW
