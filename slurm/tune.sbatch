#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -q <QOS>
#SBATCH -C cpu
#SBATCH -t <WALLTIME>
#SBATCH -N <NODES>
#SBATCH -J qnnwind-tune
#SBATCH -o <QNNWIND_SCRATCH>/logs/%x-%j.out
# Tuning studies for the classical and deep models (task group "tuning").
# Render with slurm/render.sh, then sbatch.
# Optional: export NO_NEW_AFTER_MINUTES=<minutes> to stop starting tasks near the walltime.
set -euo pipefail
export QNNWIND_REPO="<QNNWIND_REPO>"
export QNNWIND_SCRATCH="<QNNWIND_SCRATCH>"
export QNNWIND_EXTRAS="<QNNWIND_EXTRAS>"
CONFIG="${QNNWIND_CONFIG:-configs/experiment.yaml}"
NO_NEW="${NO_NEW_AFTER_MINUTES:+--no-new-after-minutes $NO_NEW_AFTER_MINUTES}"
echo "job $SLURM_JOB_ID: $SLURM_NNODES node(s) $SLURM_JOB_NODELIST, config $CONFIG"
source "$QNNWIND_REPO/slurm/git_provenance.sh" strict

# One container per node; the launchers share the task list through atomic task locks.
srun -N "$SLURM_NNODES" --ntasks-per-node=1 -c 256 --cpu-bind=none \
    "$QNNWIND_REPO/slurm/in_container.sh" scripts/launcher.py --config "$CONFIG" --group tuning --cluster $NO_NEW
