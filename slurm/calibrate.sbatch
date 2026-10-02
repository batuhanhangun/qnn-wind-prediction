#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -q <QOS>
#SBATCH -C cpu
#SBATCH -t <WALLTIME>
#SBATCH -N <NODES>
#SBATCH -J qnnwind-calibrate
#SBATCH -o <QNNWIND_SCRATCH>/logs/%x-%j.out
# Timing calibration: a 5-iteration QNN-1 probe (fold 5, N = 750) as 1,
# 32, 64, and 128 concurrent copies on one node, each copy pinned to its own physical core;
# then the time per circuit per W and the chosen W. Use QOS = regular, NODES = 1,
# WALLTIME = 01:00:00. Render with slurm/render.sh, then sbatch.
set -euo pipefail
export QNNWIND_REPO="<QNNWIND_REPO>"
export QNNWIND_SCRATCH="<QNNWIND_SCRATCH>"
export QNNWIND_EXTRAS="<QNNWIND_EXTRAS>"
CONFIG="${QNNWIND_CONFIG:-configs/experiment.yaml}"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, config $CONFIG"
source "$QNNWIND_REPO/slurm/git_provenance.sh" warn

"$QNNWIND_REPO/slurm/in_container.sh" scripts/make_tasks.py --config "$CONFIG" --calibration
for group in calibration_W001 calibration_W032 calibration_W064 calibration_W128; do
    "$QNNWIND_REPO/slurm/in_container.sh" scripts/launcher.py --config "$CONFIG" --group "$group" --cluster
done
"$QNNWIND_REPO/slurm/in_container.sh" scripts/status.py --config "$CONFIG" --calibration
