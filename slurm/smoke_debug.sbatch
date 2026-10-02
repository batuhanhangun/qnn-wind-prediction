#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -q <QOS>
#SBATCH -C cpu
#SBATCH -t <WALLTIME>
#SBATCH -N <NODES>
#SBATCH -J qnnwind-smoke
#SBATCH -o <QNNWIND_SCRATCH>/logs/%x-%j.out
# Debug-QOS smoke job: the test suite inside the container, then every
# model on the smoke grid (configs/smoke.yaml) through the task farm with 32 workers.
# Use QOS = debug, WALLTIME = 00:30:00, NODES = 1. Render with slurm/render.sh, then sbatch.
set -euo pipefail
export QNNWIND_REPO="<QNNWIND_REPO>"
export QNNWIND_SCRATCH="<QNNWIND_SCRATCH>"
export QNNWIND_EXTRAS="<QNNWIND_EXTRAS>"
CONFIG="${QNNWIND_CONFIG:-configs/smoke.yaml}"
echo "job $SLURM_JOB_ID: $SLURM_JOB_NODELIST, config $CONFIG"
source "$QNNWIND_REPO/slurm/git_provenance.sh" warn

"$QNNWIND_REPO/slurm/in_container.sh" -m pytest -q -p no:cacheprovider
"$QNNWIND_REPO/slurm/in_container.sh" scripts/make_tasks.py --config "$CONFIG"
# Tuning first (the tuned classical runs need it), then the QNN and classical runs.
"$QNNWIND_REPO/slurm/in_container.sh" scripts/launcher.py --config "$CONFIG" --workers 32 --cluster \
    --group tuning --group qnn --group classical
"$QNNWIND_REPO/slurm/in_container.sh" scripts/status.py --config "$CONFIG"
