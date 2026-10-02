#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -q <QOS>
#SBATCH -C cpu
#SBATCH -t <WALLTIME>
#SBATCH -N <NODES>
#SBATCH -J qnnwind-multinode
#SBATCH -o <QNNWIND_SCRATCH>/logs/%x-%j.out
# Multi-node check: the smoke grid (configs/smoke.yaml) through the cluster launch path -- srun with one container per
# node, dynamic claiming, W = 64, one srun per group as the barrier -- into the separate run
# directory <QNNWIND_SCRATCH>/multinode-check. The QNN group runs with the load keeper.
# Use QOS = debug, WALLTIME = 00:30:00, NODES = 2. Render with slurm/render.sh, then sbatch.
set -euo pipefail
export QNNWIND_REPO="<QNNWIND_REPO>"
export QNNWIND_SCRATCH="<QNNWIND_SCRATCH>/multinode-check"
export QNNWIND_EXTRAS="<QNNWIND_EXTRAS>"
CONFIG="${QNNWIND_CONFIG:-configs/smoke.yaml}"
echo "job $SLURM_JOB_ID: $SLURM_NNODES node(s) $SLURM_JOB_NODELIST, config $CONFIG"
echo "run directory: $QNNWIND_SCRATCH"
source "$QNNWIND_REPO/slurm/git_provenance.sh" warn
srun -N "$SLURM_NNODES" --ntasks-per-node=1 bash -c 'echo "node $SLURM_NODEID: $(hostname)"'

"$QNNWIND_REPO/slurm/in_container.sh" scripts/make_tasks.py --config "$CONFIG"
for group in tuning qnn classical; do
    srun -N "$SLURM_NNODES" --ntasks-per-node=1 -c 256 --cpu-bind=none \
        "$QNNWIND_REPO/slurm/in_container.sh" scripts/launcher.py --config "$CONFIG" \
        --group "$group" --workers 64 --cluster
done
"$QNNWIND_REPO/slurm/in_container.sh" scripts/status.py --config "$CONFIG"
