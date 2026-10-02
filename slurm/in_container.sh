#!/usr/bin/env bash
# Run a Python command inside the unchanged quantum_ml:v4 image with the clone, the run
# directory, and the extras directory mounted.
#
#   slurm/in_container.sh scripts/check_env.py --mode container
#   slurm/in_container.sh scripts/launcher.py --config configs/experiment.yaml --group qnn
#
# Directories, from environment variables with these defaults:
#   QNNWIND_REPO     the git clone          $SCRATCH/qnn-wind-prediction  (same path inside)
#   QNNWIND_SCRATCH  results, logs, tasks   $SCRATCH/qnn-wind-runs        (same path inside)
#   QNNWIND_EXTRAS   the 12 extra packages  $SCRATCH/qnn-wind-pyextras    (/opt/qnnwind-extras, read-only)
# podman-hpc does not propagate most environment variables and does not mount these
# directories on its own (NERSC podman-hpc docs), so everything is passed explicitly.
# PYTHONPATH is set only inside the container (NERSC advises against setting it for the
# podman-hpc wrapper itself). Git provenance comes from the host: job scripts export it via
# slurm/git_provenance.sh; for interactive use it is read here.
set -euo pipefail

QNNWIND_REPO="${QNNWIND_REPO:-${SCRATCH:?SCRATCH is not set}/qnn-wind-prediction}"
QNNWIND_SCRATCH="${QNNWIND_SCRATCH:-${SCRATCH:?SCRATCH is not set}/qnn-wind-runs}"
QNNWIND_EXTRAS="${QNNWIND_EXTRAS:-${SCRATCH:?SCRATCH is not set}/qnn-wind-pyextras}"
QNNWIND_IMAGE="${QNNWIND_IMAGE:-quantum_ml:v4}"
QNNWIND_PODMAN="${QNNWIND_PODMAN:-podman-hpc}"
EXTRAS_MOUNT=/opt/qnnwind-extras

if [ ! -d "$QNNWIND_EXTRAS/lightgbm" ]; then
    echo "extras directory $QNNWIND_EXTRAS is missing or incomplete; install it first" \
         "(README, 'Cluster use')" >&2
    exit 1
fi
mkdir -p "$QNNWIND_SCRATCH"

if [ -z "${QNNWIND_GIT_COMMIT:-}" ]; then
    QNNWIND_GIT_COMMIT="$(git -C "$QNNWIND_REPO" rev-parse HEAD 2>/dev/null || true)"
    status="$(git -C "$QNNWIND_REPO" status --porcelain 2>/dev/null || true)"
    QNNWIND_GIT_STATUS="$(printf '%s' "$status" | awk 'NR > 1 { printf "; " } { printf "%s", $0 }')"
    if [ -n "$status" ]; then QNNWIND_GIT_DIRTY=1; else QNNWIND_GIT_DIRTY=0; fi
fi

exec "$QNNWIND_PODMAN" run --rm \
    -v "$QNNWIND_REPO:$QNNWIND_REPO" \
    -v "$QNNWIND_SCRATCH:$QNNWIND_SCRATCH" \
    -v "$QNNWIND_EXTRAS:$EXTRAS_MOUNT:ro" \
    -w "$QNNWIND_REPO" \
    -e QNNWIND_SCRATCH="$QNNWIND_SCRATCH" \
    -e QNNWIND_EXTRAS_MOUNT="$EXTRAS_MOUNT" \
    -e PYTHONPATH="$EXTRAS_MOUNT:$QNNWIND_REPO/src" \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e QNNWIND_GIT_COMMIT="$QNNWIND_GIT_COMMIT" \
    -e QNNWIND_GIT_DIRTY="${QNNWIND_GIT_DIRTY:-}" \
    -e QNNWIND_GIT_STATUS="${QNNWIND_GIT_STATUS:-}" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
    -e SLURM_NODEID="${SLURM_NODEID:-0}" \
    -e SLURM_NNODES="${SLURM_NNODES:-1}" \
    "$QNNWIND_IMAGE" python "$@"
