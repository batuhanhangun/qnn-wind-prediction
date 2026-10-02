#!/usr/bin/env bash
# Fill a job template's placeholders and write the job script to $QNNWIND_SCRATCH/jobs.
#
#   slurm/render.sh TEMPLATE ACCOUNT QOS WALLTIME NODES
#   sbatch "$(slurm/render.sh slurm/calibrate.sbatch m1234 regular 01:00:00 1)"
#   sbatch "$(QNNWIND_CONFIG=configs/random.yaml slurm/render.sh slurm/qnn.sbatch m1234 regular 14:00:00 6)"
#
# Prints the path of the rendered script. Besides <ACCOUNT>, <QOS>, <WALLTIME>, and <NODES>,
# it fills <QNNWIND_REPO>, <QNNWIND_SCRATCH>, and <QNNWIND_EXTRAS> from the environment
# variables of the same names (defaults: $SCRATCH/qnn-wind-prediction, $SCRATCH/qnn-wind-runs,
# $SCRATCH/qnn-wind-pyextras), and creates the run directory's logs/ (Slurm output files) and
# jobs/ (rendered scripts).
#
# If QNNWIND_CONFIG is set when rendering, it becomes the job's default config (so that a
# later resubmission of the same script runs the same batch) and the rendered file is named
# <config stem>-<template>, e.g. jobs/random-qnn.sbatch.
set -euo pipefail
if [ "$#" -ne 5 ]; then
    echo "usage: $0 TEMPLATE ACCOUNT QOS WALLTIME NODES" >&2
    exit 2
fi
template="$1"
QNNWIND_REPO="${QNNWIND_REPO:-${SCRATCH:?SCRATCH is not set}/qnn-wind-prediction}"
QNNWIND_SCRATCH="${QNNWIND_SCRATCH:-${SCRATCH:?SCRATCH is not set}/qnn-wind-runs}"
QNNWIND_EXTRAS="${QNNWIND_EXTRAS:-${SCRATCH:?SCRATCH is not set}/qnn-wind-pyextras}"
mkdir -p "$QNNWIND_SCRATCH/logs" "$QNNWIND_SCRATCH/jobs"
name="$(basename "$template")"
config_rule=()
if [ -n "${QNNWIND_CONFIG:-}" ]; then
    name="$(basename "$QNNWIND_CONFIG" .yaml)-$name"
    config_rule=(-e "s|\${QNNWIND_CONFIG:-[^}]*}|\${QNNWIND_CONFIG:-$QNNWIND_CONFIG}|")
fi
out="$QNNWIND_SCRATCH/jobs/$name"
sed -e "s|<ACCOUNT>|$2|g" -e "s|<QOS>|$3|g" -e "s|<WALLTIME>|$4|g" -e "s|<NODES>|$5|g" \
    -e "s|<QNNWIND_REPO>|$QNNWIND_REPO|g" -e "s|<QNNWIND_SCRATCH>|$QNNWIND_SCRATCH|g" \
    -e "s|<QNNWIND_EXTRAS>|$QNNWIND_EXTRAS|g" ${config_rule[@]+"${config_rule[@]}"} \
    "$template" > "$out"
if grep -q '<[A-Z_]*>' "$out"; then
    echo "unfilled placeholder in $out" >&2
    exit 1
fi
echo "$out"
