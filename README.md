# Quantum neural networks for wind power prediction

[![CI](https://github.com/batuhanhangun/qnn-wind-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/batuhanhangun/qnn-wind-prediction/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/ZENODO_DOI.svg)](https://doi.org/ZENODO_DOI)

Code and archived results for the paper

> B. Hangun, O. Eyecioglu, M. Ali, O. Altun, K. Kayisli. *Quantum Neural Networks for Wind
> Power Prediction: A Benchmark Against Classical Machine Learning Models.* Quantum
> Information Processing. DOI: PAPER_DOI

Six 4-qubit QNNs (Qiskit 2.3.0, Qiskit Machine Learning 0.9.0) with the regression target
scaled to [-1, 1] (QNN-1 to QNN-6) or [0, 1] (QNN-1u to QNN-6u), against 11 classical and deep
baselines, on four meteorological inputs of one wind turbine. Evaluation: blocked
cross-validation with buffers (primary) and random cross-validation (comparison); 6 folds, 4 training
sizes, 5 seeds.

## Repository layout

| Path | Contents |
|---|---|
| `src/qnnwind/` | Library: data, folds, circuits, QNN, classical and deep models, tuning, metrics, statistics, single-run CLI |
| `scripts/` | Entry points (below), analysis, task farm |
| `configs/` | `experiment.yaml`, `blocked_unit.yaml`, `random.yaml` (the paper), `quick.yaml`, `smoke*.yaml` |
| `results/` | Results package of the paper ([results/README.md](results/README.md)) |
| `paper/` | Reference copies of the paper's tables and figures |
| `data/` | Dataset description ([data/README.md](data/README.md)); the dataset is not included |
| `environment/` | Pinned package versions for Windows, Linux, macOS, and the cluster container |
| `slurm/` | Optional Slurm templates and container wrapper |
| `tests/` | pytest suite |

## Installation

Python 3.11, from the repository root. The constraints files pin every package version.

**Windows** (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
.\.venv\Scripts\python.exe -m pip install -c environment/constraints-local-win.txt -r environment/requirements-local.txt -r environment/requirements-extra.txt torch==2.5.1+cpu
.\.venv\Scripts\python.exe scripts/check_env.py
```

**Linux** (x86_64; LightGBM needs the OpenMP runtime, e.g. `sudo apt install libgomp1`):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
.venv/bin/python -m pip install -c environment/constraints-local-linux.txt -r environment/requirements-local.txt -r environment/requirements-extra.txt torch==2.5.1+cpu
.venv/bin/python scripts/check_env.py
```

**macOS** (Apple Silicon only; torch 2.5.1 has no Intel Mac wheels; XGBoost and LightGBM need
`brew install libomp`):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -c environment/constraints-local-macos.txt -r environment/requirements-local.txt -r environment/requirements-extra.txt torch==2.5.1
.venv/bin/python scripts/check_env.py
```

Below, `python` is the environment's interpreter.

## Dataset

Not included; available from the corresponding author of the paper on reasonable request.
Format, checksum, and location: [data/README.md](data/README.md). Level 1 runs without it.

## Reproducing the paper

| Level | Command | Dataset | Time |
|---|---|---|---|
| 1. Paper from the archived results | `python scripts/reproduce_paper.py` | optional | about 2 min |
| 2. One run, compared with its archived result | `python scripts/rerun_single.py --protocol blocked --model QNN-3u --n 750 --fold 1 --seed 0` | required | seconds (baselines); about 80 min (QNN, N = 750) |
| 3. Full grid | `python scripts/run_grid.py --config configs/experiment.yaml` (and `blocked_unit.yaml`, `random.yaml`) | required | about 110 node-hours on 64-worker nodes (7,000 core-hours) |
| Quick benchmark | `python scripts/run_grid.py --config configs/quick.yaml` | required | about 50 min on 8 cores |

- **Level 1** regenerates every table and figure from `results/` into `runs/paper/` and compares
  each file with `paper/`: tables byte-identical, figures identical up to rendering differences
  between platforms. Without the dataset, T1, T1b, and A1 are copied from `paper/` instead of
  regenerated.
- **Level 2** reruns one (protocol, model, N, fold, seed) with the archived hyperparameters and
  reports the difference in every test metric and in the per-sample errors.
- **Level 3** runs every task of a configuration, then the analysis. For the paper's combined
  analysis of three local grids:

  ```bash
  python scripts/analyze.py --source primary configs/experiment.yaml runs --source blocked_unit configs/blocked_unit.yaml runs --source random configs/random.yaml runs --outputs runs/outputs/paper
  ```

- **Quick benchmark**: a reduced run (two folds, one seed, 60 QNN iterations) that shows the main
  effects; its numbers are not the paper's.

Results, logs, and task files go to `runs/` (or `--runs DIR`, or `$QNNWIND_SCRATCH`). Completed
runs are skipped, so an interrupted grid resumes. `scripts/launcher.py` uses physical cores
minus one worker processes (`--workers W`); `scripts/status.py --config ...` reports progress.
Tests: `python -m pytest`; smoke grid: `python scripts/run_grid.py --config configs/smoke.yaml`.
Without the dataset, `python scripts/make_synthetic_dataset.py` writes a synthetic dataset of the
same shape and a smoke configuration for it, `runs/synthetic/smoke_synthetic.yaml` (as in CI).

## Agreement of reruns with the archived results

The archived runs were computed on Linux (x86_64). Level 2 reruns from a fresh clone, absolute
difference in test RMSE (exact: every test metric and every per-sample error identical):

| Run (protocol, model, N, fold, seed) | Windows | Linux |
|---|---|---|
| blocked, QNN-3u, 750, 1, 0 | 8e-10 kW | exact |
| random, QNN-6, 750, 2, 3 | 5e-8 kW | |
| blocked, LR, 3000, 2, 4 | 0 kW | |
| random, SVR, 1500, 3, 1 | 2e-11 kW | |
| blocked, LightGBM, 2250, 0, 2 | exact | |
| random, LSTM, 750, 4, 0 | 3e-6 kW | |

## Cluster use (optional)

The `slurm/` templates run the grid on CPU nodes with Slurm and podman-hpc: one container per
node (`slurm/in_container.sh`), `scripts/launcher.py --cluster` with W = 64 workers per node,
and dynamic task claiming across nodes.

| Variable | Meaning | Default |
|---|---|---|
| `QNNWIND_REPO` | Clone of this repository | `$SCRATCH/qnn-wind-prediction` |
| `QNNWIND_SCRATCH` | Run directory | `$SCRATCH/qnn-wind-runs` |
| `QNNWIND_EXTRAS` | Directory with the packages of `environment/requirements-extra.txt` | `$SCRATCH/qnn-wind-pyextras` |
| `QNNWIND_IMAGE` | Container image with the packages of `environment/v4_freeze.txt` | `quantum_ml:v4` |

```bash
mkdir -p $QNNWIND_EXTRAS     # once: the extra packages, outside the image
podman-hpc run --rm -v $QNNWIND_REPO:/repo:ro -v $QNNWIND_EXTRAS:/extras $QNNWIND_IMAGE \
    python -m pip install --no-cache-dir --no-deps --target /extras -r /repo/environment/requirements-extra.txt
slurm/in_container.sh scripts/check_env.py --mode container
slurm/in_container.sh scripts/make_tasks.py --config configs/experiment.yaml
TUNE=$(sbatch --parsable "$(slurm/render.sh slurm/tune.sbatch <ACCOUNT> <QOS> 06:00:00 1)")
sbatch "$(slurm/render.sh slurm/qnn.sbatch <ACCOUNT> <QOS> 14:00:00 3)"
sbatch --dependency=afterok:$TUNE "$(slurm/render.sh slurm/classical.sbatch <ACCOUNT> <QOS> 02:00:00 1)"
slurm/in_container.sh scripts/status.py --config configs/experiment.yaml
```

A job cut by its walltime is resubmitted with `sbatch $QNNWIND_SCRATCH/jobs/<name>.sbatch`;
completed tasks are skipped.

Set `QNNWIND_CONFIG=configs/random.yaml` (or `blocked_unit.yaml`) when rendering to run another
configuration. Other templates: `aggregate`, `calibrate`, `smoke_debug`, `multinode_check`.

## Provenance

Every archived run records the SHA-256 hash of its configuration and the SHA-256 checksum of
the dataset. The configurations in this repository have the same hashes.
`python scripts/verify_provenance.py` checks this, and checks the dataset checksum when the
dataset is present.

| Batch | Runs | Configuration hash (SHA-256) |
|---|---|---|
| `configs/experiment.yaml` | 2040 | `5951445b7460403758f0be4ce3a105ed4a04a176a3b50edf265534dea42c5657` |
| `configs/blocked_unit.yaml` | 720 | `07da5c2e88db2dfdb7f1f5426616144863bb94750c83774bcf8cbf185e95541e` |
| `configs/random.yaml` | 2760 | `027cf912ac3dfe53148516d3d037571593c115abc69ffb75a58ee59f963e9e1c` |

## Citation and license

Please cite the paper ([CITATION.cff](CITATION.cff)). Code: MIT license ([LICENSE](LICENSE)).
