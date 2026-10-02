"""Build the results package (results/) from raw run directories.

For each batch (``--source NAME CONFIG RUNS``, as for scripts/analyze.py), the raw results are
read exactly as the analysis reads them (scripts/aggregate.py) and reduced to what the tables
and figures need:

* ``runs.csv.gz``: one row per run: model, kind, N, fold, seed, readout, validation and test
  metrics, the clipped test metrics of T3c, parameter counts and complexity measures, the
  QNN and MLP-PM optimizer summary and timing, the node load, the hyperparameters, and the
  configuration hash and dataset SHA-256 of the run;
* ``errors_test.csv.gz``: the signed test error (prediction minus actual, kW) of every test
  sample of every run, identified by its row index; no predictions, actual values, or inputs;
* ``pooled_seed.csv.gz``: metrics of the pooled out-of-fold predictions per (model, N, seed);
* ``curves_iter.csv.gz``, ``curves_eval.csv.gz``: QNN training and validation MSE per
  iteration, and training MSE per objective evaluation;
* ``tuning.csv.gz``: the tuned hyperparameters per (model, N, fold).

``manifest.json`` lists the batches (configuration, protocol, configuration hash, dataset
SHA-256, number of runs) and the SHA-256 of every file. Floats are written in their shortest
exact form and read back exactly, and the gzip headers carry no time stamp, so the package is
reproducible byte for byte.

Usage::

    python scripts/build_results_package.py --source primary configs/experiment.yaml RUNS1 \\
        --source blocked_unit configs/blocked_unit.yaml RUNS2 \\
        --source random configs/random.yaml RUNS2 [--out results]
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import aggregate  # noqa: E402
from qnnwind.io import write_json  # noqa: E402

METRIC_COLUMNS = [
    f"{split}_{m}"
    for split in ("val", "test")
    for m in ("r2", "rmse", "mae", "bias", "error_std", "n_negative", "frac_negative")
]
RUN_COLUMNS = [
    "model", "kind", "n_train", "fold", "seed", "readout",
    *METRIC_COLUMNS,
    "test_clip_rmse", "test_clip_r2", "test_clip_mae",
    "trainable_params", "tree_nodes", "tree_leaves", "support_vectors",
    "stored_training_samples",
    "total_training_time", "time_per_evaluation", "nit", "nfev", "scipy_message",
    "stopped_before_maxiter", "cache_near_hits", "best_is_initial",
    "node_busy_mean", "node_busy_min",
    "hyperparameters", "config_hash", "dataset_sha256",
]  # fmt: skip
TABLE_COLUMNS = {
    "errors": [*aggregate.KEY_COLUMNS, "row", "error_kw"],
    "pooled_seed": ["model", "n_train", "seed", "r2", "rmse", "mae", "bias"],
    "curves_iter": [*aggregate.KEY_COLUMNS, "iteration", "train_mse", "val_mse", "padded"],
    "curves_eval": [*aggregate.KEY_COLUMNS, "evaluation", "train_mse"],
    "tuning": ["batch", "model", "n_train", "fold", "params", "best_val_rmse_kw", "trials"],
}
# Never part of the package: per-sample data from which the dataset could be rebuilt, and
# machine or person related run metadata.
FORBIDDEN = {
    "actual_kw", "pred_kw", "pred_final_kw", "git_commit", "git_dirty", "git_source",
    "hostname", "cpu_model", "platform", "peak_rss_mb", "start_time", "end_time",
}  # fmt: skip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    assert not FORBIDDEN & set(columns), FORBIDDEN & set(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=columns).to_csv(
        path,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )


def config_name(path: Path) -> str:
    """The configuration path as recorded in the manifest (relative to the repository)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build(sources: list[aggregate.Source], out: Path, check_hashes: bool = True) -> dict:
    """Write the package of ``sources`` to ``out``. Every run must have been produced with
    the configuration's hash (``check_hashes=False`` only for synthetic test data)."""
    batches, files = [], {}
    for source in sources:
        if source.package:
            raise SystemExit(f"{source.name}: expected raw results, got a package directory")
        parts = aggregate.collect(source)
        runs = parts["runs"]
        target = out / source.name
        tables = {
            "runs": (runs, RUN_COLUMNS),
            "errors": (parts["predictions"], TABLE_COLUMNS["errors"]),
            "pooled_seed": (parts["pooled_seed"], TABLE_COLUMNS["pooled_seed"]),
            "curves_iter": (parts["curves_iter"], TABLE_COLUMNS["curves_iter"]),
            "curves_eval": (parts["curves_eval"], TABLE_COLUMNS["curves_eval"]),
            "tuning": (aggregate.hyperparameter_table(source), TABLE_COLUMNS["tuning"]),
        }
        for name, (frame, columns) in tables.items():
            write(frame, target / aggregate.PACKAGE_FILES[name], columns)
        hashes = sorted(runs["config_hash"].dropna().unique())
        datasets = sorted(runs["dataset_sha256"].dropna().unique())
        if check_hashes and hashes != [source.config.hash]:
            raise SystemExit(f"{source.name}: run config hashes {hashes} != {source.config.hash}")
        batches.append(
            {
                "name": source.name,
                "directory": source.name,
                "config": config_name(source.config.source),
                "protocol": source.protocol,
                "config_hash": source.config.hash,
                "dataset_sha256": datasets,
                "runs": int(len(runs)),
                "expected_runs": int(len(aggregate.expected_runs(source))),
            }
        )
        print(f"build_results_package: {source.name}: {len(runs)} runs -> {target}")
    written = [out / s.name / f for s in sources for f in aggregate.PACKAGE_FILES.values()]
    for path in sorted(written):
        files[path.relative_to(out).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "description": "Results package of the paper; see results/README.md.",
        "batches": batches,
        "files": files,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", nargs=3, action="append", required=True, metavar=("NAME", "CONFIG", "RUNS")
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    sources = [aggregate.make_source(n, c, Path(r)) for n, c, r in args.source]
    manifest = build(sources, args.out.resolve())
    total = sum(f["bytes"] for f in manifest["files"].values())
    largest = max(f["bytes"] for f in manifest["files"].values())
    print(f"build_results_package: {total / 1e6:.1f} MB in {len(manifest['files'])} files "
          f"(largest {largest / 1e6:.1f} MB)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
