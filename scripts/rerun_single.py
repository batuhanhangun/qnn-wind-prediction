"""Level 2: rerun one archived run and compare its test metrics with the archived result.

A run is identified by protocol, model, training size N, fold, and seed. The run uses the
configuration of its batch (blocked protocol: configs/experiment.yaml, or
configs/blocked_unit.yaml for QNN-1u to QNN-6u; random protocol: configs/random.yaml) and,
for tuned models, the archived tuned hyperparameters of its (model, N, fold). It is executed
exactly as the launcher executes it (``python -m qnnwind.runner run``, single-threaded) in a
fresh directory under the run directory, and its test metrics and per-sample test errors are
compared with the archived ones: by default from the results package (``results/``), or from
a raw run directory given with ``--archive``. Needs the dataset.

Usage::

    python scripts/rerun_single.py --protocol blocked --model QNN-3u --n 750 --fold 1 --seed 0
        [--archive DIR] [--runs DIR]
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate import require_dataset  # noqa: E402
from qnnwind.io import Config, load_config, read_json, run_dir, tuning_dir, write_json  # noqa: E402
from qnnwind.runner import is_tuned  # noqa: E402
from runs_dir import add_runs_argument, use_runs_dir  # noqa: E402

CONFIGS = ROOT / "configs"
ARCHIVE = ROOT / "results"
METRICS = ("rmse", "mae", "r2", "bias", "error_std", "n_negative", "frac_negative")


def config_for(protocol: str, model: str) -> Path:
    """The configuration of the batch that holds ``model`` under ``protocol``."""
    blocked = ["experiment.yaml", "blocked_unit.yaml"]
    candidates = ["random.yaml"] if protocol == "random" else blocked
    for name in candidates:
        config = load_config(CONFIGS / name)
        if any(model in names for names in config["models"].values()):
            return CONFIGS / name
    raise SystemExit(f"model {model!r} is not part of the {protocol} protocol")


def with_runs(path: Path, runs: Path) -> tuple[Config, Path]:
    """The configuration at ``path`` and its results directory under the run directory
    ``runs`` (paths are expanded when resolved, so the results directory is resolved here)."""
    previous = os.environ.get("QNNWIND_SCRATCH")
    os.environ["QNNWIND_SCRATCH"] = str(runs)
    try:
        config = load_config(path)
        return config, config.path("results")
    finally:
        if previous is None:
            os.environ.pop("QNNWIND_SCRATCH", None)
        else:
            os.environ["QNNWIND_SCRATCH"] = previous


def archived_run(results: Path, model: str, n: int, fold: int, seed: int) -> dict:
    """Test metrics, hyperparameters, and per-sample test errors of a raw archived run."""
    directory = run_dir(results, model, n, fold, seed)
    result = read_json(directory / "result.json")
    preds = pd.read_csv(directory / "preds_test.csv").sort_values("row", kind="stable")
    return {
        "metrics": result["metrics"]["test"],
        "hyperparameters": result.get("hyperparameters") or {},
        "errors": (preds["pred_kw"] - preds["actual_kw"]).to_numpy(),
        "config_hash": result["metadata"]["config_hash"],
    }


def packaged_run(
    archive: Path, config_path: Path, model: str, n: int, fold: int, seed: int
) -> dict:
    """The same, from the results package (the batch of ``config_path`` in its manifest)."""
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    config = config_path.relative_to(ROOT).as_posix()
    batch = next(b for b in manifest["batches"] if b["config"] == config)
    directory = archive / batch["directory"]

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        key = (frame["model"] == model) & (frame["n_train"] == n) & (frame["fold"] == fold)
        return frame[key & (frame["seed"] == seed)]

    read = {"float_precision": "round_trip"}
    runs = select(pd.read_csv(directory / "runs.csv.gz", **read))
    if len(runs) != 1:
        raise SystemExit(f"{model} N={n} fold={fold} seed={seed}: not in {directory}")
    run = runs.iloc[0]
    errors = select(pd.read_csv(directory / "errors_test.csv.gz", **read)).sort_values(
        "row", kind="stable"
    )
    return {
        "metrics": {m: run[f"test_{m}"] for m in METRICS},
        "hyperparameters": json.loads(run["hyperparameters"]),
        "errors": errors["error_kw"].to_numpy(),
        "config_hash": run["config_hash"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=("blocked", "random"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="archived results")
    add_runs_argument(parser)
    args = parser.parse_args(argv)
    runs = use_runs_dir(args.runs)

    config_path = config_for(args.protocol, args.model)
    key = f"{args.protocol}__{args.model}__N{args.n}__fold{args.fold}__seed{args.seed}"
    work = runs / "rerun" / key
    if work.exists():
        shutil.rmtree(work)  # always a fresh rerun
    key_args = (args.model, args.n, args.fold, args.seed)
    if (args.archive / "manifest.json").is_file():
        archive = packaged_run(args.archive.resolve(), config_path, *key_args)
    else:
        _, archived_results = with_runs(config_path, args.archive.resolve())
        archive = archived_run(archived_results, *key_args)
    config, results = with_runs(config_path, work)
    require_dataset(config)
    if archive["config_hash"] != config.hash:
        raise SystemExit(
            f"archived run: config hash {archive['config_hash']}; {config_path.name}: {config.hash}"
        )
    if is_tuned(config, args.model):
        study = tuning_dir(results, args.model, args.n, args.fold)
        write_json(study / "best_params.json", {"params": archive["hyperparameters"]})

    env = {**os.environ, "QNNWIND_SCRATCH": str(work), "PYTHONPATH": str(ROOT / "src")}
    argv_run = [sys.executable, "-m", "qnnwind.runner", "run", "--config", str(config_path)]
    argv_run += ["--model", args.model, "--n", str(args.n), "--fold", str(args.fold)]
    argv_run += ["--seed", str(args.seed), "--workers", "1"]
    print(f"rerun_single: {key} ({config_path.name}) in {work}", flush=True)
    start = time.perf_counter()
    subprocess.run(argv_run, cwd=ROOT, env=env, check=True)
    seconds = time.perf_counter() - start

    directory = run_dir(results, args.model, args.n, args.fold, args.seed)
    rerun = read_json(directory / "result.json")["metrics"]["test"]
    preds = pd.read_csv(directory / "preds_test.csv").sort_values("row", kind="stable")
    errors = (preds["pred_kw"] - preds["actual_kw"]).to_numpy()

    print(f"\n{'test metric':<14}{'archived':>20}{'rerun':>20}{'|difference|':>16}")
    exact = True
    for name in METRICS:
        a, b = archive["metrics"][name], rerun[name]
        exact &= a == b
        print(f"{name:<14}{a:>20.10g}{b:>20.10g}{abs(b - a):>16.3g}")
    per_sample = float(np.max(np.abs(errors - archive["errors"])))
    exact &= per_sample == 0.0
    rmse_diff = abs(rerun["rmse"] - archive["metrics"]["rmse"])
    print(f"largest per-sample test error difference: {per_sample:.3g} kW")
    verdict = "exact" if exact else f"largest |RMSE difference| {rmse_diff:.3g} kW"
    print(f"rerun_single: {key}: {verdict} ({seconds:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
