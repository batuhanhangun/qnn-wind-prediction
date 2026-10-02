"""Single-run CLI: one tuning study or one final run per invocation.

Usage (from the project root, with ``src`` on ``PYTHONPATH``; CONFIG is e.g.
``configs/experiment.yaml``)::

    python -m qnnwind.runner run --config CONFIG --model QNN-1 --n 750 --fold 0 --seed 0
    python -m qnnwind.runner tune --config CONFIG --model SVR --n 750 --fold 0
    python -m qnnwind.runner smoke --config configs/smoke.yaml
    python -m qnnwind.runner circuits --config CONFIG

``--set key=value`` overrides any existing config key (the override changes the config hash).
"""

from __future__ import annotations

import os

# Single-threaded workers: set before numpy or torch is imported.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from qnnwind.circuits import save_entanglement_maps  # noqa: E402
from qnnwind.data import Dataset, load_dataset, make_split  # noqa: E402
from qnnwind.folds import Fold, ranges  # noqa: E402
from qnnwind.io import (  # noqa: E402
    Config,
    finalize_dir,
    load_config,
    parse_override,
    peak_rss_mb,
    read_json,
    run_dir,
    run_metadata,
    staging_dir,
    tuning_dir,
    utc_now,
    validate_run_dir,
    write_csv,
    write_json,
)
from qnnwind.metrics import regression_metrics  # noqa: E402

torch.set_num_threads(1)

BASE_FILES = ("result.json", "preds_val.csv", "preds_test.csv")
CURVE_FILES = ("curve_iter.csv", "curve_eval.csv")


def model_kind(config: Config, model: str) -> str:
    """``classical``, ``deep``, ``mlp_pm``, or ``qnn`` from the ``models`` config section."""
    for kind, names in config["models"].items():
        if model in names:
            return kind
    raise ValueError(f"Model {model!r} is not listed in the config")


def is_tuned(config: Config, model: str) -> bool:
    """Whether the model has a search space (all classical and deep models except LR)."""
    return model in config["tuning"]["search_spaces"]


def required_files(config: Config, model: str) -> tuple[str, ...]:
    kind = model_kind(config, model)
    return BASE_FILES + (CURVE_FILES if kind in ("qnn", "mlp_pm") else ())


def require_same_config(config: Config, stored_hash: str | None, path: Path) -> None:
    """Refuse to reuse results produced under a different configuration.

    Completed outputs are skipped only if their config hash matches; otherwise the run stops
    rather than silently reusing or overwriting them.
    """
    if stored_hash != config.hash:
        raise RuntimeError(
            f"{path} was produced with config hash {stored_hash}, but the current config "
            f"hash is {config.hash}. Use a different results directory or move it away."
        )


def load_context(config: Config) -> tuple[Dataset, list[Fold]]:
    dataset = load_dataset(config.path("data"), config["data"])
    return dataset, config.folds(dataset.n_rows)  # blocked or random


# --------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------


def execute_tuning(config: Config, model: str, n_train: int, fold_k: int) -> Path:
    """Run or resume the study for (model, N, fold); skip if best_params.json exists."""
    from qnnwind.tuning import tune

    out = tuning_dir(config.path("results"), model, n_train, fold_k)
    best_file = out / "best_params.json"
    if best_file.is_file():
        require_same_config(config, read_json(best_file).get("config_hash"), best_file)
        print(f"[skip] tuning {model} N={n_train} fold={fold_k}: {best_file} exists")
        return best_file
    dataset, folds = load_context(config)
    split = make_split(
        dataset, folds[fold_k], n_train, int(config["tuning"]["subset_seed"]), config["scaling"]
    )
    t0 = time.perf_counter()
    content = tune(
        model, split.trainval, config, out, study_name=f"{model}_N{n_train}_fold{fold_k}"
    )
    print(
        f"[done] tuning {model} N={n_train} fold={fold_k}: best val RMSE "
        f"{content['best_val_rmse_kw']:.2f} kW in {time.perf_counter() - t0:.1f} s"
    )
    return best_file


# --------------------------------------------------------------------------------------
# Final runs
# --------------------------------------------------------------------------------------


def _build_model(config: Config, model: str, n_train: int, fold_k: int, seed: int) -> Any:
    kind = model_kind(config, model)
    if kind == "qnn":
        from qnnwind.qnn import QNNModel

        return QNNModel(model, config["qnn"], seed), {}
    if kind == "mlp_pm":
        from qnnwind.deep import MLPPM

        return MLPPM(config["mlp_pm"], seed), {}
    from qnnwind.tuning import make_model

    params: dict[str, Any] = {}
    if is_tuned(config, model):
        best_file = tuning_dir(config.path("results"), model, n_train, fold_k) / "best_params.json"
        if not best_file.is_file():
            raise FileNotFoundError(f"{model}: tuning result missing: {best_file}")
        params = read_json(best_file)["params"]
    return make_model(model, params, seed, config), params


def target_scaling(config: Config, model: str) -> dict[str, Any]:
    """The scaling section used for ``model``.

    The target range of a QNN model comes, in order, from its readout variant
    (``qnn.variants.<model>.target_range``: QNN-1u..QNN-6u use [0, 1]), from the optional
    ``qnn.target_range``, or from ``scaling.target_range`` ([-1, 1]), which also applies to
    every other model. experiment.yaml sets neither QNN key.
    """
    scaling = dict(config["scaling"])
    if model_kind(config, model) != "qnn":
        return scaling
    variants = config["qnn"].get("variants") or {}
    if model in variants and "target_range" in variants[model]:
        target = variants[model]["target_range"]
    elif "target_range" in config["qnn"]:
        target = config["qnn"]["target_range"]
    else:
        return scaling
    low, high = (float(v) for v in target)
    if not low < high:
        raise ValueError(f"target range of {model} must be increasing, got {[low, high]}")
    scaling["target_range"] = [low, high]
    return scaling


def readout_label(target_range: list[float]) -> str:
    """``[-1, 1]`` or ``[0, 1]``: the target range a model was trained on."""
    low, high = (float(v) for v in target_range)
    return f"[{low:g}, {high:g}]"


def _predictions_frame(rows: np.ndarray, actual: np.ndarray, **preds: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"row": rows, "actual_kw": actual})
    for name, values in preds.items():
        frame[name] = values
    return frame


def execute_run(
    config: Config, model: str, n_train: int, fold_k: int, seed: int, workers: int = 1
) -> Path:
    """Fit one model for (N, fold, seed) and write its raw results.

    Skips the run if its directory already validates. The test fold is used only for
    ``predict`` after fitting and for the final metrics.
    """
    final = run_dir(config.path("results"), model, n_train, fold_k, seed)
    files = required_files(config, model)
    if validate_run_dir(final, files):
        stored = read_json(final / "result.json")["metadata"]["config_hash"]
        require_same_config(config, stored, final)
        print(f"[skip] {model} N={n_train} fold={fold_k} seed={seed}: already complete")
        return final

    dataset, folds = load_context(config)
    fold = folds[fold_k]
    meta = run_metadata(config, dataset.sha256, workers)
    scaling = target_scaling(config, model)
    split = make_split(dataset, fold, n_train, seed, scaling)
    trainval = split.trainval
    estimator, params = _build_model(config, model, n_train, fold_k, seed)

    t0 = time.perf_counter()
    estimator.fit(trainval)
    fit_time = time.perf_counter() - t0

    y_scaler = trainval.y_scaler
    val_actual = y_scaler.inverse(trainval.y_val)
    val_pred = y_scaler.inverse(estimator.predict(trainval.X_val))
    test_pred = y_scaler.inverse(estimator.predict(split.X_test))
    metrics = {
        "val": regression_metrics(val_actual, val_pred),
        "test": regression_metrics(split.y_test_kw, test_pred),
    }
    val_frame = {"pred_kw": val_pred}
    test_frame = {"pred_kw": test_pred}
    if hasattr(estimator, "predict_final"):
        val_final = y_scaler.inverse(estimator.predict_final(trainval.X_val))
        test_final = y_scaler.inverse(estimator.predict_final(split.X_test))
        metrics["val_final"] = regression_metrics(val_actual, val_final)
        metrics["test_final"] = regression_metrics(split.y_test_kw, test_final)
        val_frame["pred_final_kw"] = val_final
        test_frame["pred_final_kw"] = test_final

    staging = staging_dir(final)
    write_csv(
        staging / "preds_val.csv", _predictions_frame(trainval.val_rows, val_actual, **val_frame)
    )
    write_csv(
        staging / "preds_test.csv",
        _predictions_frame(split.test_rows, split.y_test_kw, **test_frame),
    )
    if "curve_iter.csv" in files:
        write_csv(staging / "curve_iter.csv", estimator.curve_iter)
        write_csv(staging / "curve_eval.csv", estimator.curve_eval)
    result = {
        "status": "complete",
        "model": model,
        "kind": model_kind(config, model),
        "protocol": config.protocol,
        "readout": readout_label(scaling["target_range"]),
        "n_train": n_train,
        "fold": fold_k,
        "seed": seed,
        "blocks": {
            "train_rows": int(trainval.train_rows.size),
            "train_pool_ranges": ranges(fold.train_pool),
            "validation_ranges": ranges(fold.validation),
            "test_ranges": ranges(fold.test),
            "buffer_ranges": ranges(fold.buffers),
        },
        "scaling": {
            "x_min": trainval.x_scaler.data_min,
            "x_max": trainval.x_scaler.data_max,
            "y_min_kw": float(y_scaler.data_min[0]),
            "y_max_kw": float(y_scaler.data_max[0]),
            "target_range": list(scaling["target_range"]),
        },
        "hyperparameters": params,
        "fit_time": fit_time,
        "model_summary": estimator.summary,
        "metrics": metrics,
        "metadata": {**meta, "end_time": utc_now(), "peak_rss_mb": peak_rss_mb()},
    }
    write_json(staging / "result.json", result)
    finalize_dir(staging, final)
    print(
        f"[done] {model} N={n_train} fold={fold_k} seed={seed}: test RMSE "
        f"{metrics['test']['rmse']:.2f} kW, val RMSE {metrics['val']['rmse']:.2f} kW, "
        f"fit {fit_time:.1f} s"
    )
    return final


# --------------------------------------------------------------------------------------
# Smoke test and circuits
# --------------------------------------------------------------------------------------


def all_models(config: Config) -> list[str]:
    return [m for names in config["models"].values() for m in names]


def execute_smoke(config: Config) -> None:
    """Tune every tunable model, then run every model, over the config's (small) grid."""
    save_entanglement_maps(
        config.path("outputs") / "tables" / "entanglement_maps.json", config["qnn"]
    )
    models = all_models(config)
    for fold_k in config["folds"]["run"]:
        for n_train in config["sizes"]:
            for model in models:
                if is_tuned(config, model):
                    execute_tuning(config, model, n_train, fold_k)
            for seed in config["seeds"]:
                for model in models:
                    execute_run(config, model, n_train, fold_k, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "tune", "smoke", "circuits"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name in ("run", "tune"):
            p.add_argument("--model", required=True)
            p.add_argument("--n", required=True, type=int)
            p.add_argument("--fold", required=True, type=int)
        if name == "run":
            p.add_argument("--seed", required=True, type=int)
            p.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    config = load_config(args.config, dict(parse_override(s) for s in args.set))

    if args.command == "run":
        execute_run(config, args.model, args.n, args.fold, args.seed, args.workers)
    elif args.command == "tune":
        execute_tuning(config, args.model, args.n, args.fold)
    elif args.command == "smoke":
        execute_smoke(config)
    else:
        path = config.path("outputs") / "tables" / "entanglement_maps.json"
        save_entanglement_maps(path, config["qnn"])
        print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
