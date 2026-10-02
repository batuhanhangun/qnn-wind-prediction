"""A synthetic run directory in the exact format the runner and launcher write, with known
properties, for testing the analysis without training anything.

Primary batch. Grid: 6 folds x N in {750, 1500} x seeds {0, 1, 2}; models LR, SVR (tuned),
MLP (tuned), MLP-PM, QNN-1, QNN-5, QNN-6. Known properties:

* QNN-1 and QNN-5 have identical predictions and curves (the same function); QNN-5 has the
  lowest validation error of the QNNs, so the best QNN is QNN-5 (tie with QNN-1, fewer gates);
* time per objective evaluation = rate[config] * N, with QNN-1 / QNN-5 = 40 / 34;
* node load: 64 busy workers, except fold 0 (busy_mean 50: excluded) and fold 1 seed 2 at
  N = 750 (no node_load record: excluded);
* QNN curves: 60 evaluations for seed 0, 30 for seeds 1 and 2 (padded in the 50-window);
  QNN-6 has one early stop with cache near hits per N.

``build_all`` also writes the blocked_unit and random batches into the same run
directory, each with its own config, results subtree, commit, and config hash:

* ``blocked_unit``: QNN-1u, QNN-5u, QNN-6u, with the properties of their [-1, 1] counterparts,
  lower noise, and curves at a quarter of the level;
* ``random``: the random K-fold protocol, every model above, noise times 0.8.

These write ``protocol``, ``readout``, and ``scaling`` into result.json as the runner does;
the primary batch omits ``protocol`` and ``readout``, like its archived results.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from qnnwind.data import load_dataset
from qnnwind.io import load_config, run_dir, tuning_dir, write_csv, write_json
from qnnwind.metrics import regression_metrics
from qnnwind.runner import readout_label, target_scaling

MODELS = {
    "classical": ["LR", "SVR"],
    "deep": ["MLP"],
    "mlp_pm": ["MLP-PM"],
    "qnn": ["QNN-1", "QNN-5", "QNN-6"],
}
NOISE = {
    "LR": 300.0,
    "SVR": 150.0,
    "MLP": 200.0,
    "MLP-PM": 350.0,
    "QNN-1": 250.0,
    "QNN-5": 250.0,
    "QNN-6": 280.0,
    "QNN-1u": 180.0,
    "QNN-5u": 180.0,
    "QNN-6u": 210.0,
}
RATE = {"QNN-1": 40.0e-4, "QNN-5": 34.0e-4, "QNN-6": 34.0e-4}  # seconds per sample per evaluation
RATE.update({f"{m}u": r for m, r in list(RATE.items())})
SIZES, SEEDS = (750, 1500), (0, 1, 2)
COMMIT, CONFIG_HASH = "a" * 40, "c" * 64
UNIT_MODELS = {"classical": [], "deep": [], "mlp_pm": [], "qnn": ["QNN-1u", "QNN-5u", "QNN-6u"]}
RANDOM_MODELS = {**MODELS, "qnn": MODELS["qnn"] + UNIT_MODELS["qnn"]}
RANDOM_NOISE = 0.8  # the random split is optimistic
SAME = {"QNN-5": "QNN-1", "QNN-5u": "QNN-1u"}  # same unitary: identical predictions
# name: (base config, models, results subtree, commit, config hash)
BATCHES = {
    "primary": ("experiment.yaml", MODELS, "results", COMMIT, CONFIG_HASH),
    "blocked_unit": ("blocked_unit.yaml", UNIT_MODELS, "results/blocked_unit", "b" * 40, "d" * 64),
    "random": ("random.yaml", RANDOM_MODELS, "results/random", "e" * 40, "f" * 64),
}


def write_config(
    root: Path,
    repo: Path,
    name: str = "synthetic",
    base: str = "experiment.yaml",
    models: dict | None = None,
    results: str = "results",
) -> Path:
    cfg = root / f"{name}.yaml"
    models = models or MODELS
    models_yaml = "{" + ", ".join(f"{k}: [{', '.join(v)}]" for k, v in models.items()) + "}"
    lines = [
        f"base: {(repo / 'configs' / base).as_posix()}",
        "paths:",
        f"  data: {(repo / 'data' / 'total_dataset.csv').as_posix()}",
        "  results: ${QNNWIND_SCRATCH:-.}/" + results,
        f"  outputs: {(root / 'outputs').as_posix()}",
        f"models: {models_yaml}",
        f"sizes: [{', '.join(map(str, SIZES))}]",
        f"seeds: [{', '.join(map(str, SEEDS))}]",
        "",
    ]
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return cfg


def _curve(n_evals: int, level: float, early: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    evals = np.arange(1, n_evals + 1)
    train = level + 0.5 * np.exp(-evals / 8.0)
    train[12] += 0.05  # a spike after evaluation 10 (MS > 0)
    curve_eval = pd.DataFrame(
        {
            "evaluation": evals,
            "train_mse": train,
            "wall_time": evals * 1.0,
            "optimizer_time": evals * 0.9,
            "cache": ["near" if early and i > n_evals - 3 else "miss" for i in range(n_evals)],
        }
    )
    iters = np.arange(0, 101)
    val = level * 0.8 + 0.4 * np.exp(-iters / 10.0)
    curve_iter = pd.DataFrame(
        {
            "iteration": iters,
            "val_mse": val,
            "train_mse": level + 0.5 * np.exp(-iters / 9.0),
            "grad_norm": np.exp(-iters / 20.0),
            "padded": iters > 80,
        }
    )
    return curve_iter, curve_eval


def build(root: Path, repo: Path) -> Path:
    """Write the synthetic primary batch under ``root``; returns the config path."""
    return build_batch(root, repo, "primary", config_name="synthetic")


def build_all(root: Path, repo: Path) -> dict[str, Path]:
    """Write every batch (primary, blocked_unit, random) under ``root``; returns the configs."""
    return {name: build_batch(root, repo, name) for name in BATCHES}


def build_batch(root: Path, repo: Path, batch: str, config_name: str | None = None) -> Path:
    """Write one synthetic batch under ``root`` (the run directory); returns the config path."""
    base, models, results_sub, commit, config_hash = BATCHES[batch]
    cfg = write_config(root, repo, config_name or batch, base, models, results_sub)
    previous = os.environ.get("QNNWIND_SCRATCH")
    os.environ["QNNWIND_SCRATCH"] = str(root)
    try:
        config = load_config(cfg)
        results = config.path("results")
    finally:
        if previous is None:
            del os.environ["QNNWIND_SCRATCH"]
        else:
            os.environ["QNNWIND_SCRATCH"] = previous
    dataset = load_dataset(config.path("data"), config["data"])
    folds = config.folds(dataset.n_rows)
    kinds = {m: k for k, names in models.items() for m in names}
    factor = RANDOM_NOISE if config.protocol == "random" else 1.0

    for n in SIZES:
        for fold in folds:
            for model in (m for m in ("SVR", "MLP") if m in kinds):
                write_json(
                    tuning_dir(results, model, n, fold.k) / "best_params.json",
                    {
                        "params": {"C": 10.0} if model == "SVR" else {"width": 32},
                        "best_val_rmse_kw": 200.0,
                        "n_trials_completed": 30,
                    },
                )
            for seed in SEEDS:
                for model, kind in kinds.items():
                    source = SAME.get(model, model)
                    rng = np.random.default_rng(
                        zlib.crc32(f"{batch}|{source}|{n}|{fold.k}|{seed}".encode())
                    )
                    scale = NOISE[model] * factor
                    scale *= 0.9 if model in ("QNN-1", "QNN-5", "QNN-1u", "QNN-5u") else 1.0
                    test_pred = dataset.target[fold.test] + rng.normal(0, scale, fold.test.size)
                    val_pred = dataset.target[fold.validation] + rng.normal(
                        0, scale, fold.validation.size
                    )
                    directory = run_dir(results, model, n, fold.k, seed)
                    write_csv(
                        directory / "preds_test.csv",
                        pd.DataFrame(
                            {
                                "row": fold.test,
                                "actual_kw": dataset.target[fold.test],
                                "pred_kw": test_pred,
                            }
                        ),
                    )
                    write_csv(
                        directory / "preds_val.csv",
                        pd.DataFrame(
                            {
                                "row": fold.validation,
                                "actual_kw": dataset.target[fold.validation],
                                "pred_kw": val_pred,
                            }
                        ),
                    )
                    summary: dict = {
                        "model": model,
                        "trainable_params": {"LR": 5, "SVR": 301, "MLP": 1300, "MLP-PM": 13}.get(
                            model, 12
                        ),
                    }
                    if model == "SVR":
                        summary["complexity"] = {"support_vectors": 300}
                    meta = {
                        "git": {"commit": commit, "dirty": False, "status": "", "source": "host"},
                        "config_hash": config_hash,
                        "dataset_sha256": dataset.sha256,
                        "hostname": f"nid{fold.k:04d}",
                        "cpu_model": "AMD EPYC 7763 64-Core Processor",
                        "workers_per_node": 64,
                        "start_time": "2026-09-26T00:00:00+00:00",
                        "end_time": "2026-09-26T01:00:00+00:00",
                        "peak_rss_mb": 500.0,
                    }
                    if kind in ("qnn", "mlp_pm"):
                        n_evals = 60 if seed == 0 else 30
                        early = model in ("QNN-6", "QNN-6u") and fold.k == 2 and seed == 0
                        level = 0.07 if model in ("QNN-6", "QNN-6u") else 0.05
                        level *= 0.25 if model.endswith("u") else 1.0
                        curve_iter, curve_eval = _curve(n_evals, level, early)
                        write_csv(directory / "curve_iter.csv", curve_iter)
                        write_csv(directory / "curve_eval.csv", curve_eval)
                        tpe = RATE.get(model, 0.001) * n
                        summary.update(
                            scipy={
                                "nit": 76 if early else 100,
                                "nfev": n_evals,
                                "message": "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
                                if early
                                else "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
                            },
                            stopped_before_maxiter=early,
                            cache_near_hits=11 if early else 0,
                            best_iteration=29,
                            best_is_initial=False,
                            time_per_evaluation=tpe,
                            optimizer_time=tpe * n_evals,
                            total_training_time=tpe * n_evals + 10.0,
                        )
                        if kind == "qnn" and not (fold.k == 1 and seed == 2 and n == 750):
                            meta["node_load"] = {
                                "workers": 64,
                                "heartbeat_seconds": 30.0,
                                "samples": 100,
                                "busy_mean": 50.0 if fold.k == 0 else 64.0,
                                "busy_min": 40 if fold.k == 0 else 64,
                                "busy_max": 64,
                            }
                    pool = dataset.target[fold.train_pool]
                    target_range = target_scaling(config, model)["target_range"]
                    result = {
                        "status": "complete",
                        "model": model,
                        "kind": kind,
                        "n_train": n,
                        "fold": fold.k,
                        "seed": seed,
                        "scaling": {
                            "target_range": target_range,
                            "y_min_kw": float(pool.min()),
                            "y_max_kw": float(pool.max()),
                        },
                        "hyperparameters": {},
                        "fit_time": 1.0,
                        "model_summary": summary,
                        "metrics": {
                            "val": regression_metrics(dataset.target[fold.validation], val_pred),
                            "test": regression_metrics(dataset.target[fold.test], test_pred),
                        },
                        "metadata": meta,
                    }
                    if batch != "primary":  # absent from the primary batch
                        result.update(protocol=config.protocol, readout=readout_label(target_range))
                    write_json(directory / "result.json", result)
    (root / f"{config_name or batch}.json").write_text(
        json.dumps({"config": str(cfg)}), encoding="utf-8"
    )
    return cfg
