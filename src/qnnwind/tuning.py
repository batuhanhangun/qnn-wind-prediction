"""Optuna TPE tuning per (model, N, fold) on the seed-0 training subset.

Optuna 5.0.0 API used (checked with ``inspect``): ``TPESampler(seed=...)`` (multivariate TPE
is always on in 5.x; the other constructor arguments are left at their defaults),
``create_study(storage=..., sampler=..., study_name=..., direction=..., load_if_exists=...)``,
and ``Study.optimize(func, n_trials=...)``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import optuna

from qnnwind.classical import CLASSICAL_MODELS, ClassicalModel
from qnnwind.data import TrainVal
from qnnwind.deep import DEEP_MODELS, DeepModel
from qnnwind.io import Config, sqlite_url, utc_now, write_json

TUNED_MODELS = tuple(m for m in (*CLASSICAL_MODELS, *DEEP_MODELS) if m != "LR")


def suggest(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample one configuration from a config search space.

    ``int_or_none`` parameters (DTR ``max_depth`` in {None, 2..30}) are sampled as a boolean
    ``<name>_limited`` followed, if true, by the integer.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        for bound in ("low", "high"):
            # PyYAML reads e.g. `1.0e3` (no exponent sign) as a string; never coerce silently.
            if bound in spec and not isinstance(spec[bound], int | float):
                raise TypeError(f"{name}.{bound} = {spec[bound]!r} is not a number")
        if kind == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif kind == "int_or_none":
            limited = trial.suggest_categorical(f"{name}_limited", [False, True])
            params[name] = trial.suggest_int(name, spec["low"], spec["high"]) if limited else None
        else:
            raise ValueError(f"Unknown search-space type {kind!r} for {name}")
    return params


def params_from_trial(trial_params: dict[str, Any], space: dict[str, dict[str, Any]]) -> dict:
    """Rebuild model parameters from a stored trial's raw Optuna parameters."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if spec["type"] == "int_or_none":
            params[name] = trial_params[name] if trial_params[f"{name}_limited"] else None
        else:
            params[name] = trial_params[name]
    return params


def make_model(name: str, params: dict[str, Any], seed: int, config: Config) -> Any:
    """A classical or deep model with the given hyperparameters."""
    if name in CLASSICAL_MODELS:
        return ClassicalModel(name, params, seed, config["boosting"])
    if name in DEEP_MODELS:
        return DeepModel(name, params, seed, config["deep"])
    raise ValueError(f"{name!r} is not a tunable model")


def tune(
    name: str, data: TrainVal, config: Config, out_dir: Path, study_name: str
) -> dict[str, Any]:
    """Run (or resume) the Optuna study for one (model, N, fold); write best_params.json.

    The objective is the validation RMSE in kW. Only ``data`` (training and validation
    blocks) is used; the test fold never reaches this function.

    Returns:
        The content written to ``best_params.json``.
    """
    tuning_cfg = config["tuning"]
    space = tuning_cfg["search_spaces"][name]
    n_trials = int(tuning_cfg["n_trials"])
    fit_seed = int(tuning_cfg["subset_seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        storage=sqlite_url(out_dir / "study.db"),
        sampler=optuna.samplers.TPESampler(seed=int(tuning_cfg["sampler_seed"])),
        study_name=study_name,
        direction="minimize",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        model = make_model(name, suggest(trial, space), fit_seed, config)
        model.fit(data)
        return data.val_rmse_kw(model.predict(data.X_val))

    def completed_trials() -> list[optuna.trial.FrozenTrial]:
        return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Resuming keeps completed trials and runs only the missing ones. An interrupted and
    # resumed study is not guaranteed to sample the same sequence as an uninterrupted one.
    remaining = n_trials - len(completed_trials())
    t_start = time.perf_counter()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    completed = completed_trials()
    if len(completed) < n_trials:
        raise RuntimeError(f"{name}: only {len(completed)} of {n_trials} trials completed")

    best = study.best_trial
    content = {
        "model": name,
        "study_name": study_name,
        "params": params_from_trial(best.params, space),
        "optuna_params": best.params,
        "best_trial": best.number,
        "best_val_rmse_kw": best.value,
        "n_trials_completed": len(completed),
        "n_trials_total": len(study.trials),
        "sampler_seed": int(tuning_cfg["sampler_seed"]),
        "fit_seed": fit_seed,
        "tuning_time_this_session": time.perf_counter() - t_start,
        "finished": utc_now(),
        "config_hash": config.hash,
        "optuna_version": optuna.__version__,
    }
    write_json(out_dir / "best_params.json", content)
    return content
