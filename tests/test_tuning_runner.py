"""Tuning and the single-run CLI, on small settings."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from qnnwind import runner
from qnnwind.data import TrainVal, make_split
from qnnwind.io import load_config, read_json
from qnnwind.tuning import params_from_trial, suggest, tune

from conftest import ROOT, poisoned


def small_config(tmp_path, **extra):
    overrides = {
        "paths.results": str(tmp_path / "results"),
        "paths.outputs": str(tmp_path / "outputs"),
        "tuning.n_trials": 3,
        "deep.max_epochs": 3,
        "qnn.optimizer.maxiter": 2,
        "mlp_pm.optimizer.maxiter": 3,
    }
    overrides.update(extra)
    return load_config(ROOT / "configs" / "experiment.yaml", overrides)


def test_suggest_int_or_none():
    space = {"max_depth": {"type": "int_or_none", "low": 2, "high": 30}}
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    seen = set()
    for _ in range(20):
        trial = study.ask()
        params = suggest(trial, space)
        study.tell(trial, 0.0)
        seen.add(params["max_depth"] is None)
        if params["max_depth"] is not None:
            assert 2 <= params["max_depth"] <= 30
        assert params_from_trial(trial.params, space) == params
    assert seen == {True, False}


def test_tuning_is_seeded_resumable_and_blind_to_test(tmp_path, dataset, folds):
    config = small_config(tmp_path)
    fold = folds[3]
    results = []
    for i, ds in enumerate((dataset, poisoned(dataset, fold.test))):
        split = make_split(ds, fold, 200, 0, config["scaling"])
        results.append(tune("kNN", split.trainval, config, tmp_path / f"t{i}", "kNN_test"))
    assert results[0]["params"] == results[1]["params"]
    assert results[0]["best_val_rmse_kw"] == results[1]["best_val_rmse_kw"]
    assert results[0]["n_trials_completed"] == 3
    # Resuming a finished study runs no new trials.
    split = make_split(dataset, fold, 200, 0, config["scaling"])
    again = tune("kNN", split.trainval, config, tmp_path / "t0", "kNN_test")
    assert again["n_trials_total"] == 3 and again["params"] == results[0]["params"]


def test_every_fit_receives_only_trainval(tmp_path, monkeypatch, dataset):
    """Spy on every model class: fit gets a TrainVal (no test rows); test data only via predict."""
    config = small_config(tmp_path, **{"folds.run": [1], "sizes": [60], "seeds": [0]})
    calls = []

    def spy(cls):
        original = cls.fit

        def fit(self, data):
            assert isinstance(data, TrainVal)
            assert np.intersect1d(data.train_rows, data.fold.test).size == 0
            assert np.intersect1d(data.val_rows, data.fold.test).size == 0
            calls.append(type(self).__name__)
            return original(self, data)

        monkeypatch.setattr(cls, "fit", fit)

    from qnnwind.classical import ClassicalModel
    from qnnwind.deep import MLPPM, DeepModel
    from qnnwind.qnn import QNNModel

    for cls in (ClassicalModel, DeepModel, MLPPM, QNNModel):
        spy(cls)
    for model in ("LR", "kNN", "MLP", "MLP-PM", "QNN-6"):
        if runner.is_tuned(config, model):
            runner.execute_tuning(config, model, 60, 1)
        runner.execute_run(config, model, 60, 1, 0)
    assert calls.count("ClassicalModel") == 1 + 1 + 3  # LR, kNN run, 3 kNN trials
    assert calls.count("DeepModel") == 1 + 3
    assert calls.count("MLPPM") == 1 and calls.count("QNNModel") == 1


def test_execute_run_outputs_and_skip(tmp_path, capsys, dataset):
    config = small_config(tmp_path)
    out = runner.execute_run(config, "MLP-PM", 80, 2, 1)
    for name in (
        "result.json",
        "preds_val.csv",
        "preds_test.csv",
        "curve_iter.csv",
        "curve_eval.csv",
    ):
        assert (out / name).is_file()
        assert b"\r\n" not in (out / name).read_bytes()
    result = read_json(out / "result.json")
    assert result["status"] == "complete" and result["fold"] == 2 and result["seed"] == 1
    meta = result["metadata"]
    for key in (
        "git",
        "config_hash",
        "dataset_sha256",
        "packages",
        "hostname",
        "cpu_model",
        "workers_per_node",
        "threads",
        "start_time",
        "end_time",
    ):
        assert key in meta, key
    assert meta["dataset_sha256"] == config["data"]["sha256"]
    assert result["blocks"]["test_ranges"] == [[1488, 2231]]
    preds_test = pd.read_csv(out / "preds_test.csv")
    assert list(preds_test["row"]) == list(range(1488, 2232))
    assert {"actual_kw", "pred_kw", "pred_final_kw"} <= set(preds_test.columns)
    assert result["metrics"]["test"]["rmse"] == pytest.approx(
        np.sqrt(np.mean((preds_test["pred_kw"] - preds_test["actual_kw"]) ** 2))
    )
    stamp = (out / "result.json").stat().st_mtime_ns
    runner.execute_run(config, "MLP-PM", 80, 2, 1)
    assert "[skip]" in capsys.readouterr().out
    assert (out / "result.json").stat().st_mtime_ns == stamp


def test_tuned_model_requires_tuning_result(tmp_path):
    config = small_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.execute_run(config, "SVR", 80, 2, 0)


def test_search_spaces_match_spec(config):
    """The search spaces, exactly."""
    spaces = config["tuning"]["search_spaces"]

    def rng_(kind, low, high, log=False):
        spec = {"type": kind, "low": low, "high": high}
        return spec | {"log": True} if log else spec

    assert spaces["DTR"] == {
        "max_depth": {"type": "int_or_none", "low": 2, "high": 30},
        "min_samples_leaf": rng_("int", 1, 50),
        "min_samples_split": rng_("int", 2, 50),
    }
    assert spaces["kNN"] == {
        "n_neighbors": rng_("int", 1, 50),
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        "p": {"type": "categorical", "choices": [1, 2]},
    }
    assert spaces["SVR"] == {
        "C": rng_("float", 1e-2, 1e3, log=True),
        "epsilon": rng_("float", 1e-3, 1e-1, log=True),
        "gamma": rng_("float", 1e-3, 1e1, log=True),
    }
    assert spaces["XGBoost"]["min_child_weight"] == rng_("float", 1.0, 20.0)
    assert spaces["XGBoost"]["max_depth"] == rng_("int", 2, 10)
    assert spaces["LightGBM"]["num_leaves"] == rng_("int", 4, 128)
    assert spaces["LightGBM"]["min_child_samples"] == rng_("int", 5, 100)
    assert set(spaces) == {
        "kNN",
        "DTR",
        "SVR",
        "XGBoost",
        "LightGBM",
        "MLP",
        "LSTM",
        "GRU",
        "Transformer",
    }


def test_dtr_samples_follow_spec():
    from qnnwind.io import load_config as _load

    space = _load(ROOT / "configs" / "experiment.yaml")["tuning"]["search_spaces"]["DTR"]
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    for _ in range(40):
        trial = study.ask()
        p = suggest(trial, space)
        study.tell(trial, 0.0)
        assert p["max_depth"] is None or (
            isinstance(p["max_depth"], int) and 2 <= p["max_depth"] <= 30
        )
        assert isinstance(p["min_samples_leaf"], int) and 1 <= p["min_samples_leaf"] <= 50
        assert isinstance(p["min_samples_split"], int) and 2 <= p["min_samples_split"] <= 50


def test_completed_run_with_other_config_is_not_reused(tmp_path, dataset):
    config = small_config(tmp_path)
    runner.execute_run(config, "LR", 80, 2, 0)
    changed = small_config(tmp_path, **{"tuning.n_trials": 4})
    with pytest.raises(RuntimeError, match="config hash"):
        runner.execute_run(changed, "LR", 80, 2, 0)
