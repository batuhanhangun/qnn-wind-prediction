"""Metrics and statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from qnnwind.metrics import pad_iteration_curve, regression_metrics
from qnnwind.stats import (
    autocorrelation,
    curve_stability,
    diebold_mariano,
    linear_fit,
    newey_west_lag,
    select_best_qnn,
    stability_scores,
    wilcoxon_signed_rank,
)


def test_regression_metrics():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 1.5, 3.5, -0.5])
    m = regression_metrics(actual, pred)
    err = pred - actual
    assert m["rmse"] == pytest.approx(math.sqrt(np.mean(err**2)))
    assert m["mae"] == pytest.approx(np.mean(np.abs(err)))
    assert m["bias"] == pytest.approx(err.mean())
    assert m["error_std"] == pytest.approx(err.std())
    assert m["rmse"] ** 2 == pytest.approx(m["bias"] ** 2 + m["error_std"] ** 2)
    assert m["r2"] == pytest.approx(1 - np.sum(err**2) / np.sum((actual - 2.5) ** 2))
    assert m["n_negative"] == 1 and m["frac_negative"] == 0.25


def test_pad_iteration_curve():
    curve = pd.DataFrame({"iteration": [0, 1, 2], "train_mse": [3.0, 2.0, 1.0]})
    padded = pad_iteration_curve(curve, 5)
    assert list(padded["iteration"]) == [0, 1, 2, 3, 4, 5]
    assert list(padded["train_mse"]) == [3.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    assert list(padded["padded"]) == [False, False, False, True, True, True]
    assert not pad_iteration_curve(curve, 2)["padded"].any()


def test_autocorrelation():
    x = np.sin(np.linspace(0, 20 * np.pi, 2000))
    acf = autocorrelation(x, [1, 50, 100, 200])  # period 200 samples
    assert acf[1] > 0.99 and abs(acf[50]) < 0.05 and acf[100] < -0.9 and acf[200] > 0.85
    rng = np.random.default_rng(0)
    assert abs(autocorrelation(rng.normal(size=5000), [1])[1]) < 0.05


def test_diebold_mariano_matches_manual_computation():
    rng = np.random.default_rng(3)
    e1 = rng.normal(0, 1.0, 500)
    e2 = rng.normal(0, 1.2, 500) + 0.3 * e1
    res = diebold_mariano(e1, e2, horizon=1, lag=4)
    d = e1**2 - e2**2
    n = d.size
    c = d - d.mean()
    var = c @ c / n + 2 * sum((1 - k / 5) * (c[k:] @ c[:-k]) / n for k in range(1, 5))
    dm = d.mean() / math.sqrt(var / n) * math.sqrt((n - 1) / n)
    assert res["statistic"] == pytest.approx(dm, rel=1e-12)
    assert res["p_value"] == pytest.approx(2 * scipy.stats.t.sf(abs(dm), n - 1), rel=1e-12)
    assert res["statistic"] < 0  # model 1 has smaller errors
    assert newey_west_lag(4464) == 9 and diebold_mariano(e1, e2)["lag"] == newey_west_lag(500)


def test_diebold_mariano_is_antisymmetric():
    rng = np.random.default_rng(4)
    e1, e2 = rng.normal(size=300), rng.normal(size=300)
    a, b = diebold_mariano(e1, e2), diebold_mariano(e2, e1)
    assert a["statistic"] == pytest.approx(-b["statistic"])
    assert a["p_value"] == pytest.approx(b["p_value"])


def test_wilcoxon():
    rng = np.random.default_rng(5)
    a = rng.normal(100, 5, 30)
    b = a + 3 + rng.normal(0, 1, 30)
    res = wilcoxon_signed_rank(a, b)
    assert res["p_value"] < 1e-4
    assert res["p_value"] == pytest.approx(scipy.stats.wilcoxon(a, b).pvalue)


def test_linear_fit():
    x = np.array([750, 1500, 2250, 3000] * 3, dtype=float)
    y = 0.5 * x + 10
    fit = linear_fit(x, y)
    assert fit["a"] == pytest.approx(0.5) and fit["b"] == pytest.approx(10) and fit["r2"] == 1.0


def test_curve_stability_and_scores():
    loss = np.array([5, 4, 3, 2, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 1.2, 0.8, 0.85])
    s = curve_stability(loss, start_eval=10, window=14)
    tail = loss[10:]
    assert s["SD"] == pytest.approx(tail.std())
    assert s["MS"] == pytest.approx(0.3)  # 0.9 -> 1.2
    assert s["FL"] == 0.85 and not s["padded"]
    # Longer curves are cut at the window; shorter ones are padded with their final loss.
    cut = curve_stability(np.concatenate([loss, [9.0, 9.0]]), start_eval=10, window=14)
    assert cut == s
    short = curve_stability(loss[:12], start_eval=10, window=16)
    padded_curve = np.concatenate([loss[:12], np.full(4, loss[11])])
    assert short["padded"] and short["FL"] == loss[11]
    assert short["SD"] == pytest.approx(padded_curve[10:].std())
    assert short["MS"] == pytest.approx(0.3)
    per_run = pd.DataFrame(
        {
            "config": ["A", "A", "B", "B", "C", "C"],
            "n_train": [750, 1500] * 3,
            "SD": [1.0, 3.0, 2.0, 2.0, 0.0, 0.0],
            "MS": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "FL": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "padded": [True, False, False, False, True, True],
        }
    )
    scores = stability_scores(per_run).set_index("config")
    assert scores.loc["A", "SD_norm"] == 1.0 and scores.loc["C", "SD_norm"] == 0.0
    assert scores.loc["A", "MS_norm"] == 0.0 and scores.loc["C", "MS_norm"] == 1.0
    assert (scores["FL_norm"] == 0.0).all()
    assert scores["padded_runs"].to_dict() == {"A": 1, "B": 0, "C": 2}
    assert scores.loc["B", "SC"] == pytest.approx(1.0 + 0.5)
    assert scores["rank"].to_dict() == {"A": 1, "B": 3, "C": 1}  # A and C tie at SC = 1


def test_dm_lag_is_fixed_at_72(config):
    assert config["stats"]["dm"] == {"horizon": 1, "hac_lag": 72}
    rng = np.random.default_rng(6)
    e1, e2 = rng.normal(size=4464), rng.normal(size=4464)
    assert diebold_mariano(e1, e2, lag=config["stats"]["dm"]["hac_lag"])["lag"] == 72


def test_stability_rank_shared_for_identical_configs():
    per_run = pd.DataFrame(
        {
            "config": ["QNN-1", "QNN-5", "QNN-2"],
            "n_train": [750] * 3,
            "SD": [1.0, 1.0, 2.0],
            "MS": [0.5, 0.5, 0.1],
            "FL": [0.2, 0.2, 0.3],
        }
    )
    ranks = stability_scores(per_run).set_index("config")["rank"]
    assert ranks["QNN-1"] == ranks["QNN-5"]


def test_select_best_qnn_breaks_ties_by_fewer_gates():
    gates = {"QNN-1": 40, "QNN-2": 34, "QNN-5": 34, "QNN-3": 36}
    rmse = {"QNN-1": 200.0, "QNN-5": 200.0, "QNN-2": 210.0, "QNN-3": 250.0}
    assert select_best_qnn(rmse, gates) == "QNN-5"
    assert select_best_qnn(rmse | {"QNN-2": 199.0}, gates) == "QNN-2"


def test_stability_config(config):
    assert config["stats"]["stability_window"] == 50
    assert config["stats"]["stability_start_eval"] == 10
