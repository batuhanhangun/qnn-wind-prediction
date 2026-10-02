"""Statistics: autocorrelation, Diebold-Mariano, Wilcoxon, timing fits, stability."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.stats


def autocorrelation(x: np.ndarray, lags: Sequence[int]) -> dict[int, float]:
    """Sample autocorrelation r_k = sum (x_t - m)(x_{t+k} - m) / sum (x_t - m)^2."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    return {int(k): float(np.dot(centered[:-k], centered[k:]) / denom) for k in lags}


def newey_west_lag(n: int) -> int:
    """Automatic Newey-West (1994) bandwidth floor(4 * (n / 100) ** (2 / 9))."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(
    errors_a: np.ndarray, errors_b: np.ndarray, horizon: int = 1, lag: int | str = "auto"
) -> dict[str, float]:
    """Diebold-Mariano test on squared errors with HLN correction and Newey-West variance.

    The loss differential is d_t = e_a,t² - e_b,t² in the given (row) order. Its long-run
    variance uses Bartlett weights 1 - k/(L+1) for k = 1..L. The statistic is multiplied by
    the Harvey-Leybourne-Newbold factor sqrt((T + 1 - 2h + h(h-1)/T) / T) and compared with a
    Student t distribution with T - 1 degrees of freedom (two-sided). Negative values mean
    model A has lower squared error.
    """
    d = np.asarray(errors_a, dtype=np.float64) ** 2 - np.asarray(errors_b, dtype=np.float64) ** 2
    n = d.size
    lag_value = newey_west_lag(n) if lag == "auto" else int(lag)
    centered = d - d.mean()
    variance = float(np.dot(centered, centered)) / n
    for k in range(1, lag_value + 1):
        gamma = float(np.dot(centered[k:], centered[:-k])) / n
        variance += 2.0 * (1.0 - k / (lag_value + 1.0)) * gamma
    if variance <= 0:
        raise ValueError("Non-positive HAC variance of the loss differential")
    dm = d.mean() / math.sqrt(variance / n)
    h = horizon
    hln = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p_value = 2.0 * scipy.stats.t.sf(abs(hln), df=n - 1)
    return {"statistic": float(hln), "p_value": float(p_value), "lag": lag_value, "n": n}


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values for one family of tests.

    Missing (NaN) p-values are not part of the family and stay NaN. With m available tests
    sorted ascending, adj_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)).
    """
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p.shape, np.nan)
    available = np.flatnonzero(~np.isnan(p))
    m = available.size
    if m == 0:
        return adjusted
    order = available[np.argsort(p[available], kind="stable")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def wilcoxon_signed_rank(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on paired values (SciPy defaults)."""
    result = scipy.stats.wilcoxon(values_a, values_b)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares fit y = a * x + b with the R² of the fit."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - (a * x + b)
    r2 = 1.0 - float(np.sum(residual**2)) / float(np.sum((y - y.mean()) ** 2))
    return {"a": float(a), "b": float(b), "r2": r2}


def curve_stability(loss: np.ndarray, start_eval: int, window: int) -> dict[str, float | bool]:
    """SD, MS, and FL of one per-evaluation training-loss curve over evaluations 1..window.

    A curve shorter than ``window`` belongs to a run that converged early and is
    padded with its final loss; ``padded`` reports this. Longer curves are cut at ``window``.
    With ``start_eval`` = 10, SD and MS use evaluations 11..E (0-based 10..E-1); a spike is
    max(0, loss[i] - loss[i-1]). FL is the loss at evaluation E.
    """
    loss = np.asarray(loss, dtype=np.float64)
    if loss.size == 0:
        raise ValueError("Empty loss curve")
    if not 0 < start_eval < window:
        raise ValueError(f"Need 0 < start_eval ({start_eval}) < window ({window})")
    padded = loss.size < window
    curve = np.concatenate([loss, np.full(max(0, window - loss.size), loss[-1])])[:window]
    tail = curve[start_eval:]
    spikes = np.maximum(0.0, curve[start_eval:] - curve[start_eval - 1 : -1])
    return {
        "SD": float(np.std(tail)),
        "MS": float(spikes.max()),
        "FL": float(curve[-1]),
        "padded": bool(padded),
    }


def stability_scores(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run SD/MS/FL into the stability score SC per configuration.

    Args:
        per_run: Columns ``config``, ``n_train``, ``SD``, ``MS``, ``FL`` (one row per run).

    Returns:
        One row per configuration with raw and min-max normalized metrics, ``SC``,
        ``rank`` (1 = most stable; equal scores, e.g. QNN-1 and QNN-5, share a rank), and
        ``padded_runs`` when ``per_run`` has a boolean ``padded`` column.
    """
    by_size = per_run.groupby(["config", "n_train"])[["SD", "MS", "FL"]].mean()
    by_config = by_size.groupby("config").mean()
    if "padded" in per_run:
        by_config["padded_runs"] = per_run.groupby("config")["padded"].sum().astype(int)
    for col in ("SD", "MS", "FL"):
        low, high = by_config[col].min(), by_config[col].max()
        span = high - low
        by_config[f"{col}_norm"] = (by_config[col] - low) / span if span > 0 else 0.0
    by_config["SC"] = by_config[["SD_norm", "MS_norm", "FL_norm"]].sum(axis=1)
    by_config["rank"] = by_config["SC"].rank(method="min").astype(int)
    return by_config.reset_index()


def select_best_qnn(mean_val_rmse: dict[str, float], total_gates: dict[str, int]) -> str:
    """Best QNN configuration at one N.

    Lowest mean validation RMSE; exact ties (QNN-1 and QNN-5 always tie) go to the
    configuration with fewer total gates, then to the name for determinism.
    """
    return min(mean_val_rmse, key=lambda name: (mean_val_rmse[name], total_gates[name], name))
