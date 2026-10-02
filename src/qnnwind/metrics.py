"""Prediction metrics and the best-validation selection rule."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def regression_metrics(actual_kw: np.ndarray, predicted_kw: np.ndarray) -> dict[str, float]:
    """R², RMSE, MAE, bias, error std, and negative-prediction counts, all in kW.

    ``bias`` is mean(prediction - actual). ``error_std`` is the population standard deviation
    (ddof=0) of prediction - actual, so that RMSE² = bias² + error_std².
    """
    actual = np.asarray(actual_kw, dtype=np.float64).ravel()
    predicted = np.asarray(predicted_kw, dtype=np.float64).ravel()
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    error = predicted - actual
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    n_negative = int(np.sum(predicted < 0))
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "error_std": float(np.std(error, ddof=0)),
        "n_negative": n_negative,
        "frac_negative": n_negative / actual.size,
        "n": int(actual.size),
    }


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean squared error between two arrays of the same size."""
    return float(np.mean((np.ravel(predicted) - np.ravel(actual)) ** 2))


@dataclass
class BestValidation:
    """Keeps the candidate with the lowest validation MSE; ties go to the earliest.

    Candidates must be offered in increasing iteration order, starting with iteration 0
    (the initial weights).
    """

    iteration: int = -1
    val_mse: float = np.inf
    weights: np.ndarray = field(default_factory=lambda: np.empty(0))

    def offer(self, iteration: int, val_mse: float, weights: np.ndarray) -> None:
        if iteration <= self.iteration:
            raise ValueError("Candidates must arrive in increasing iteration order")
        if not np.isfinite(val_mse):
            raise ValueError(f"Non-finite validation MSE at iteration {iteration}")
        if val_mse < self.val_mse:  # strict: ties keep the earlier iteration
            self.iteration = iteration
            self.val_mse = val_mse
            self.weights = np.array(weights, dtype=np.float64, copy=True)


def pad_iteration_curve(curve: pd.DataFrame, maxiter: int) -> pd.DataFrame:
    """Pad a per-iteration curve (rows 0..nit) to rows 0..maxiter with its last row.

    Adds a boolean ``padded`` column (an early stop is padded with the last value; the true
    ``nit`` is logged separately).
    """
    curve = curve.copy()
    curve["padded"] = False
    missing = maxiter + 1 - len(curve)
    if missing > 0:
        filler = pd.DataFrame([curve.iloc[-1].to_dict()] * missing)
        filler["iteration"] = np.arange(len(curve), maxiter + 1)
        filler["padded"] = True
        curve = pd.concat([curve, filler], ignore_index=True)
    return curve
