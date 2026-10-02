"""Dataset loading, min-max scaling, and the train/validation container.

Leakage guard: models, tuning, and model selection only ever receive a :class:`TrainVal`,
which cannot hold test rows (its constructor asserts this). Test features reach a model
only through ``predict`` in the runner, after fitting is complete.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qnnwind.folds import Fold, assert_disjoint_from_test, training_subset


@dataclass(frozen=True)
class Dataset:
    """The raw dataset as float64 arrays, in file row order.

    Attributes:
        features: Shape (n_rows, 4), columns in ``feature_columns`` order.
        target: Shape (n_rows,), power in kW.
        feature_columns: Feature names (qubit order q0..q3).
        target_column: Target name.
        sha256: SHA-256 of the file as read.
    """

    features: np.ndarray
    target: np.ndarray
    feature_columns: tuple[str, ...]
    target_column: str
    sha256: str

    @property
    def n_rows(self) -> int:
        return int(self.target.shape[0])


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path, data_cfg: dict) -> Dataset:
    """Load and validate ``data/total_dataset.csv``.

    Asserts the SHA-256, the column names and order, the row count, float64 dtypes, and the
    absence of missing values. The file itself is never modified.
    """
    sha = file_sha256(path)
    if sha != data_cfg["sha256"]:
        raise AssertionError(f"{path}: SHA-256 {sha} != expected {data_cfg['sha256']}")
    frame = pd.read_csv(path, sep=data_cfg["separator"])
    expected_columns = [*data_cfg["feature_columns"], data_cfg["target_column"]]
    if list(frame.columns) != expected_columns:
        raise AssertionError(f"Columns {list(frame.columns)} != {expected_columns}")
    if len(frame) != data_cfg["n_rows"]:
        raise AssertionError(f"{len(frame)} rows != expected {data_cfg['n_rows']}")
    if not all(dtype == np.float64 for dtype in frame.dtypes):
        raise AssertionError(f"Non-float64 columns: {frame.dtypes.to_dict()}")
    if frame.isna().to_numpy().any():
        raise AssertionError("Missing values in dataset")
    return Dataset(
        features=frame[data_cfg["feature_columns"]].to_numpy(dtype=np.float64, copy=True),
        target=frame[data_cfg["target_column"]].to_numpy(dtype=np.float64, copy=True),
        feature_columns=tuple(data_cfg["feature_columns"]),
        target_column=data_cfg["target_column"],
        sha256=sha,
    )


@dataclass(frozen=True)
class MinMax:
    """Column-wise min-max scaling to ``[low, high]``, fitted on training rows only.

    Values outside the fitted range map outside ``[low, high]``; there is no clipping.
    """

    data_min: np.ndarray
    data_max: np.ndarray
    low: float
    high: float

    @classmethod
    def fit(cls, values: np.ndarray, feature_range: tuple[float, float]) -> MinMax:
        values = np.asarray(values, dtype=np.float64)
        data_min, data_max = values.min(axis=0), values.max(axis=0)
        if np.any(data_max <= data_min):
            raise ValueError("Min-max scaling needs max > min in every column")
        return cls(data_min, data_max, float(feature_range[0]), float(feature_range[1]))

    def transform(self, values: np.ndarray) -> np.ndarray:
        unit = (np.asarray(values, dtype=np.float64) - self.data_min) / (
            self.data_max - self.data_min
        )
        return self.low + unit * (self.high - self.low)

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        unit = (np.asarray(scaled, dtype=np.float64) - self.low) / (self.high - self.low)
        return self.data_min + unit * (self.data_max - self.data_min)


@dataclass(frozen=True)
class TrainVal:
    """Scaled training and validation data for one (fold, N, seed): all a model may see.

    Attributes:
        X_train, y_train: Scaled training inputs (n, 4) and target (n,).
        X_val, y_val: Scaled validation inputs and target.
        train_rows, val_rows: File row indices of the two blocks.
        x_scaler, y_scaler: Scalers fitted on the training rows.
        fold: The fold; its test rows are checked, never stored here.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_rows: np.ndarray
    val_rows: np.ndarray
    x_scaler: MinMax
    y_scaler: MinMax
    fold: Fold = field(repr=False)

    def __post_init__(self) -> None:
        assert_disjoint_from_test(self.train_rows, self.fold, "TrainVal.train_rows")
        assert_disjoint_from_test(self.val_rows, self.fold, "TrainVal.val_rows")
        assert np.isin(self.train_rows, self.fold.train_pool).all(), "training rows outside pool"
        assert np.array_equal(self.val_rows, self.fold.validation), "validation rows mismatch"
        assert len(self.X_train) == len(self.y_train) == self.train_rows.size, "train sizes"
        assert len(self.X_val) == len(self.y_val) == self.val_rows.size, "validation sizes"

    def val_rmse_kw(self, y_val_pred_scaled: np.ndarray) -> float:
        """Validation RMSE in kW for scaled validation predictions."""
        pred = self.y_scaler.inverse(np.ravel(y_val_pred_scaled))
        actual = self.y_scaler.inverse(self.y_val)
        return float(np.sqrt(np.mean((pred - actual) ** 2)))


@dataclass(frozen=True)
class Split:
    """Everything the runner needs for one (fold, N, seed).

    ``trainval`` goes to models; ``X_test`` and ``y_test_kw`` stay in the runner and are
    used only for prediction after fitting and for the final metrics.
    """

    trainval: TrainVal
    X_test: np.ndarray
    y_test_kw: np.ndarray
    test_rows: np.ndarray


def fit_scalers(
    dataset: Dataset, rows: np.ndarray, fold: Fold, scaling_cfg: dict
) -> tuple[MinMax, MinMax]:
    """Fit input and target scalers on ``rows`` (the training set only)."""
    assert_disjoint_from_test(rows, fold, "fit_scalers")
    assert np.isin(rows, fold.train_pool).all(), "scalers must be fitted on training rows only"
    x_scaler = MinMax.fit(dataset.features[rows], tuple(scaling_cfg["input_range"]))
    y_scaler = MinMax.fit(dataset.target[rows, None], tuple(scaling_cfg["target_range"]))
    return x_scaler, y_scaler


def make_trainval(
    dataset: Dataset, fold: Fold, n_train: int, seed: int, scaling_cfg: dict
) -> TrainVal:
    """Only the scaled training and validation data for (fold, N, seed): no test-fold array is
    built (for work that must not touch the test fold at all)."""
    return make_split(dataset, fold, n_train, seed, scaling_cfg, with_test=False).trainval


def make_split(
    dataset: Dataset,
    fold: Fold,
    n_train: int,
    seed: int,
    scaling_cfg: dict,
    with_test: bool = True,
) -> Split:
    """Build the scaled data for (fold, N, seed).

    With ``with_test=False`` the test fields are empty arrays and the test rows are never read.
    """
    train_rows = training_subset(fold, n_train, seed)
    val_rows = fold.validation
    x_scaler, y_scaler = fit_scalers(dataset, train_rows, fold, scaling_cfg)

    def scale_y(rows: np.ndarray) -> np.ndarray:
        return y_scaler.transform(dataset.target[rows, None])[:, 0]

    trainval = TrainVal(
        X_train=x_scaler.transform(dataset.features[train_rows]),
        y_train=scale_y(train_rows),
        X_val=x_scaler.transform(dataset.features[val_rows]),
        y_val=scale_y(val_rows),
        train_rows=train_rows,
        val_rows=val_rows,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        fold=fold,
    )
    if not with_test:
        empty = np.empty(0, dtype=np.int64)
        return Split(trainval, np.empty((0, dataset.features.shape[1])), np.empty(0), empty)
    return Split(
        trainval=trainval,
        X_test=x_scaler.transform(dataset.features[fold.test]),
        y_test_kw=dataset.target[fold.test].copy(),
        test_rows=fold.test,
    )
