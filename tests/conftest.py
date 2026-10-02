"""Shared fixtures. Thread limits are set before numpy or torch is imported."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from qnnwind.data import Dataset, load_dataset, make_split  # noqa: E402
from qnnwind.folds import folds_from_config  # noqa: E402
from qnnwind.io import Config, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config(ROOT / "configs" / "experiment.yaml")


@pytest.fixture(scope="session")
def dataset(config: Config) -> Dataset:
    path = config.path("data")
    if not path.is_file():
        pytest.skip(f"dataset not available at {path}")
    return load_dataset(path, config["data"])


@pytest.fixture(scope="session")
def folds(config: Config, dataset: Dataset) -> list:
    return folds_from_config(config["folds"], dataset.n_rows)


@pytest.fixture()
def small_split(config: Config, dataset: Dataset, folds: list):
    """Fold 5, N = 40, seed 0: real data, small enough for fast training tests."""
    return make_split(dataset, folds[5], 40, 0, config["scaling"])


def poisoned(dataset: Dataset, rows: np.ndarray) -> Dataset:
    """A copy of ``dataset`` whose ``rows`` (features and target) are NaN."""
    features, target = dataset.features.copy(), dataset.target.copy()
    features[rows] = np.nan
    target[rows] = np.nan
    return Dataset(features, target, dataset.feature_columns, dataset.target_column, dataset.sha256)
