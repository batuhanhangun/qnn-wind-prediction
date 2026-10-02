"""Loading, integrity, scaling, and the test-fold leakage guards."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qnnwind.data import MinMax, TrainVal, fit_scalers, load_dataset, make_split

from conftest import poisoned


def test_dataset_integrity(dataset, config):
    assert dataset.sha256 == config["data"]["sha256"]
    assert dataset.features.shape == (4464, 4) and dataset.target.shape == (4464,)
    assert dataset.feature_columns == ("Temperature", "Pressure", "Theta", "Velocity")
    assert dataset.target_column == "Power"
    assert dataset.features.dtype == np.float64
    # Known properties of the dataset.
    assert dataset.target.mean() == pytest.approx(666.60, abs=0.005)
    assert dataset.target.min() == 2.24 and dataset.target.max() == 2033.12
    assert dataset.features[:, 3].mean() == pytest.approx(8.65, abs=0.005)
    assert (dataset.target > 0).all() and (dataset.features[:, 3] > 0).all()


def test_wrong_checksum_is_rejected(config, tmp_path):
    other = tmp_path / "other.csv"  # any file whose checksum differs from the expected one
    other.write_text("Temperature;Pressure;Theta;Velocity;Power\n1;2;3;4;5\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA-256"):
        load_dataset(other, config["data"])


def test_minmax_ranges_and_no_clipping():
    values = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    unit = MinMax.fit(values, (0.0, 1.0))
    sym = MinMax.fit(values, (-1.0, 1.0))
    assert np.allclose(unit.transform(values), [[0, 0], [0.5, 0.5], [1, 1]])
    assert np.allclose(sym.transform(values), [[-1, -1], [0, 0], [1, 1]])
    outside = np.array([[20.0, 0.0]])
    assert np.allclose(unit.transform(outside), [[2.0, -0.5]])  # no clipping
    assert np.allclose(sym.inverse(sym.transform(outside)), outside)


def test_scalers_fitted_on_training_rows_only(dataset, folds, config):
    split = make_split(dataset, folds[5], 750, 3, config["scaling"])
    tv = split.trainval
    assert np.allclose(tv.X_train.min(axis=0), 0.0) and np.allclose(tv.X_train.max(axis=0), 1.0)
    assert np.isclose(tv.y_train.min(), -1.0) and np.isclose(tv.y_train.max(), 1.0)
    train_features = dataset.features[tv.train_rows]
    assert np.array_equal(tv.x_scaler.data_min, train_features.min(axis=0))
    assert np.array_equal(tv.x_scaler.data_max, train_features.max(axis=0))
    assert np.allclose(tv.y_scaler.inverse(tv.y_val), dataset.target[folds[5].validation])
    assert np.array_equal(split.y_test_kw, dataset.target[folds[5].test])
    # Fold 5's test block reaches 21.07 m/s, above the training maximum: allowed, unclipped.
    assert split.X_test[:, 3].max() > 1.0


def test_fit_scalers_rejects_test_rows(dataset, folds, config):
    fold = folds[1]
    rows = np.concatenate([fold.train_pool[:50], fold.test[:1]])
    with pytest.raises(AssertionError):
        fit_scalers(dataset, rows, fold, config["scaling"])
    with pytest.raises(AssertionError):
        fit_scalers(dataset, fold.validation, fold, config["scaling"])


def test_trainval_rejects_test_rows(small_split):
    tv = small_split.trainval
    leaky_rows = tv.train_rows.copy()
    leaky_rows[0] = tv.fold.test[0]
    with pytest.raises(AssertionError):
        dataclasses.replace(tv, train_rows=leaky_rows)
    with pytest.raises(AssertionError):
        dataclasses.replace(tv, val_rows=tv.fold.test[: tv.val_rows.size])


def test_poisoned_test_fold_does_not_change_trainval(dataset, folds, config):
    """Scaling and splitting never read test rows: NaN there changes nothing but X_test."""
    fold = folds[2]
    clean = make_split(dataset, fold, 1500, 4, config["scaling"])
    dirty = make_split(poisoned(dataset, fold.test), fold, 1500, 4, config["scaling"])
    for name in ("X_train", "y_train", "X_val", "y_val", "train_rows"):
        assert np.array_equal(getattr(clean.trainval, name), getattr(dirty.trainval, name)), name
    assert np.isnan(dirty.X_test).all() and not np.isnan(clean.X_test).any()


def test_trainval_has_no_test_attribute(small_split):
    names = {f.name for f in dataclasses.fields(TrainVal)}
    assert not any("test" in name for name in names)
    assert isinstance(small_split.trainval, TrainVal)
