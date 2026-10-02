"""The QNN target scaling (readout): [-1, 1] for QNN-1..QNN-6, [0, 1] for QNN-1u..QNN-6u."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qnnwind.data import make_split, make_trainval
from qnnwind.io import Config, load_config, read_json, run_dir
from qnnwind.runner import execute_run, target_scaling

from conftest import ROOT

# The configuration hash recorded in every archived run of experiment.yaml.
EXPERIMENT_CONFIG_HASH = "5951445b7460403758f0be4ce3a105ed4a04a176a3b50edf265534dea42c5657"


def test_experiment_config_matches_the_archived_hash():
    config = load_config(ROOT / "configs" / "experiment.yaml")
    assert config.hash == EXPERIMENT_CONFIG_HASH
    assert "target_range" not in config["qnn"]
    assert target_scaling(config, "QNN-5")["target_range"] == [-1.0, 1.0]


def test_target_scaling_only_for_qnn():
    unit = load_config(ROOT / "configs" / "blocked_unit.yaml")
    assert target_scaling(unit, "QNN-5u")["target_range"] == [0.0, 1.0]
    base = load_config(ROOT / "configs" / "experiment.yaml")
    raw = copy.deepcopy(base.raw)
    raw["qnn"]["target_range"] = [0.0, 1.0]  # the optional range for every QNN
    config = Config(raw=raw, source=base.source, root=base.root)
    assert target_scaling(config, "QNN-5")["target_range"] == [0.0, 1.0]
    assert target_scaling(config, "LR")["target_range"] == [-1.0, 1.0]
    assert target_scaling(config, "MLP-PM")["target_range"] == [-1.0, 1.0]
    raw["qnn"]["target_range"] = [1.0, 0.0]
    with pytest.raises(ValueError):
        target_scaling(Config(raw=raw, source=base.source, root=base.root), "QNN-5")


def test_zero_one_readout_run(tmp_path, dataset):
    """A QNN-5u run: scaled training target in [0, 1], recorded range."""
    config = load_config(
        ROOT / "configs" / "blocked_unit.yaml",
        {"qnn.optimizer.maxiter": 1, "paths.results": str(tmp_path)},
    )
    folds = config.folds(dataset.n_rows)
    tv = make_trainval(dataset, folds[5], 60, 0, target_scaling(config, "QNN-5u"))
    assert tv.y_train.min() == pytest.approx(0.0) and tv.y_train.max() == pytest.approx(1.0)
    out = execute_run(config, "QNN-5u", 60, 5, 0)
    result = read_json(out / "result.json")
    assert result["scaling"]["target_range"] == [0.0, 1.0] and result["readout"] == "[0, 1]"
    assert run_dir(tmp_path, "QNN-5u", 60, 5, 0) == out


def test_make_trainval_builds_no_test_data(config, dataset, folds):
    full = make_split(dataset, folds[2], 300, 1, config["scaling"])
    only = make_trainval(dataset, folds[2], 300, 1, config["scaling"])
    assert np.array_equal(full.trainval.X_train, only.X_train)
    assert np.array_equal(full.trainval.y_val, only.y_val)
    empty = make_split(dataset, folds[2], 300, 1, config["scaling"], with_test=False)
    assert empty.X_test.size == 0 and empty.y_test_kw.size == 0 and empty.test_rows.size == 0
