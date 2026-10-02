"""Classical, deep, and parameter-matched models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qnnwind.classical import ClassicalModel
from qnnwind.data import make_split
from qnnwind.deep import MLPPM, DeepModel, FeatureTokenizer, build_network, count_trainable

from conftest import poisoned

CLASSICAL_PARAMS = {
    "LR": {},
    "kNN": {"n_neighbors": 5, "weights": "distance", "p": 2},
    "DTR": {"max_depth": None, "min_samples_leaf": 2, "min_samples_split": 4},
    "SVR": {"C": 10.0, "epsilon": 0.01, "gamma": 1.0},
    "XGBoost": {
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    },
    "LightGBM": {
        "num_leaves": 8,
        "learning_rate": 0.1,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}
DEEP_PARAMS = {
    "MLP": {
        "n_layers": 2,
        "width": 16,
        "activation": "tanh",
        "lr": 1e-2,
        "weight_decay": 1e-5,
        "batch_size": 32,
    },
    "LSTM": {
        "d_token": 8,
        "n_layers": 2,
        "hidden_size": 8,
        "dropout": 0.1,
        "lr": 1e-2,
        "weight_decay": 1e-5,
        "batch_size": 32,
    },
    "GRU": {
        "d_token": 8,
        "n_layers": 1,
        "hidden_size": 8,
        "dropout": 0.1,
        "lr": 1e-2,
        "weight_decay": 1e-5,
        "batch_size": 32,
    },
    "Transformer": {
        "d_token": 16,
        "n_heads": 2,
        "n_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.1,
        "lr": 1e-2,
        "weight_decay": 1e-5,
        "batch_size": 32,
    },
}


@pytest.fixture(scope="module")
def split(config, dataset, folds):
    return make_split(dataset, folds[4], 400, 2, config["scaling"])


@pytest.mark.parametrize("name", list(CLASSICAL_PARAMS))
def test_classical_fit_predict(config, split, name):
    model = ClassicalModel(name, CLASSICAL_PARAMS[name], seed=0, boosting_cfg=config["boosting"])
    model.fit(split.trainval)
    pred = model.predict(split.X_test)
    assert pred.shape == (744,) and np.isfinite(pred).all()
    assert "trainable_params" in model.summary and "complexity" in model.summary
    if name == "LR":
        assert model.summary["trainable_params"] == 5
    if name in ("XGBoost", "LightGBM"):
        assert model.summary["complexity"]["best_iteration"] < config["boosting"]["n_estimators"]


def test_lightgbm_settings(config, split):
    model = ClassicalModel("LightGBM", CLASSICAL_PARAMS["LightGBM"], 0, config["boosting"])
    model.fit(split.trainval)
    params = model.estimator.get_params()
    assert params["subsample_freq"] == 1 and params["verbosity"] == -1 and params["n_jobs"] == 1
    assert params["n_estimators"] == 1000


def test_xgboost_settings(config, split):
    model = ClassicalModel("XGBoost", CLASSICAL_PARAMS["XGBoost"], 0, config["boosting"])
    model.fit(split.trainval)
    params = model.estimator.get_params()
    assert params["early_stopping_rounds"] == 50 and params["n_estimators"] == 1000
    assert params["verbosity"] == 0 and params["n_jobs"] == 1


@pytest.mark.parametrize("name", list(DEEP_PARAMS))
def test_deep_fit_predict_reproducible(config, split, name):
    deep_cfg = {"max_epochs": 6, "patience": 30}
    first = DeepModel(name, DEEP_PARAMS[name], seed=3, deep_cfg=deep_cfg)
    first.fit(split.trainval)
    second = DeepModel(name, DEEP_PARAMS[name], seed=3, deep_cfg=deep_cfg)
    second.fit(split.trainval)
    pred = first.predict(split.X_test)
    assert pred.shape == (744,) and np.isfinite(pred).all()
    assert np.array_equal(pred, second.predict(split.X_test))
    best = first.summary["best_epoch"]
    val = np.mean((first.predict(split.trainval.X_val) - split.trainval.y_val) ** 2)
    assert val == pytest.approx(first.curve["val_mse"].iloc[best - 1], rel=1e-5)


def test_early_stopping_patience(split):
    model = DeepModel(
        "MLP", DEEP_PARAMS["MLP"] | {"lr": 0.05}, 0, {"max_epochs": 500, "patience": 3}
    )
    model.fit(split.trainval)
    s = model.summary
    assert s["stopped_early"] and s["epochs_run"] == s["best_epoch"] + 3


def test_feature_tokenizer():
    tok = FeatureTokenizer(4, 8)
    x = torch.rand(5, 4)
    out = tok(x)
    assert out.shape == (5, 4, 8)
    assert torch.allclose(out[2, 1], x[2, 1] * tok.weight[1] + tok.bias[1])


def test_transformer_has_no_positional_encoding_and_cls_first():
    net = build_network("Transformer", DEEP_PARAMS["Transformer"])
    names = [n for n, _ in net.named_parameters()]
    assert "cls" in names and not any("pos" in n for n in names)
    assert net.encoder.layers[0].self_attn.batch_first


def test_mlp_pm(config, split):
    cfg = config["mlp_pm"] | {"optimizer": config["mlp_pm"]["optimizer"] | {"maxiter": 15}}
    model = MLPPM(cfg, seed=4)
    assert count_trainable(model.network) == 13
    assert all(p.dtype == torch.float64 for p in model.network.parameters())
    assert np.array_equal(model.network[0].bias.detach().numpy(), np.zeros(2))
    model.fit(split.trainval)
    s = model.summary
    assert s["on_iteration_calls"] == s["scipy"]["nit"]
    assert s["objective_evaluations"] == s["scipy"]["nfev"]
    assert s["iterations_without_gradient_at_xk"] == 0
    curve = model.curve_iter
    assert len(curve) == 16
    best = int(curve["val_mse"].iloc[: s["scipy"]["nit"] + 1].idxmin())
    assert s["best_iteration"] == best
    again = MLPPM(cfg, seed=4)
    again.fit(split.trainval)
    assert np.array_equal(again.final_weights, model.final_weights)
    assert not np.array_equal(MLPPM(cfg, seed=5)._get_vector(), MLPPM(cfg, seed=4)._get_vector())


def test_mlp_pm_gradient_matches_finite_differences(split):
    model = MLPPM({"hidden_units": 2, "optimizer": {}}, seed=0)
    X = torch.as_tensor(split.trainval.X_train)
    y = torch.as_tensor(split.trainval.y_train)
    theta = model._get_vector()

    def loss(t):
        model._set_vector(t)
        return float(torch.mean((model.network(X).squeeze(-1) - y) ** 2))

    model._set_vector(theta)
    for p in model._params:
        p.grad = None
    torch.mean((model.network(X).squeeze(-1) - y) ** 2).backward()
    grad = torch.cat([p.grad.reshape(-1) for p in model._params]).numpy()
    eps = 1e-6
    numeric = np.array(
        [(loss(theta + eps * e) - loss(theta - eps * e)) / (2 * eps) for e in np.eye(13)]
    )
    assert np.max(np.abs(grad - numeric)) < 1e-7


@pytest.mark.parametrize("name", ["LR", "kNN", "DTR", "SVR", "XGBoost", "LightGBM"])
def test_poisoned_test_fold_does_not_change_fit(config, dataset, folds, name):
    """Models never see test rows: NaN in the test fold leaves validation predictions unchanged."""
    fold = folds[0]
    preds = []
    for ds in (dataset, poisoned(dataset, fold.test)):
        split = make_split(ds, fold, 300, 1, config["scaling"])
        model = ClassicalModel(name, CLASSICAL_PARAMS[name], 1, config["boosting"])
        model.fit(split.trainval)
        preds.append(model.predict(split.trainval.X_val))
    assert np.array_equal(preds[0], preds[1])
