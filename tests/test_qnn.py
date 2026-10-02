"""QNN construction, gradients, loss, callbacks, and selection."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.utils import algorithm_globals

from qnnwind.circuits import build_circuit
from qnnwind.io import load_config
from qnnwind.metrics import BestValidation
from qnnwind.qnn import (
    ForwardCacheTracker,
    QNNModel,
    build_qnn,
    filter_iprint_deprecation,
    initial_point,
)

from conftest import ROOT


@pytest.fixture(scope="module")
def qnn1(config):
    qc = build_circuit("QNN-1", "full", config["qnn"])
    qnn, estimator = build_qnn(qc)
    return qc, qnn, estimator


def test_forward_is_deterministic_and_exact(qnn1):
    qc, qnn, _ = qnn1
    rng = np.random.default_rng(7)
    x, w = rng.random((6, 4)), rng.random(12)
    first, second = qnn.forward(x, w).ravel(), qnn.forward(x, w).ravel()
    assert np.array_equal(first, second)
    for i in range(6):
        state = Statevector(qc.circuit.assign_parameters(np.concatenate([x[i], w])))
        exact = state.expectation_value(SparsePauliOp("ZZZZ")).real
        assert abs(first[i] - exact) < 1e-12


def test_qnn_arguments(qnn1):
    _, qnn, estimator = qnn1
    assert qnn.num_inputs == 4 and qnn.num_weights == 12
    assert qnn.output_shape == (1,)
    assert qnn.estimator is estimator
    assert qnn.gradient._estimator is estimator


def test_param_shift_matches_central_differences(qnn1):
    _, qnn, _ = qnn1
    rng = np.random.default_rng(11)
    x, w = rng.random((5, 4)), rng.uniform(-np.pi, np.pi, 12)
    _, grad = qnn.backward(x, w)
    grad = grad[:, 0, :]
    eps = 1e-5
    numeric = np.empty_like(grad)
    for j in range(12):
        step = np.zeros(12)
        step[j] = eps
        plus = qnn.forward(x, w + step)[:, 0]
        minus = qnn.forward(x, w - step)[:, 0]
        numeric[:, j] = (plus - minus) / (2 * eps)
    assert np.max(np.abs(grad - numeric)) < 1e-6


def test_initial_point_reproduces_algorithm_globals():
    for seed in range(5):
        algorithm_globals.random_seed = seed
        expected = algorithm_globals.random.random(12)
        assert np.array_equal(initial_point(seed, 12), expected)


def test_iprint_filter_is_exact():
    """The project filter hides SciPy's iprint deprecation and nothing else."""
    import scipy.optimize

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with filter_iprint_deprecation():
            scipy.optimize.minimize(
                lambda x: float(x @ x), np.ones(2), method="L-BFGS-B", options={"iprint": -1}
            )
            warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)
    assert [str(w.message) for w in caught] == ["unrelated deprecation"]


@pytest.fixture(scope="module")
def trained(dataset, folds):
    """QNN-2 trained for 4 iterations on 30 rows of fold 5 (seed 1)."""
    from qnnwind.data import make_split

    config = load_config(ROOT / "configs" / "experiment.yaml", {"qnn.optimizer.maxiter": 4})
    split = make_split(dataset, folds[5], 30, 1, config["scaling"])
    model = QNNModel("QNN-2", config["qnn"], seed=1)
    model.fit(split.trainval)  # pytest turns any unfiltered DeprecationWarning into an error
    return model, split


def test_loss_is_sse_over_n(trained):
    model, split = trained
    tv = split.trainval
    x0 = model.summary["initial_params"]
    f0 = model._forward(tv.X_train, x0)
    assert model.curve_eval["train_mse"].iloc[0] == pytest.approx(
        np.sum((f0 - tv.y_train) ** 2) / tv.y_train.size, abs=1e-14
    )
    final = model._forward(tv.X_train, model.final_weights)
    assert model.curve_iter.query("not padded")["train_mse"].iloc[-1] == pytest.approx(
        np.mean((final - tv.y_train) ** 2), abs=1e-14
    )


def test_callback_counts(trained):
    model, _ = trained
    s = model.summary
    assert s["regressor_callback_calls"] == s["scipy"]["nfev"]
    assert s["on_iteration_calls"] == s["scipy"]["nit"]
    assert s["scipy"]["options"] == {
        "maxiter": 4,
        "maxfun": 15000,
        "ftol": 2.220446049250313e-15,
        "gtol": 1e-12,
        "iprint": -1,
        "eps": 1e-08,
    }
    assert s["iterations_without_gradient_at_xk"] == 0


def test_iteration_curve_and_selection(trained):
    model, split = trained
    curve = model.curve_iter
    nit = model.summary["scipy"]["nit"]
    assert list(curve["iteration"]) == list(range(5))
    assert not curve["padded"].iloc[: nit + 1].any()
    assert curve[["train_mse", "val_mse", "grad_norm"]].notna().all().all()
    assert curve["objective_evaluations"].is_monotonic_increasing
    assert curve["circuits_evaluated"].is_monotonic_increasing
    best = int(curve["val_mse"].iloc[: nit + 1].idxmin())  # idxmin returns the first minimum
    assert model.summary["best_iteration"] == best
    assert model.summary["best_is_initial"] == (best == 0)
    tv = split.trainval
    val_best = np.mean((model.predict(tv.X_val) - tv.y_val) ** 2)
    assert val_best == pytest.approx(curve["val_mse"].iloc[best], abs=1e-14)


def test_circuit_counts_per_evaluation(trained):
    model, split = trained
    n = split.trainval.X_train.shape[0]
    s = model.summary
    # Each objective evaluation: 1 forward job (n circuits) and 1 gradient job (24 n circuits).
    assert s["circuits_evaluated"] == s["scipy"]["nfev"] * (n + 24 * n)
    assert s["estimator_calls"] == 2 * s["scipy"]["nfev"]
    assert s["validation_estimator_calls"] == s["scipy"]["nit"] + 1


def test_best_validation_rule():
    best = BestValidation()
    best.offer(0, 0.5, np.zeros(2))
    best.offer(1, 0.5, np.ones(2))  # tie: earlier iteration kept
    assert best.iteration == 0
    best.offer(2, 0.4, np.full(2, 2.0))
    best.offer(3, 0.45, np.full(2, 3.0))
    assert best.iteration == 2 and np.array_equal(best.weights, [2.0, 2.0])
    with pytest.raises(ValueError):
        best.offer(1, 0.1, np.zeros(2))


def test_forward_cache_tracker_mirrors_qiskit_rule():
    tracker = ForwardCacheTracker()
    w = np.array([0.5, 0.25])
    assert tracker.observe(w) == "miss"
    assert tracker.observe(w.copy()) == "exact"
    assert tracker.observe(w + 1e-9) == "near"  # isclose (atol 1e-8), not equal
    assert tracker.observe(w + 2e-9) == "near"  # cache still holds w: near hits do not update it
    assert tracker.observe(w + 1e-3) == "miss"
    assert tracker.observe(w) == "miss"  # compared with the new cached weights
    assert tracker.counts == {"miss": 3, "exact": 1, "near": 2}


def test_cache_logging_is_consistent(trained):
    model, _ = trained
    s = model.summary
    assert s["cache_tracker_consistent"]
    assert s["cache_misses"] == s["training_forward_passes"]
    counts = model.curve_eval["cache"].value_counts().to_dict()
    assert counts.get("near", 0) == s["cache_near_hits"]
    assert sum(counts.values()) == s["scipy"]["nfev"]
    assert s["stopped_before_maxiter"] == (s["scipy"]["nit"] < 4)


def test_time_per_evaluation(trained):
    model, _ = trained
    s = model.summary
    assert s["time_per_evaluation"] == pytest.approx(s["optimizer_time"] / s["scipy"]["nfev"])
