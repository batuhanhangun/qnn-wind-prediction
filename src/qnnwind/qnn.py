"""QNN construction and training with Qiskit 2.3.0 and Qiskit Machine Learning 0.9.0.

Instrumentation, without changing what Qiskit ML does:

* :class:`CountingStatevectorEstimator` subclasses ``StatevectorEstimator`` only to count
  estimator calls and circuit evaluations; the simulation code is inherited unchanged.
* :func:`_observe_scipy_minimize` temporarily wraps the ``scipy.optimize.minimize`` reference
  that Qiskit ML's ``SciPyOptimizer`` calls. The wrapper forwards every argument unchanged; it
  records the options actually sent to SciPy, the gradient at each evaluated point (for the
  gradient norm), and SciPy's full ``OptimizeResult`` (Qiskit ML drops ``message``).
"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import qiskit_machine_learning.optimizers.scipy_optimizer as _qml_scipy_optimizer
import scipy.optimize
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import L_BFGS_B

from qnnwind.circuits import QNNCircuit, build_circuit, circuit_for
from qnnwind.data import TrainVal
from qnnwind.metrics import BestValidation, mse, pad_iteration_curve

# The one tolerated warning: SciPy 1.17 deprecates `iprint`, which Qiskit ML
# 0.9.0 always passes to L-BFGS-B. Only this exact message is filtered.
IPRINT_DEPRECATION_MESSAGE = re.escape(
    "scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated"
)


@contextmanager
def filter_iprint_deprecation() -> Iterator[None]:
    """Ignore SciPy's L-BFGS-B ``iprint`` DeprecationWarning, and nothing else."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=IPRINT_DEPRECATION_MESSAGE, category=DeprecationWarning
        )
        yield


class CountingStatevectorEstimator(StatevectorEstimator):
    """``StatevectorEstimator`` that counts ``run`` calls and simulated circuits.

    A "circuit evaluation" is one bound circuit simulated to a statevector, i.e. one element
    of a PUB's broadcast shape (``StatevectorEstimator._run_pub`` loops over exactly these).
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.circuits = 0
        # Forward passes submit one PUB (one observable); gradient jobs submit 24 per sample.
        self.single_pub_calls = 0

    def run(self, pubs, *, precision=None):  # noqa: ANN001 - signature of the base class
        pubs = list(pubs)
        self.calls += 1
        self.single_pub_calls += len(pubs) == 1
        return super().run(pubs, precision=precision)

    def _run(self, pubs):  # noqa: ANN001 - signature of the base class
        self.circuits += sum(int(np.prod(pub.shape, dtype=np.int64)) for pub in pubs)
        return super()._run(pubs)

    def snapshot(self) -> tuple[int, int]:
        return self.calls, self.circuits


class ForwardCacheTracker:
    """Mirror of Qiskit ML 0.9.0's forward cache in ``ObjectiveFunction._neural_network_forward``.

    Qiskit ML recomputes the forward pass unless ``np.all(np.isclose(weights, cached))`` with
    NumPy's default tolerances, and only a recomputation replaces the cached weights. Each
    objective evaluation is classified as a ``miss`` (forward pass computed), an ``exact`` hit
    (identical weights), or a ``near`` hit (isclose but not identical: the objective returns
    the loss of slightly different weights). The library behaviour is not changed.
    """

    def __init__(self) -> None:
        self.cached: np.ndarray | None = None
        self.counts = {"miss": 0, "exact": 0, "near": 0}

    def observe(self, weights: np.ndarray) -> str:
        weights = np.asarray(weights)
        if self.cached is None or not np.all(np.isclose(weights, self.cached)):
            self.cached = np.array(weights, copy=True)
            kind = "miss"
        elif np.array_equal(weights, self.cached):
            kind = "exact"
        else:
            kind = "near"
        self.counts[kind] += 1
        return kind


def build_qnn(qc: QNNCircuit) -> tuple[EstimatorQNN, CountingStatevectorEstimator]:
    """The ``EstimatorQNN``, with every relevant argument explicit."""
    estimator = CountingStatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=qc.circuit,
        estimator=estimator,
        observables=SparsePauliOp("ZZZZ"),
        input_params=list(qc.feature_map.parameters),
        weight_params=list(qc.ansatz.parameters),
        gradient=ParamShiftEstimatorGradient(estimator),
        input_gradients=False,
        default_precision=0.0,  # REQUIRED: the default 0.015625 adds noise to forward passes
    )
    return qnn, estimator


def filler_simulation(
    qnn_cfg: dict, model: str, should_stop: Callable[[], bool], batch: int, seed: int = 0
) -> int:
    """Load-keeper filler: QNN circuit simulation that writes no results.

    Repeats a forward and a parameter-shift backward pass of ``model`` on ``batch`` random
    samples (``batch * (1 + 24)`` circuits per chunk for 12 weights, the same work as the
    calibration probe) until ``should_stop()`` is true; checked between chunks.

    Returns:
        The number of chunks run.
    """
    qc = build_circuit(model, qnn_cfg["entanglement"][model], qnn_cfg)
    qnn, _ = build_qnn(qc)
    rng = np.random.default_rng(seed)
    chunks = 0
    while not should_stop():
        x, w = rng.random((batch, qnn.num_inputs)), rng.random(qnn.num_weights)
        qnn.forward(x, w)
        qnn.backward(x, w)
        chunks += 1
    return chunks


def initial_point(seed: int, num_weights: int) -> np.ndarray:
    """``np.random.default_rng(seed).random(num_weights)``."""
    return np.random.default_rng(seed).random(num_weights)


@contextmanager
def _observe_scipy_minimize(record: dict[str, Any]) -> Iterator[None]:
    """Wrap the ``minimize`` that Qiskit ML's ``SciPyOptimizer`` calls; see module docstring."""
    original = _qml_scipy_optimizer.minimize
    if original is not scipy.optimize.minimize:
        raise RuntimeError("Qiskit ML no longer calls scipy.optimize.minimize directly")

    def observed_minimize(*args: Any, **kwargs: Any) -> scipy.optimize.OptimizeResult:
        if args:
            raise RuntimeError("Qiskit ML passed positional arguments to minimize")
        record["method"] = kwargs.get("method")
        record["options"] = dict(kwargs.get("options") or {})
        record["other_kwargs"] = sorted(
            k for k in kwargs if k not in {"fun", "x0", "method", "jac", "bounds", "options"}
        )
        jac = kwargs["jac"]
        if jac is None:
            raise RuntimeError("Qiskit ML did not pass an analytic gradient")

        def observed_jac(x: np.ndarray) -> np.ndarray:
            gradient = jac(x)
            record["gradients"].append((np.array(x, copy=True), np.array(gradient, copy=True)))
            record["on_gradient"]()
            return gradient

        kwargs["jac"] = observed_jac
        result = original(**kwargs)
        record["result"] = result
        return result

    record["gradients"] = []
    _qml_scipy_optimizer.minimize = observed_minimize
    try:
        yield
    finally:
        _qml_scipy_optimizer.minimize = original


class QNNModel:
    """One QNN configuration trained with ``NeuralNetworkRegressor`` and L-BFGS-B.

    Args:
        name: Configuration name, e.g. ``"QNN-1"``.
        qnn_cfg: The ``qnn`` config section.
        seed: Run seed (initial point).
    """

    def __init__(self, name: str, qnn_cfg: dict, seed: int) -> None:
        self.name = name
        self.cfg = qnn_cfg
        self.seed = seed
        self.qc = circuit_for(qnn_cfg, name)  # readout variants use the base circuit
        self.qnn, self.estimator = build_qnn(self.qc)
        self.best = BestValidation()
        self.final_weights: np.ndarray | None = None
        self.curve_iter: pd.DataFrame | None = None
        self.curve_eval: pd.DataFrame | None = None
        self.summary: dict[str, Any] = {}

    def _forward(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.asarray(self.qnn.forward(X, weights), dtype=np.float64).ravel()

    def fit(self, data: TrainVal) -> None:
        """Train on ``data`` and select the best-validation weights (iteration 0 included)."""
        maxiter = int(self.cfg["optimizer"]["maxiter"])
        x0 = initial_point(self.seed, self.qnn.num_weights)
        estimator = self.estimator

        eval_rows: list[dict[str, Any]] = []
        iter_rows: list[dict[str, Any]] = []
        timers = {"val": 0.0}
        record: dict[str, Any] = {}
        cache = ForwardCacheTracker()

        def optimizer_time(now: float) -> float:
            return now - t_start - timers["val"]

        def validation_mse(weights: np.ndarray) -> float:
            calls, circuits = estimator.snapshot()
            t0 = time.perf_counter()
            value = mse(data.y_val, self._forward(data.X_val, weights))
            timers["val"] += time.perf_counter() - t0
            # Validation work is excluded from the optimizer's estimator counters.
            timers["val_calls"] = timers.get("val_calls", 0) + estimator.calls - calls
            timers["val_circuits"] = timers.get("val_circuits", 0) + estimator.circuits - circuits
            return value

        def optimizer_counts() -> tuple[int, int]:
            return (
                estimator.calls - timers.get("val_calls", 0),
                estimator.circuits - timers.get("val_circuits", 0),
            )

        def on_evaluation(weights: np.ndarray, objective_value: float) -> None:
            now = time.perf_counter()
            eval_rows.append(
                {
                    "evaluation": len(eval_rows) + 1,
                    "train_mse": float(objective_value),
                    "wall_time": now - t_start,
                    "optimizer_time": optimizer_time(now),
                    "cache": cache.observe(weights),
                }
            )
            record["last_eval_x"] = np.array(weights, copy=True)

        def on_first_gradient() -> None:
            # Cumulative cost at the moment iteration 0 (x0) has its objective and gradient.
            if "iter0" not in record:
                calls, circuits = optimizer_counts()
                record["iter0"] = {
                    "objective_evaluations": len(eval_rows),
                    "estimator_calls": calls,
                    "circuits_evaluated": circuits,
                    "optimizer_time": optimizer_time(time.perf_counter()),
                }

        record["on_gradient"] = on_first_gradient

        def on_iteration(xk: np.ndarray) -> None:
            now = time.perf_counter()
            calls, circuits = optimizer_counts()
            grad_x, grad = record["gradients"][-1]
            grad_matches = bool(np.array_equal(grad_x, xk))
            train_value = eval_rows[-1]["train_mse"]
            train_matches = bool(np.array_equal(record["last_eval_x"], xk))
            row = {
                "iteration": len(iter_rows),
                "train_mse": train_value if train_matches else np.nan,
                "val_mse": np.nan,
                "grad_norm": float(np.linalg.norm(grad)) if grad_matches else np.nan,
                "objective_evaluations": len(eval_rows),
                "estimator_calls": calls,
                "circuits_evaluated": circuits,
                "optimizer_time": optimizer_time(now),
            }
            row["val_mse"] = validation_mse(xk)
            row["validation_time"] = timers["val"]
            iter_rows.append(row)
            self.best.offer(row["iteration"], row["val_mse"], xk)

        # Iteration 0: the initial weights, evaluated on validation before optimization.
        t_start = time.perf_counter()
        val0 = validation_mse(x0)
        iter_rows.append({"iteration": 0, "val_mse": val0, "validation_time": timers["val"]})
        self.best.offer(0, val0, x0)

        optimizer = L_BFGS_B(
            maxiter=maxiter,
            options={"gtol": float(self.cfg["optimizer"]["gtol"])},
            callback=on_iteration,
        )
        regressor = NeuralNetworkRegressor(
            neural_network=self.qnn,
            loss="squared_error",
            optimizer=optimizer,
            initial_point=x0,
            callback=on_evaluation,
        )
        with filter_iprint_deprecation(), _observe_scipy_minimize(record):
            regressor.fit(data.X_train, data.y_train)
        total_time = time.perf_counter() - t_start

        result = record["result"]
        self.final_weights = np.array(result.x, dtype=np.float64, copy=True)

        # Complete the iteration-0 row from the first objective evaluation (at x0).
        first_x, first_grad = record["gradients"][0]
        if not np.array_equal(first_x, x0):
            raise RuntimeError("The first gradient evaluation was not at the initial point")
        iter_rows[0].update(
            train_mse=eval_rows[0]["train_mse"],
            grad_norm=float(np.linalg.norm(first_grad)),
            **record["iter0"],
        )

        curve_iter = pd.DataFrame(iter_rows)
        n_grad_mismatch = int(curve_iter["grad_norm"].isna().sum())
        self.curve_iter = pad_iteration_curve(curve_iter, maxiter)
        self.curve_eval = pd.DataFrame(eval_rows)

        opt_calls, opt_circuits = optimizer_counts()
        opt_time = total_time - timers["val"]
        # Every validation call is a single-PUB forward pass; the rest are training forwards.
        training_forwards = estimator.single_pub_calls - int(timers.get("val_calls", 0))
        nit = int(result.nit)
        self.summary = {
            "model": self.name,
            "entanglement": self.qc.entanglement,
            "seed": self.seed,
            "n_train": int(data.X_train.shape[0]),
            "num_weights": int(self.qnn.num_weights),
            "trainable_params": int(self.qnn.num_weights),
            "scipy": {
                "method": record["method"],
                "options": record["options"],
                "other_kwargs": record["other_kwargs"],
                "nit": nit,
                "nfev": int(result.nfev),
                "njev": int(result.njev),
                "status": int(result.status),
                "success": bool(result.success),
                "message": str(result.message),
            },
            "stopped_before_maxiter": nit < maxiter,
            "regressor_callback_calls": len(eval_rows),
            "cache_near_hits": cache.counts["near"],
            "cache_exact_hits": cache.counts["exact"],
            "cache_misses": cache.counts["miss"],
            "training_forward_passes": training_forwards,
            # The tracker mirrors Qiskit ML's cache iff its misses equal the forward passes run.
            "cache_tracker_consistent": cache.counts["miss"] == training_forwards,
            "on_iteration_calls": len(iter_rows) - 1,
            "best_iteration": self.best.iteration,
            "best_is_initial": self.best.iteration == 0,
            "best_val_mse": self.best.val_mse,
            "initial_params": x0,
            "best_params": self.best.weights,
            "final_params": self.final_weights,
            "total_training_time": total_time,
            "optimizer_time": opt_time,
            "validation_time": timers["val"],
            "mean_time_per_iteration": opt_time / nit if nit else np.nan,
            # Primary timing quantity: independent of how long a run takes to converge.
            "time_per_evaluation": opt_time / int(result.nfev),
            "estimator_calls": opt_calls,
            "circuits_evaluated": opt_circuits,
            "validation_estimator_calls": int(timers.get("val_calls", 0)),
            "validation_circuits_evaluated": int(timers.get("val_circuits", 0)),
            "iterations_without_gradient_at_xk": n_grad_mismatch,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scaled predictions from the best-validation weights."""
        return self._forward(X, self.best.weights)

    def predict_final(self, X: np.ndarray) -> np.ndarray:
        """Scaled predictions from the final weights."""
        assert self.final_weights is not None, "fit() first"
        return self._forward(X, self.final_weights)
