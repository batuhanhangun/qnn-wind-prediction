"""Environment check for the container and for a local virtual environment.

The container is the unchanged ``quantum_ml:v4`` image with the extras directory mounted
read-only at ``/opt/qnnwind-extras`` (``QNNWIND_EXTRAS_MOUNT``, set by slurm/in_container.sh)
and first on ``PYTHONPATH``.

1. Package versions:
   * container mode: every package in ``environment/v4_freeze.txt`` still has its v4 version,
     and every pin in ``environment/requirements-extra.txt`` is installed;
   * local mode (Windows, Linux, or macOS): Python 3.11, every pin in
     ``requirements-local.txt`` and ``requirements-extra.txt`` is installed, torch is the
     platform's CPU build, and every installed package listed in the platform's constraints
     file (``constraints-local-{win,linux,macos}.txt``) matches it.
2. The 12 extras import (``lightgbm`` needs the system ``libgomp.so.1``); in the container
   they load from the extras directory, which holds exactly those 12 distributions.
3. The Qiskit and SciPy API facts that the QNN implementation relies on, re-checked in the
   installed
   versions, plus the XGBoost, LightGBM, Optuna, and torch behaviour used by the project.

Usage: ``python scripts/check_env.py [--mode auto|container|local]``. Exits non-zero on any
failure. ``auto`` means ``container`` when the extras directory is mounted
(``QNNWIND_EXTRAS_MOUNT`` is set or ``/opt/qnnwind-extras`` exists) and ``local`` otherwise.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import inspect  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Callable  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "environment"
sys.path.insert(0, str(ROOT / "src"))

CHECKS: list[tuple[str, Callable[[], str]]] = []

EXTRAS_MOUNT = Path("/opt/qnnwind-extras")
# Local environments: the constraints file and the torch build of each platform.
LOCAL_PINS = {
    "win32": ("constraints-local-win.txt", "2.5.1+cpu"),
    "linux": ("constraints-local-linux.txt", "2.5.1+cpu"),
    "darwin": ("constraints-local-macos.txt", "2.5.1"),
}


def in_container() -> bool:
    """True inside the cluster container, which mounts the extras directory."""
    return bool(os.environ.get("QNNWIND_EXTRAS_MOUNT")) or EXTRAS_MOUNT.is_dir()


def local_platform() -> str:
    for key in LOCAL_PINS:
        if sys.platform.startswith(key):
            return key
    raise RuntimeError(f"unsupported platform {sys.platform!r} (Windows, Linux, or macOS)")


def check(func: Callable[[], str]) -> Callable[[], str]:
    CHECKS.append((func.__name__, func))
    return func


def read_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def compare(pins: dict[str, str], must_exist: bool) -> list[str]:
    problems = []
    for name, version in pins.items():
        found = installed_version(name)
        if found is None:
            if must_exist:
                problems.append(f"{name}: not installed (expected {version})")
        elif found != version:
            problems.append(f"{name}: {found} installed, expected {version}")
    return problems


def check_versions(mode: str) -> list[str]:
    extra = read_pins(ENV_DIR / "requirements-extra.txt")
    if mode == "container":
        problems = compare(read_pins(ENV_DIR / "v4_freeze.txt"), must_exist=True)
        problems += compare(extra, must_exist=True)
        expected_python = (ENV_DIR / "v4_python_version.txt").read_text(encoding="utf-8").strip()
        if f"Python {sys.version.split()[0]}" != expected_python:
            problems.append(f"Python {sys.version.split()[0]} != {expected_python}")
    else:
        constraints, torch_build = LOCAL_PINS[local_platform()]
        problems = compare(read_pins(ENV_DIR / "requirements-local.txt"), must_exist=True)
        problems += compare(extra, must_exist=True)
        problems += compare({"torch": torch_build}, must_exist=True)
        pins = read_pins(ENV_DIR / constraints)
        pins.pop("torch", None)  # checked above
        problems += compare(pins, must_exist=False)
        if sys.version_info[:2] != (3, 11):
            problems.append(f"Python {sys.version.split()[0]}: 3.11 is required")
    return problems


# --------------------------------------------------------------------------------------
# Extras directory
# --------------------------------------------------------------------------------------


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


@check
def extras_importable() -> str:
    """All 12 extras import; in the container they come from the mounted extras directory."""
    import importlib

    pins = read_pins(ENV_DIR / "requirements-extra.txt")
    try:
        import lightgbm  # noqa: F401
    except OSError as exc:  # the wheel needs the system OpenMP runtime
        raise AssertionError(
            f"import lightgbm failed ({exc}); the OpenMP runtime is missing (container: "
            "libgomp.so.1 in the image; Linux: install libgomp1; macOS: brew install libomp)"
        ) from exc
    # Every one of the 12 imports under its normalized distribution name.
    modules = {d: importlib.import_module(_normalize(d).replace("-", "_")) for d in pins}
    mount = os.environ.get("QNNWIND_EXTRAS_MOUNT")
    if not mount:
        return f"{len(modules)} extras import (no extras mount: local environment)"
    root = Path(mount).resolve()
    outside = [d for d, m in modules.items() if root not in Path(m.__file__).resolve().parents]
    assert not outside, f"extras not loaded from {root}: {outside}"
    installed = {_normalize(d.metadata["Name"]) for d in metadata.distributions(path=[str(root)])}
    expected = {_normalize(d) for d in pins}
    assert installed == expected, (
        f"{root} must hold exactly the {len(expected)} extras: unexpected "
        f"{sorted(installed - expected)}, missing {sorted(expected - installed)}"
    )
    return f"{len(modules)} extras import from {root}, which holds exactly those distributions"


# --------------------------------------------------------------------------------------
# API facts
# --------------------------------------------------------------------------------------


@check
def circuits_match_spec_table() -> str:
    from qnnwind.circuits import all_circuits, cnot_blocks
    from qnnwind.io import load_config

    expected = {  # config: (cnots, total gates, depth)
        "QNN-1": (12, 40, 16),
        "QNN-2": (6, 34, 12),
        "QNN-3": (8, 36, 15),
        "QNN-4": (8, 36, 15),
        "QNN-5": (6, 34, 12),
        "QNN-6": (6, 34, 11),
    }
    config = load_config(ROOT / "configs" / "experiment.yaml")
    for qc in all_circuits(config["qnn"]):
        ops = qc.circuit.count_ops()
        got = (ops.get("cx", 0), sum(ops.values()), qc.circuit.depth())
        assert got == expected[qc.name], f"{qc.name}: {got} != {expected[qc.name]}"
        assert set(ops) == {"h", "p", "ry", "cx"}, f"{qc.name}: unexpected gates {dict(ops)}"
        assert len(cnot_blocks(qc.ansatz, 2)) == 2
    return "gate counts and depths of QNN-1..6 match the expected circuits"


@check
def estimator_qnn_default_precision() -> str:
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    default = inspect.signature(EstimatorQNN.__init__).parameters["default_precision"].default
    assert default == 0.015625, default
    return f"EstimatorQNN default_precision default = {default} (project passes 0.0)"


@check
def forward_is_exact_and_deterministic() -> str:
    import numpy as np
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from qnnwind.circuits import build_circuit
    from qnnwind.io import load_config
    from qnnwind.qnn import build_qnn

    config = load_config(ROOT / "configs" / "experiment.yaml")
    qc = build_circuit("QNN-1", "full", config["qnn"])
    qnn, _ = build_qnn(qc)
    rng = np.random.default_rng(1)
    x, w = rng.random((3, 4)), rng.random(12)
    a, b = qnn.forward(x, w).ravel(), qnn.forward(x, w).ravel()
    assert np.array_equal(a, b)
    for i in range(3):
        bound = qc.circuit.assign_parameters(np.concatenate([x[i], w]))
        exact = Statevector(bound).expectation_value(SparsePauliOp("ZZZZ")).real
        assert abs(a[i] - exact) < 1e-12
    return "default_precision=0.0: repeated forward calls identical and exact to 1e-12"


@check
def lbfgsb_options_sent_to_scipy() -> str:
    import numpy as np
    import qiskit_machine_learning.optimizers.scipy_optimizer as so
    import scipy.optimize
    from qiskit_machine_learning.optimizers import L_BFGS_B

    assert so.minimize is scipy.optimize.minimize, "Qiskit ML calls a different minimize"
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return scipy.optimize.OptimizeResult(x=kwargs["x0"], fun=0.0, nfev=0, nit=0)

    def callback(xk):
        return None

    so.minimize = spy
    try:
        L_BFGS_B(maxiter=100, options={"gtol": 1e-12}, callback=callback).minimize(
            fun=lambda x: 0.0, x0=np.zeros(2), jac=lambda x: np.zeros(2)
        )
    finally:
        so.minimize = scipy.optimize.minimize
    expected = {
        "maxiter": 100,
        "maxfun": 15000,
        "ftol": 2.220446049250313e-15,
        "gtol": 1e-12,
        "iprint": -1,
        "eps": 1e-08,
    }
    assert seen["options"] == expected, seen["options"]
    assert seen["callback"] is callback and seen["method"] == "l-bfgs-b"
    return f"L_BFGS_B sends options {expected} and forwards callback"


@check
def scipy_iprint_deprecation_message() -> str:
    import numpy as np
    import scipy.optimize

    from qnnwind.qnn import IPRINT_DEPRECATION_MESSAGE

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scipy.optimize.minimize(
            lambda x: float(x @ x), np.ones(2), method="L-BFGS-B", options={"iprint": -1}
        )
    import re

    hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert hits and re.match(IPRINT_DEPRECATION_MESSAGE, str(hits[0].message)), [
        str(w.message) for w in caught
    ]
    return "SciPy emits the iprint DeprecationWarning with the filtered message"


@check
def regressor_callbacks_and_loss() -> str:
    import numpy as np

    from qnnwind.data import MinMax, TrainVal
    from qnnwind.folds import make_fold
    from qnnwind.io import load_config
    from qnnwind.qnn import QNNModel

    config = load_config(ROOT / "configs" / "experiment.yaml", {"qnn.optimizer.maxiter": 3})
    fold = make_fold(5, 4464, 6, 744, 12, 464)
    rng = np.random.default_rng(0)
    rows = fold.train_pool[:12]
    X, y = rng.random((12, 4)), rng.uniform(-1, 1, 12)
    Xv, yv = rng.random((464, 4)), rng.uniform(-1, 1, 464)
    scaler = MinMax(np.zeros(4), np.ones(4), 0.0, 1.0)
    data = TrainVal(X, y, Xv, yv, rows, fold.validation, scaler, scaler, fold)
    model = QNNModel("QNN-2", config["qnn"], seed=0)
    model.fit(data)
    s = model.summary
    assert s["regressor_callback_calls"] == s["scipy"]["nfev"], s
    assert s["on_iteration_calls"] == s["scipy"]["nit"], s
    assert s["cache_tracker_consistent"], s
    f0 = model._forward(X, model.summary["initial_params"])
    assert abs(model.curve_eval["train_mse"].iloc[0] - np.sum((f0 - y) ** 2) / len(y)) < 1e-14
    return (
        f"regressor callbacks = nfev = {s['scipy']['nfev']}, on_iteration calls = nit = "
        f"{s['scipy']['nit']}, objective = SSE / n"
    )


@check
def initial_point_matches_algorithm_globals() -> str:
    import numpy as np
    from qiskit_machine_learning.utils import algorithm_globals

    for seed in range(5):
        algorithm_globals.random_seed = seed
        assert np.array_equal(
            algorithm_globals.random.random(12), np.random.default_rng(seed).random(12)
        )
    return "algorithm_globals.random.random(12) == default_rng(seed).random(12) for seeds 0-4"


@check
def estimator_internals_for_counting() -> str:
    from qiskit.primitives import StatevectorEstimator

    assert callable(getattr(StatevectorEstimator, "_run", None))
    params = list(inspect.signature(StatevectorEstimator.run).parameters)
    assert params == ["self", "pubs", "precision"], params
    return "StatevectorEstimator.run(pubs, *, precision) and _run(pubs) exist"


@check
def boosting_best_iteration() -> str:
    import lightgbm
    import numpy as np
    import xgboost

    params = inspect.signature(xgboost.XGBModel.__init__).parameters
    assert "early_stopping_rounds" in params
    rng = np.random.default_rng(0)
    X, Xv = rng.random((200, 4)), rng.random((100, 4))
    y, yv = X.sum(1) + rng.normal(0, 0.3, 200), Xv.sum(1) + rng.normal(0, 0.3, 100)
    xgb = xgboost.XGBRegressor(
        n_estimators=1000, early_stopping_rounds=50, n_jobs=1, verbosity=0, random_state=0
    ).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    full = xgb.predict(Xv, iteration_range=(0, xgb.best_iteration + 1))
    assert np.array_equal(xgb.predict(Xv), full)
    lgb = lightgbm.LGBMRegressor(
        n_estimators=1000, subsample_freq=1, subsample=0.8, verbosity=-1, n_jobs=1, random_state=0
    ).fit(X, y, eval_X=Xv, eval_y=yv, callbacks=[lightgbm.early_stopping(50, verbose=False)])
    assert np.array_equal(lgb.predict(Xv), lgb.predict(Xv, num_iteration=lgb.best_iteration_))
    return (
        f"XGBoost {xgboost.__version__} (early_stopping_rounds in constructor) and LightGBM "
        f"{lightgbm.__version__} predict with the best iteration"
    )


@check
def optuna_api() -> str:
    import optuna

    from qnnwind.io import sqlite_url

    assert optuna.__version__.startswith("5."), optuna.__version__
    tpe = inspect.signature(optuna.samplers.TPESampler.__init__).parameters
    assert "seed" in tpe and tpe["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    create = inspect.signature(optuna.create_study).parameters
    for name in ("storage", "sampler", "study_name", "direction", "load_if_exists"):
        assert name in create, name
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        url = sqlite_url(Path(tmp) / "check.db")
        study = optuna.create_study(
            storage=url,
            sampler=optuna.samplers.TPESampler(seed=0),
            study_name="s",
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(lambda t: t.suggest_float("x", -1, 1) ** 2, n_trials=3)
        again = optuna.create_study(storage=url, study_name="s", load_if_exists=True)
        assert len(again.trials) == 3
    return f"Optuna {optuna.__version__}: TPESampler(seed=...), SQLite storage resume"


@check
def torch_single_thread() -> str:
    import torch

    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1
    return f"torch {torch.__version__}: set_num_threads(1) works"


@check
def no_other_quantum_frameworks() -> str:
    import qnnwind.circuits  # noqa: F401
    import qnnwind.qnn  # noqa: F401

    loaded = [m for m in ("qiskit_algorithms", "qiskit_aer", "pennylane") if m in sys.modules]
    assert not loaded, loaded
    return "qnnwind imports none of qiskit_algorithms, qiskit_aer, pennylane"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("auto", "container", "local"), default="auto")
    args = parser.parse_args(argv)
    mode = args.mode if args.mode != "auto" else ("container" if in_container() else "local")

    failures = 0
    print(f"check_env: mode={mode}, Python {sys.version.split()[0]}, {sys.platform}")
    problems = check_versions(mode)
    if problems:
        failures += 1
        print("[FAIL] package versions:")
        for p in problems:
            print(f"       {p}")
    else:
        print("[ OK ] package versions match the pins")

    warnings.simplefilter("error", DeprecationWarning)
    warnings.simplefilter("error", FutureWarning)  # includes LGBMDeprecationWarning
    for name, func in CHECKS:
        try:
            with warnings.catch_warnings():
                if name == "scipy_iprint_deprecation_message":
                    warnings.simplefilter("always")
                detail = func()
            print(f"[ OK ] {name}: {detail}")
        except Exception:  # noqa: BLE001 - report every failing check
            failures += 1
            print(f"[FAIL] {name}:")
            traceback.print_exc(limit=3)
    print("check_env: " + ("all checks passed" if not failures else f"{failures} check(s) failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
