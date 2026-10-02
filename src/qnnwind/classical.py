"""Classical regression baselines: LR, kNN, DTR, SVR, XGBoost, LightGBM.

All run single-threaded (``n_jobs=1``). XGBoost and LightGBM use a fixed ``n_estimators``
with early stopping on the validation block; both then predict with the best iteration.
"""

from __future__ import annotations

from typing import Any

import lightgbm
import numpy as np
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from qnnwind.data import TrainVal

CLASSICAL_MODELS = ("LR", "kNN", "DTR", "SVR", "XGBoost", "LightGBM")


class ClassicalModel:
    """A scikit-learn-style regressor behind the project's fit/predict interface.

    Args:
        name: One of :data:`CLASSICAL_MODELS`.
        params: Tuned hyperparameters (empty for LR).
        seed: Run seed (``random_state`` for DTR, XGBoost, and LightGBM).
        boosting_cfg: The ``boosting`` config section.
    """

    def __init__(self, name: str, params: dict[str, Any], seed: int, boosting_cfg: dict) -> None:
        if name not in CLASSICAL_MODELS:
            raise ValueError(f"Unknown classical model {name!r}")
        self.name = name
        self.params = dict(params)
        self.seed = seed
        self.boosting_cfg = boosting_cfg
        self.estimator: Any = None
        self.summary: dict[str, Any] = {}

    def _build(self) -> Any:
        p, seed = self.params, self.seed
        if self.name == "LR":
            return LinearRegression()
        if self.name == "kNN":
            return KNeighborsRegressor(
                n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"], n_jobs=1
            )
        if self.name == "DTR":
            return DecisionTreeRegressor(
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                min_samples_split=p["min_samples_split"],
                random_state=seed,
            )
        if self.name == "SVR":
            return SVR(kernel="rbf", C=p["C"], epsilon=p["epsilon"], gamma=p["gamma"])
        if self.name == "XGBoost":
            return xgboost.XGBRegressor(
                n_estimators=self.boosting_cfg["n_estimators"],
                early_stopping_rounds=self.boosting_cfg["early_stopping_rounds"],
                max_depth=p["max_depth"],
                learning_rate=p["learning_rate"],
                subsample=p["subsample"],
                colsample_bytree=p["colsample_bytree"],
                min_child_weight=p["min_child_weight"],
                reg_lambda=p["reg_lambda"],
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
        return lightgbm.LGBMRegressor(
            n_estimators=self.boosting_cfg["n_estimators"],
            num_leaves=p["num_leaves"],
            learning_rate=p["learning_rate"],
            min_child_samples=p["min_child_samples"],
            subsample=p["subsample"],
            subsample_freq=1,
            colsample_bytree=p["colsample_bytree"],
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )

    def fit(self, data: TrainVal) -> None:
        """Fit on the training block; boosting models early-stop on the validation block."""
        self.estimator = self._build()
        if self.name == "XGBoost":
            self.estimator.fit(
                data.X_train, data.y_train, eval_set=[(data.X_val, data.y_val)], verbose=False
            )
        elif self.name == "LightGBM":
            # LightGBM 4.7 deprecates eval_set (LGBMDeprecationWarning) in favour of eval_X/y.
            self.estimator.fit(
                data.X_train,
                data.y_train,
                eval_X=data.X_val,
                eval_y=data.y_val,
                callbacks=[
                    lightgbm.early_stopping(
                        self.boosting_cfg["early_stopping_rounds"], verbose=False
                    )
                ],
            )
        else:
            self.estimator.fit(data.X_train, data.y_train)
        self.summary = {"model": self.name, "params": self.params, **self._complexity()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scaled predictions (boosting models use their best iteration)."""
        return np.asarray(self.estimator.predict(X), dtype=np.float64).ravel()

    def _complexity(self) -> dict[str, Any]:
        """Trainable parameter count, or a labeled complexity measure."""
        est = self.estimator
        if self.name == "LR":
            return {"trainable_params": int(est.coef_.size + 1), "complexity": {}}
        if self.name == "kNN":
            return {
                "trainable_params": 0,
                "complexity": {"stored_training_samples": int(est.n_samples_fit_)},
            }
        if self.name == "DTR":
            return {
                "trainable_params": None,
                "complexity": {
                    "tree_nodes": int(est.tree_.node_count),
                    "tree_leaves": int(est.get_n_leaves()),
                    "tree_depth": int(est.get_depth()),
                },
            }
        if self.name == "SVR":
            n_sv = int(est.support_.size)
            return {
                "trainable_params": n_sv + 1,  # dual coefficients + intercept
                "complexity": {"support_vectors": n_sv},
            }
        if self.name == "XGBoost":
            best = int(est.best_iteration)
            trees = est.get_booster()[: best + 1].trees_to_dataframe()
            return {
                "trainable_params": None,
                "complexity": {
                    "trees": best + 1,
                    "tree_nodes": int(len(trees)),
                    "tree_leaves": int((trees["Feature"] == "Leaf").sum()),
                    "best_iteration": best,
                },
            }
        booster = est.booster_
        best = int(est.best_iteration_) if est.best_iteration_ else booster.current_iteration()
        dump = booster.dump_model(num_iteration=best)
        leaves = [tree["num_leaves"] for tree in dump["tree_info"]]
        return {
            "trainable_params": None,
            "complexity": {
                "trees": len(leaves),
                "tree_nodes": int(sum(2 * n - 1 for n in leaves)),
                "tree_leaves": int(sum(leaves)),
                "best_iteration": best,
            },
        }
