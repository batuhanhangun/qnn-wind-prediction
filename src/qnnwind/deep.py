"""PyTorch models: MLP, LSTM/GRU/Transformer on feature tokens, and MLP-PM.

The deep models train with Adam in float32, shuffle minibatches every epoch with a generator
seeded from the run seed, and early-stop on validation loss (the best-epoch weights are
restored). MLP-PM trains in float64 with SciPy L-BFGS-B exactly like the QNN.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
import torch
from torch import nn

from qnnwind.data import TrainVal
from qnnwind.metrics import BestValidation, mse, pad_iteration_curve

DEEP_MODELS = ("MLP", "LSTM", "GRU", "Transformer")
N_FEATURES = 4


def count_trainable(module: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class FeatureTokenizer(nn.Module):
    """Token e_j = x_j * w_j + b_j for each feature j, with learned w_j, b_j in R^d."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        bound = d_token**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> tokens: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class MLP(nn.Module):
    """``n_layers`` hidden layers of equal ``width`` and a linear output."""

    def __init__(self, n_layers: int, width: int, activation: str) -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        size = N_FEATURES
        for _ in range(n_layers):
            layers += [nn.Linear(size, width), act()]
            size = width
        layers.append(nn.Linear(size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RecurrentOnTokens(nn.Module):
    """LSTM or GRU over the 4 feature tokens; a linear head reads the final hidden state.

    Dropout is applied between recurrent layers (only when ``n_layers`` = 2, where PyTorch
    supports it) and to the final hidden state before the head.
    """

    def __init__(
        self, cell: str, d_token: int, n_layers: int, hidden_size: int, dropout: float
    ) -> None:
        super().__init__()
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
        self.tokenizer = FeatureTokenizer(N_FEATURES, d_token)
        self.rnn = rnn_cls(
            input_size=d_token,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(self.tokenizer(x))
        return self.head(self.dropout(output[:, -1, :])).squeeze(-1)


class TransformerOnTokens(nn.Module):
    """FT-Transformer style: [CLS] + 4 feature tokens, no positional encoding."""

    def __init__(
        self, d_token: int, n_heads: int, n_layers: int, dim_feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(N_FEATURES, d_token)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.uniform_(self.cls, -(d_token**-0.5), d_token**-0.5)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0, :]).squeeze(-1)


def build_network(name: str, params: dict[str, Any]) -> nn.Module:
    """Instantiate a deep model from its (tuned) hyperparameters."""
    if name == "MLP":
        return MLP(params["n_layers"], params["width"], params["activation"])
    if name in ("LSTM", "GRU"):
        return RecurrentOnTokens(
            name, params["d_token"], params["n_layers"], params["hidden_size"], params["dropout"]
        )
    if name == "Transformer":
        return TransformerOnTokens(
            params["d_token"],
            params["n_heads"],
            params["n_layers"],
            params["dim_feedforward"],
            params["dropout"],
        )
    raise ValueError(f"Unknown deep model {name!r}")


class DeepModel:
    """Adam training with per-epoch shuffled minibatches and early stopping.

    Args:
        name: One of :data:`DEEP_MODELS`.
        params: Architecture and optimizer hyperparameters (``lr``, ``weight_decay``,
            ``batch_size`` plus the architecture keys).
        seed: Run seed; seeds weight initialization, dropout, and the minibatch generator.
        deep_cfg: The ``deep`` config section (``max_epochs``, ``patience``).
    """

    def __init__(self, name: str, params: dict[str, Any], seed: int, deep_cfg: dict) -> None:
        if name not in DEEP_MODELS:
            raise ValueError(f"Unknown deep model {name!r}")
        self.name = name
        self.params = dict(params)
        self.seed = seed
        self.cfg = deep_cfg
        self.network: nn.Module | None = None
        self.curve: pd.DataFrame | None = None
        self.summary: dict[str, Any] = {}

    def fit(self, data: TrainVal) -> None:
        """Train on the training block, early-stopping on validation MSE."""
        torch.manual_seed(self.seed)
        network = build_network(self.name, self.params)
        batch_gen = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]
        )
        loss_fn = nn.MSELoss()
        X = torch.as_tensor(data.X_train, dtype=torch.float32)
        y = torch.as_tensor(data.y_train, dtype=torch.float32)
        X_val = torch.as_tensor(data.X_val, dtype=torch.float32)
        y_val = torch.as_tensor(data.y_val, dtype=torch.float32)
        batch_size = int(self.params["batch_size"])

        best_val, best_epoch, best_state = np.inf, 0, copy.deepcopy(network.state_dict())
        rows = []
        t_start = time.perf_counter()
        for epoch in range(1, int(self.cfg["max_epochs"]) + 1):
            network.train()
            order = torch.randperm(X.shape[0], generator=batch_gen)
            total = 0.0
            for start in range(0, X.shape[0], batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(network(X[idx]), y[idx])
                loss.backward()
                optimizer.step()
                total += float(loss) * idx.numel()
            network.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(network(X_val), y_val))
            rows.append({"epoch": epoch, "train_loss": total / X.shape[0], "val_mse": val_loss})
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(network.state_dict())
            elif epoch - best_epoch >= int(self.cfg["patience"]):
                break
        network.load_state_dict(best_state)
        network.eval()
        self.network = network
        self.curve = pd.DataFrame(rows)
        self.summary = {
            "model": self.name,
            "params": self.params,
            "trainable_params": count_trainable(network),
            "epochs_run": len(rows),
            "best_epoch": best_epoch,
            "best_val_mse": best_val,
            "stopped_early": len(rows) < int(self.cfg["max_epochs"]),
            "total_training_time": time.perf_counter() - t_start,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scaled predictions from the restored best-epoch weights."""
        assert self.network is not None, "fit() first"
        with torch.no_grad():
            out = self.network(torch.as_tensor(X, dtype=torch.float32))
        return out.numpy().astype(np.float64)


class MLPPM:
    """Parameter-matched MLP 4-2-1 (tanh hidden, linear output), trained like the QNN.

    Full-batch MSE in float64, torch-autograd gradients, ``scipy.optimize.minimize`` with
    L-BFGS-B and the QNN's SciPy options, best-validation selection including iteration 0.
    Weights use seeded Glorot-uniform initialization; biases start at zero.
    """

    def __init__(self, mlp_pm_cfg: dict, seed: int) -> None:
        self.cfg = mlp_pm_cfg
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        hidden = int(mlp_pm_cfg["hidden_units"])
        self.network = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        ).to(torch.float64)
        for layer in (self.network[0], self.network[2]):
            nn.init.xavier_uniform_(layer.weight, generator=generator)
            nn.init.zeros_(layer.bias)
        self._params = list(self.network.parameters())
        self.best = BestValidation()
        self.final_weights: np.ndarray | None = None
        self.curve_iter: pd.DataFrame | None = None
        self.curve_eval: pd.DataFrame | None = None
        self.summary: dict[str, Any] = {}

    def _get_vector(self) -> np.ndarray:
        return torch.cat([p.detach().reshape(-1) for p in self._params]).numpy().copy()

    def _set_vector(self, theta: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for p in self._params:
                n = p.numel()
                p.copy_(torch.as_tensor(theta[offset : offset + n], dtype=torch.float64).view_as(p))
                offset += n

    def _forward(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        self._set_vector(theta)
        with torch.no_grad():
            out = self.network(torch.as_tensor(X, dtype=torch.float64))
        return out.numpy().ravel().astype(np.float64)

    def fit(self, data: TrainVal) -> None:
        opt_cfg = self.cfg["optimizer"]
        maxiter = int(opt_cfg["maxiter"])
        X = torch.as_tensor(data.X_train, dtype=torch.float64)
        y = torch.as_tensor(data.y_train, dtype=torch.float64)
        x0 = self._get_vector()

        eval_rows: list[dict[str, Any]] = []
        iter_rows: list[dict[str, Any]] = []
        timers = {"val": 0.0}
        last: dict[str, Any] = {}

        def fun_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            self._set_vector(theta)
            for p in self._params:
                p.grad = None
            loss = torch.mean((self.network(X).squeeze(-1) - y) ** 2)
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1) for p in self._params]).numpy().copy()
            now = time.perf_counter()
            value = float(loss)
            eval_rows.append(
                {
                    "evaluation": len(eval_rows) + 1,
                    "train_mse": value,
                    "wall_time": now - t_start,
                    "optimizer_time": now - t_start - timers["val"],
                }
            )
            last.update(x=np.array(theta, copy=True), f=value, g=grad)
            last.setdefault("first", (np.array(theta, copy=True), value, grad, len(eval_rows)))
            return value, grad

        def validation_mse(theta: np.ndarray) -> float:
            t0 = time.perf_counter()
            value = mse(data.y_val, self._forward(data.X_val, theta))
            timers["val"] += time.perf_counter() - t0
            return value

        def on_iteration(xk: np.ndarray) -> None:
            now = time.perf_counter()
            at_xk = bool(np.array_equal(last["x"], xk))
            row = {
                "iteration": len(iter_rows),
                "train_mse": last["f"] if at_xk else np.nan,
                "grad_norm": float(np.linalg.norm(last["g"])) if at_xk else np.nan,
                "objective_evaluations": len(eval_rows),
                "optimizer_time": now - t_start - timers["val"],
            }
            row["val_mse"] = validation_mse(xk)
            row["validation_time"] = timers["val"]
            iter_rows.append(row)
            self.best.offer(row["iteration"], row["val_mse"], xk)

        t_start = time.perf_counter()
        val0 = validation_mse(x0)
        iter_rows.append({"iteration": 0, "val_mse": val0, "validation_time": timers["val"]})
        self.best.offer(0, val0, x0)
        result = scipy.optimize.minimize(
            fun_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "maxfun": int(opt_cfg["maxfun"]),
                "ftol": float(opt_cfg["ftol"]),
                "gtol": float(opt_cfg["gtol"]),
            },
            callback=on_iteration,
        )
        total_time = time.perf_counter() - t_start

        first_x, first_f, first_g, _ = last["first"]
        if not np.array_equal(first_x, x0):
            raise RuntimeError("The first evaluation was not at the initial point")
        iter_rows[0].update(
            train_mse=first_f,
            grad_norm=float(np.linalg.norm(first_g)),
            objective_evaluations=1,
            optimizer_time=eval_rows[0]["optimizer_time"],
        )
        self.final_weights = np.array(result.x, dtype=np.float64, copy=True)
        curve_iter = pd.DataFrame(iter_rows)
        self.curve_iter = pad_iteration_curve(curve_iter, maxiter)
        self.curve_eval = pd.DataFrame(eval_rows)
        nit = int(result.nit)
        opt_time = total_time - timers["val"]
        self.summary = {
            "model": "MLP-PM",
            "seed": self.seed,
            "trainable_params": count_trainable(self.network),
            "scipy": {
                "nit": nit,
                "nfev": int(result.nfev),
                "njev": int(result.njev),
                "status": int(result.status),
                "success": bool(result.success),
                "message": str(result.message),
            },
            "stopped_before_maxiter": nit < maxiter,
            "objective_evaluations": len(eval_rows),
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
            "time_per_evaluation": opt_time / int(result.nfev),
            "iterations_without_gradient_at_xk": int(curve_iter["grad_norm"].isna().sum()),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scaled predictions from the best-validation weights."""
        return self._forward(X, self.best.weights)

    def predict_final(self, X: np.ndarray) -> np.ndarray:
        """Scaled predictions from the final weights."""
        assert self.final_weights is not None, "fit() first"
        return self._forward(X, self.final_weights)
