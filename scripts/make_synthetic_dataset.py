"""Write a synthetic dataset of the same shape as the real one, and a smoke configuration for it.

The file has the columns, separator, row count, and value ranges of ``data/total_dataset.csv``
(see data/README.md), with power following a smooth power curve of the wind speed plus noise.
It is for testing the pipeline without the real dataset (for example in continuous
integration); its results mean nothing. The configuration inherits ``configs/smoke.yaml`` and
points to the synthetic file and its SHA-256, with its own results subtree.

Usage::

    python scripts/make_synthetic_dataset.py [--out DIR]
    python scripts/run_grid.py --config DIR/smoke_synthetic.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qnnwind.io import load_config  # noqa: E402
from runs_dir import DEFAULT_RUNS  # noqa: E402


def synthetic_frame(n_rows: int, columns: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    velocity = np.clip(rng.gamma(4.0, 2.2, n_rows), 0.3, 21.0)
    power_curve = 2050.0 / (1.0 + np.exp(-(velocity - 9.0) / 1.4))
    power = np.clip(power_curve + rng.normal(0.0, 60.0, n_rows), 2.0, 2035.0)
    values = {
        "Temperature": np.clip(rng.normal(4.0, 2.0, n_rows), -5.3, 10.0),
        "Pressure": np.clip(rng.normal(1019.5, 13.0, n_rows), 979.8, 1035.7),
        "Theta": rng.uniform(100.7, 359.8, n_rows),
        "Velocity": velocity,
        "Power": power,
    }
    return pd.DataFrame({c: np.round(values[c], 2) for c in columns})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_RUNS / "synthetic")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    smoke = ROOT / "configs" / "smoke.yaml"
    data_cfg = load_config(smoke)["data"]
    columns = [*data_cfg["feature_columns"], data_cfg["target_column"]]
    data = out / "total_dataset.csv"
    synthetic_frame(int(data_cfg["n_rows"]), columns).to_csv(
        data, sep=data_cfg["separator"], index=False, lineterminator="\n"
    )
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    config = out / "smoke_synthetic.yaml"
    lines = [
        "# Smoke configuration on a synthetic dataset (scripts/make_synthetic_dataset.py).",
        f"base: {smoke.as_posix()}",
        "paths:",
        f"  data: {data.as_posix()}",
        "  results: ${QNNWIND_SCRATCH:-.}/results/smoke_synthetic",
        "  logs: ${QNNWIND_SCRATCH:-.}/logs/smoke_synthetic",
        "  tasks: ${QNNWIND_SCRATCH:-.}/tasks/smoke_synthetic",
        "data:",
        f"  sha256: {sha}",
        "",
    ]
    config.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"make_synthetic_dataset: {data} (SHA-256 {sha[:12]}), config {config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
