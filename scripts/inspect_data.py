"""Dataset inspection.

Asserts the dataset SHA-256 and format, checks the blocked folds, and writes to
``outputs/tables``:

* ``t1_descriptive.csv``: per-variable mean, median, std (ddof=1), min, max, range (T1);
* ``t1b_folds.csv``: per-fold row ranges and power / wind-speed mean and max per block (T1b);
* ``autocorrelation.csv``: autocorrelation of each variable in row order at the configured lags.

Usage: ``python scripts/inspect_data.py [--config configs/experiment.yaml]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aggregate import require_dataset  # noqa: E402
from qnnwind.data import load_dataset  # noqa: E402
from qnnwind.folds import check_fold, folds_from_config, ranges  # noqa: E402
from qnnwind.io import load_config, write_csv  # noqa: E402
from qnnwind.stats import autocorrelation  # noqa: E402


def descriptive(frame: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "mean": frame.mean(),
            "median": frame.median(),
            "std": frame.std(ddof=1),
            "min": frame.min(),
            "max": frame.max(),
        }
    )
    stats["range"] = stats["max"] - stats["min"]
    return stats.rename_axis("variable").reset_index()


def fold_table(frame: pd.DataFrame, folds: list, target: str, speed: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        for block, idx in (
            ("train_pool", fold.train_pool),
            ("validation", fold.validation),
            ("test", fold.test),
        ):
            sub = frame.iloc[idx]
            rows.append(
                {
                    "fold": fold.k,
                    "block": block,
                    "rows": " ; ".join(f"{a}-{b}" for a, b in ranges(idx)),
                    "size": int(idx.size),
                    "power_mean": sub[target].mean(),
                    "power_max": sub[target].max(),
                    "speed_mean": sub[speed].mean(),
                    "speed_max": sub[speed].max(),
                }
            )
        rows.append(
            {
                "fold": fold.k,
                "block": "buffers",
                "rows": " ; ".join(f"{a}-{b}" for a, b in ranges(fold.buffers)),
                "size": int(fold.buffers.size),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cfg = config["data"]
    require_dataset(config)
    dataset = load_dataset(config.path("data"), data_cfg)  # asserts SHA-256 and format
    columns = [*data_cfg["feature_columns"], data_cfg["target_column"]]
    frame = pd.DataFrame(np.column_stack([dataset.features, dataset.target]), columns=columns)
    folds = folds_from_config(config["folds"], dataset.n_rows)
    for fold in folds:
        check_fold(fold, dataset.n_rows, config["folds"]["buffer"])

    stats = descriptive(frame)
    t1b = fold_table(frame, folds, data_cfg["target_column"], data_cfg["wind_speed_column"])
    lags = config["inspect"]["acf_lags"]
    acf = pd.DataFrame(
        [
            {"variable": c, **{f"lag_{k}": v for k, v in autocorrelation(frame[c], lags).items()}}
            for c in columns
        ]
    )

    tables = config.path("outputs") / "tables"
    write_csv(tables / "t1_descriptive.csv", stats)
    write_csv(tables / "t1b_folds.csv", t1b)
    write_csv(tables / "autocorrelation.csv", acf)

    pd.set_option("display.width", 160)
    print(f"SHA-256 OK: {dataset.sha256}")
    print(f"Rows: {dataset.n_rows}; columns: {columns}")
    print("\nT1 descriptive statistics:")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nT1b per-fold blocks:")
    print(t1b.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nAutocorrelation in row order:")
    print(acf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote t1_descriptive.csv, t1b_folds.csv, autocorrelation.csv to {tables}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
