"""Aggregate results into tidy CSVs.

Input: one or more *sources*, each a batch of runs: a name, a config, and either the run
directory that holds its raw results or a batch directory of the results package (detected by
its ``runs.csv.gz``; see results/README.md). The batches of the paper are
``configs/experiment.yaml`` (blocked protocol, [-1, 1] readout), ``configs/blocked_unit.yaml``
(QNN-1u to QNN-6u), and ``configs/random.yaml`` (random K-fold protocol). Sources are grouped
by protocol, and each
protocol is aggregated into ``<aggregated>/<protocol>/``:

* ``runs.csv``: one row per run, with ``batch``, ``protocol``, and ``readout``;
* ``completeness.csv``/``.json`` and ``provenance.json``: expected vs found runs and the git
  commit, config hash, and dirty flags, all per batch;
* ``predictions_test.csv.gz`` (per-sample test errors, plus predictions and actual values for
  raw results), ``curves_iter.csv``, ``curves_eval.csv``: inputs for figures;
* ``main.csv`` (mean +- std over folds and seeds, with the clipped metrics of T3c),
  ``per_fold.csv``, ``pooled.csv``;
* ``best_qnn.csv`` and ``stats_tests.csv``: per readout;
* ``timing_*.csv``, ``convergence.csv``, ``stability*.csv``: per readout;
* ``hyperparameters.csv``, ``complexity.csv``.

With both protocols, ``<aggregated>/protocol_comparison.csv`` holds T9 (random minus blocked).
A partial grid is handled: whatever cannot be computed is left out and reported as such.

Usage::

    python scripts/aggregate.py --config configs/experiment.yaml             # one batch
    python scripts/aggregate.py --source primary configs/experiment.yaml RUNS1 \\
        --source blocked_unit configs/blocked_unit.yaml RUNS2 \\
        --source random configs/random.yaml RUNS2 --aggregated DIR
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qnnwind.circuits import total_gates  # noqa: E402
from qnnwind.io import (  # noqa: E402
    Config,
    load_config,
    read_json,
    run_dir,
    tuning_dir,
    validate_run_dir,
    write_csv,
    write_json,
)
from qnnwind.metrics import regression_metrics  # noqa: E402
from qnnwind.runner import readout_label, target_scaling  # noqa: E402
from qnnwind.stats import (  # noqa: E402
    curve_stability,
    diebold_mariano,
    holm,
    linear_fit,
    select_best_qnn,
    stability_scores,
    wilcoxon_signed_rank,
)
from runs_dir import use_runs_dir  # noqa: E402

# Node load: only QNN runs with busy_mean >= 0.9 W enter the timing fits.
NODE_LOAD_FRACTION = 0.9
METRICS = ("r2", "rmse", "mae", "bias", "error_std", "n_negative", "frac_negative")
BASE_FILES = ("result.json", "preds_val.csv", "preds_test.csv")
CURVE_FILES = ("curve_iter.csv", "curve_eval.csv")
PRIMARY_READOUT = "[-1, 1]"
# Pairs that implement the same unitary, for each readout.
SAME_UNITARY_PAIRS = (("QNN-1", "QNN-5"), ("QNN-1u", "QNN-5u"))

# Results package (results/README.md): one directory per batch with these files.
PACKAGE_FILES = {
    "runs": "runs.csv.gz",
    "errors": "errors_test.csv.gz",
    "pooled_seed": "pooled_seed.csv.gz",
    "curves_iter": "curves_iter.csv.gz",
    "curves_eval": "curves_eval.csv.gz",
    "tuning": "tuning.csv.gz",
}
KEY_COLUMNS = ["model", "n_train", "fold", "seed"]


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One batch of runs: a name, its config, and its resolved results directory."""

    name: str
    config: Config
    results: Path
    package: bool = False

    @property
    def protocol(self) -> str:
        return self.config.protocol


@contextmanager
def scratch(runs_dir: Path | None) -> Iterator[None]:
    """Temporarily point QNNWIND_SCRATCH at ``runs_dir`` (the directory holding results/)."""
    previous = os.environ.get("QNNWIND_SCRATCH")
    if runs_dir is not None:
        os.environ["QNNWIND_SCRATCH"] = str(Path(runs_dir).resolve())
    try:
        yield
    finally:
        if runs_dir is not None:
            if previous is None:
                os.environ.pop("QNNWIND_SCRATCH", None)
            else:
                os.environ["QNNWIND_SCRATCH"] = previous


def is_package(directory: Path | str | None) -> bool:
    """True for a batch directory of the results package."""
    return directory is not None and (Path(directory) / PACKAGE_FILES["runs"]).is_file()


def make_source(name: str, config_path: Path | str, runs_dir: Path | None = None) -> Source:
    """A source: ``runs_dir`` is a batch directory of the results package, or the run
    directory whose results are resolved with QNNWIND_SCRATCH = ``runs_dir``."""
    if is_package(runs_dir):
        config = load_config(config_path)
        return Source(name, config, Path(runs_dir).resolve(), package=True)
    with scratch(runs_dir):
        config = load_config(config_path)
        return Source(name, config, config.path("results"))


def dataset_available(config: Config) -> bool:
    """True if the dataset file is present (its checksum is verified when it is loaded)."""
    return config.path("data").is_file()


def require_dataset(config: Config) -> None:
    """Stop with a clear message if the dataset is missing (it is not part of the repository)."""
    if not dataset_available(config):
        raise SystemExit(
            f"Dataset not found: {config.path('data')}. It is not part of this repository; "
            "see data/README.md for how to obtain it and where to place it."
        )


def model_kinds(config: Config) -> dict[str, str]:
    return {m: kind for kind, names in config["models"].items() for m in names}


def model_readouts(config: Config) -> dict[str, str]:
    """The readout (target range) of every model of a config."""
    return {
        m: readout_label(target_scaling(config, m)["target_range"]) for m in model_kinds(config)
    }


def expected_runs(source: Source) -> pd.DataFrame:
    """Every (model, N, fold, seed) the batch's grid should produce."""
    config = source.config
    rows = [
        {"batch": source.name, "model": m, "kind": k, "n_train": n, "fold": f, "seed": s}
        for m, k in model_kinds(config).items()
        for n in config["sizes"]
        for f in config["folds"]["run"]
        for s in config["seeds"]
    ]
    return pd.DataFrame(rows, columns=["batch", "model", "kind", "n_train", "fold", "seed"])


def _flatten(result: dict[str, Any]) -> dict[str, Any]:
    """One tidy row from a result.json."""
    summary = result.get("model_summary", {})
    meta = result.get("metadata", {})
    scipy_info = summary.get("scipy", {})
    load = meta.get("node_load") or {}
    git = meta.get("git") or {}
    complexity = summary.get("complexity") or {}
    scaling = result.get("scaling") or {}
    row: dict[str, Any] = {
        "model": result["model"],
        "kind": result.get("kind"),
        "n_train": int(result["n_train"]),
        "fold": int(result["fold"]),
        "seed": int(result["seed"]),
    }
    for split, metrics in result.get("metrics", {}).items():
        for name in METRICS:
            row[f"{split}_{name}"] = metrics.get(name)
    row.update(
        trainable_params=summary.get("trainable_params"),
        complexity=json.dumps(complexity, sort_keys=True) if complexity else "",
        tree_nodes=complexity.get("tree_nodes"),
        tree_leaves=complexity.get("tree_leaves"),
        support_vectors=complexity.get("support_vectors"),
        stored_training_samples=complexity.get("stored_training_samples"),
        y_min_kw=scaling.get("y_min_kw"),
        y_max_kw=scaling.get("y_max_kw"),
        fit_time=result.get("fit_time"),
        total_training_time=summary.get("total_training_time"),
        optimizer_time=summary.get("optimizer_time"),
        time_per_evaluation=summary.get("time_per_evaluation"),
        mean_time_per_iteration=summary.get("mean_time_per_iteration"),
        nit=scipy_info.get("nit"),
        nfev=scipy_info.get("nfev"),
        scipy_message=scipy_info.get("message"),
        stopped_before_maxiter=summary.get("stopped_before_maxiter"),
        cache_near_hits=summary.get("cache_near_hits"),
        best_iteration=summary.get("best_iteration"),
        best_is_initial=summary.get("best_is_initial"),
        epochs_run=summary.get("epochs_run"),
        best_epoch=summary.get("best_epoch"),
        node_workers=load.get("workers"),
        node_busy_mean=load.get("busy_mean"),
        node_busy_min=load.get("busy_min"),
        node_samples=load.get("samples"),
        workers_per_node=meta.get("workers_per_node"),
        git_commit=git.get("commit"),
        git_dirty=git.get("dirty"),
        git_source=git.get("source"),
        config_hash=meta.get("config_hash"),
        dataset_sha256=meta.get("dataset_sha256"),
        hostname=meta.get("hostname"),
        cpu_model=meta.get("cpu_model"),
        peak_rss_mb=meta.get("peak_rss_mb"),
        start_time=meta.get("start_time"),
        end_time=meta.get("end_time"),
        hyperparameters=json.dumps(result.get("hyperparameters") or {}, sort_keys=True),
    )
    return row


def collect(source: Source) -> dict[str, pd.DataFrame]:
    """Read every expected run of one batch that exists and validates."""
    if source.package:
        return collect_package(source)
    kinds = model_kinds(source.config)
    readouts = model_readouts(source.config)
    expected = expected_runs(source)
    rows, preds, iters, evals = [], [], [], []
    found = []
    for rec in expected.itertuples(index=False):
        directory = run_dir(source.results, rec.model, rec.n_train, rec.fold, rec.seed)
        files = BASE_FILES + (CURVE_FILES if kinds[rec.model] in ("qnn", "mlp_pm") else ())
        ok = validate_run_dir(directory, files)
        found.append(ok)
        if not ok:
            continue
        key = {"model": rec.model, "n_train": rec.n_train, "fold": rec.fold, "seed": rec.seed}
        result = read_json(directory / "result.json")
        row = _flatten(result)
        protocol = result.get("protocol", source.protocol)  # absent from the primary batch
        if protocol != source.protocol:
            raise ValueError(f"{directory}: protocol {protocol!r} in a {source.protocol} batch")
        target = (result.get("scaling") or {}).get("target_range")
        readout = result.get("readout") or (
            readout_label(target) if target else readouts[rec.model]
        )
        test = pd.read_csv(directory / "preds_test.csv")
        # T3c (secondary): predictions clipped to [0, maximum training power].
        clipped = test["pred_kw"].clip(lower=0.0, upper=row["y_max_kw"])
        clip = regression_metrics(test["actual_kw"].to_numpy(), clipped.to_numpy())
        row.update(
            batch=source.name,
            protocol=protocol,
            readout=readout,
            test_clip_rmse=clip["rmse"],
            test_clip_r2=clip["r2"],
            test_clip_mae=clip["mae"],
        )
        rows.append(row)
        test = test.assign(error_kw=test["pred_kw"] - test["actual_kw"])
        preds.append(test[["row", "actual_kw", "pred_kw", "error_kw"]].assign(**key))
        if kinds[rec.model] == "qnn":
            ci = pd.read_csv(directory / "curve_iter.csv")
            iters.append(ci[["iteration", "train_mse", "val_mse", "padded"]].assign(**key))
            ce = pd.read_csv(directory / "curve_eval.csv")
            evals.append(ce[["evaluation", "train_mse"]].assign(**key))
    completeness = expected.assign(found=found)

    def concat(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

    key_cols = KEY_COLUMNS
    predictions = concat(preds, [*key_cols, "row", "actual_kw", "pred_kw", "error_kw"])
    n_rows, n_folds = int(source.config["data"]["n_rows"]), int(source.config["folds"]["n_folds"])
    return {
        "runs": pd.DataFrame(rows),
        "completeness": completeness,
        "predictions": predictions,
        "pooled_seed": pooled_seed_metrics(pooled_errors(predictions, n_rows, n_folds)),
        "curves_iter": concat(iters, [*key_cols, "iteration", "train_mse", "val_mse", "padded"]),
        "curves_eval": concat(evals, [*key_cols, "evaluation", "train_mse"]),
    }


def read_package(directory: Path, name: str) -> pd.DataFrame:
    """One table of a package batch directory, with every float read back exactly."""
    path = directory / PACKAGE_FILES[name]
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, float_precision="round_trip")


def collect_package(source: Source) -> dict[str, pd.DataFrame]:
    """The same tables as :func:`collect`, from a batch directory of the results package:
    per-run fields, per-sample test errors (no predictions or actual values), pooled metrics
    per seed, and QNN loss curves."""
    expected = expected_runs(source)
    runs = read_package(source.results, "runs")
    present = runs[KEY_COLUMNS].assign(found=True)
    found = expected.merge(present, on=KEY_COLUMNS, how="left")["found"].eq(True)
    return {
        "runs": runs.assign(batch=source.name, protocol=source.protocol),
        "completeness": expected.assign(found=found.to_numpy()),
        "predictions": read_package(source.results, "errors"),
        "pooled_seed": read_package(source.results, "pooled_seed"),
        "curves_iter": read_package(source.results, "curves_iter"),
        "curves_eval": read_package(source.results, "curves_eval"),
    }


# --------------------------------------------------------------------------------------
# Completeness and provenance
# --------------------------------------------------------------------------------------


def completeness_report(completeness: pd.DataFrame) -> dict[str, Any]:
    by_model = (
        completeness.groupby(["model", "n_train"])["found"].agg(["size", "sum"]).reset_index()
    )
    missing = completeness[~completeness["found"]]
    return {
        "expected": int(len(completeness)),
        "found": int(completeness["found"].sum()),
        "complete": bool(completeness["found"].all()),
        "per_model_n": [
            {
                "model": r.model,
                "n_train": int(r.n_train),
                "expected": int(r.size),
                "found": int(r.sum),
            }
            for r in by_model.itertuples(index=False)
        ],
        "missing": [
            f"{r.model} N={r.n_train} fold={r.fold} seed={r.seed}"
            for r in missing.itertuples(index=False)
        ],
    }


def provenance_report(runs: pd.DataFrame) -> dict[str, Any]:
    if runs.empty:
        return {"runs": 0}

    def distinct(column: str) -> list:
        return sorted({str(v) for v in runs[column].dropna()}) if column in runs else []

    if "git_commit" not in runs:  # results package: configuration hash and dataset checksum
        return {
            "runs": int(len(runs)),
            "config_hashes": distinct("config_hash"),
            "dataset_sha256": distinct("dataset_sha256"),
            "single_config_hash": len(distinct("config_hash")) == 1,
        }
    dirty = runs["git_dirty"]
    return {
        "runs": int(len(runs)),
        "git_commits": distinct("git_commit"),
        "config_hashes": distinct("config_hash"),
        "dataset_sha256": distinct("dataset_sha256"),
        "git_sources": distinct("git_source"),
        "dirty_true": int((dirty == True).sum()),  # noqa: E712 - object column
        "dirty_unknown": int(dirty.isna().sum()),
        "workers_per_node": distinct("workers_per_node"),
        "hostnames": len(distinct("hostname")),
        "cpu_models": distinct("cpu_model"),
        "single_commit": len(distinct("git_commit")) == 1,
        "single_config_hash": len(distinct("config_hash")) == 1,
        "all_clean": bool((dirty == False).all()),  # noqa: E712
    }


# --------------------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------------------


def mean_std(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys)[columns]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    return out.join(grouped.size().rename("runs")).reset_index()


def main_results(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std over (fold, seed) runs per (model, N): test, validation, and the clipped
    test metrics of T3c."""
    cols = [f"{s}_{m}" for s in ("test", "val") for m in ("r2", "rmse", "mae", "bias")]
    cols += ["test_clip_rmse", "test_clip_r2", "test_clip_mae", "test_frac_negative"]
    out = mean_std(runs, ["model", "n_train"], cols)
    readout = runs.groupby("model")["readout"].first()
    return out.assign(readout=out["model"].map(readout))


def per_fold(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean test RMSE over seeds per (model, N, fold) (table T3b)."""
    return (
        runs.groupby(["model", "n_train", "fold"])["test_rmse"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "test_rmse", "size": "seeds"})
        .reset_index()
    )


def pooled_errors(predictions: pd.DataFrame, n_rows: int, n_folds: int) -> pd.DataFrame:
    """Pooled out-of-fold predictions per (model, N, seed) that cover every row exactly once."""
    parts = []
    for _key, group in predictions.groupby(["model", "n_train", "seed"]):
        if group["fold"].nunique() != n_folds:
            continue
        ordered = group.sort_values("row", kind="stable")
        if not np.array_equal(ordered["row"].to_numpy(), np.arange(n_rows)):
            continue
        parts.append(ordered)
    if not parts:
        return predictions.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def pooled_seed_metrics(pooled: pd.DataFrame) -> pd.DataFrame:
    """Metrics of the pooled out-of-fold predictions of each (model, N, seed)."""
    per_seed = []
    for (model, n, seed), g in pooled.groupby(["model", "n_train", "seed"]):
        m = regression_metrics(g["actual_kw"].to_numpy(), g["pred_kw"].to_numpy())
        per_seed.append({"model": model, "n_train": n, "seed": seed, **m})
    return pd.DataFrame(per_seed)


def pooled_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per seed, then mean +- std over seeds per (model, N)."""
    if per_seed.empty:
        return pd.DataFrame(columns=["model", "n_train", "seeds"])
    out = mean_std(per_seed, ["model", "n_train"], ["r2", "rmse", "mae", "bias"])
    return out.rename(columns={"runs": "seeds"})


def best_qnn(runs: pd.DataFrame, kinds: dict[str, str], gates: dict[str, int]) -> pd.DataFrame:
    """Best QNN per readout and N by mean validation RMSE; ties to fewer gates."""
    qnn = runs[runs["model"].map(kinds) == "qnn"]
    rows = []
    for (readout, n), group in qnn.groupby(["readout", "n_train"]):
        means = group.groupby("model")["val_rmse"].mean().to_dict()
        best = select_best_qnn(means, gates)
        tied = sorted(m for m, v in means.items() if v == means[best])
        rows.append(
            {
                "readout": readout,
                "n_train": n,
                "best": best,
                "val_rmse_mean": means[best],
                "tied_with": ";".join(m for m in tied if m != best),
                "configs_compared": len(means),
            }
        )
    columns = ["readout", "n_train", "best", "val_rmse_mean", "tied_with", "configs_compared"]
    return pd.DataFrame(rows, columns=columns)


def statistical_tests(
    runs: pd.DataFrame,
    pooled: pd.DataFrame,
    best: pd.DataFrame,
    kinds: dict[str, str],
    dm_cfg: dict,
) -> pd.DataFrame:
    """Best QNN of each readout vs every baseline at each N: DM per seed (median), Wilcoxon,
    and Holm-adjusted p-values with the baselines at one (readout, N) as a family."""
    rows = []
    baselines = sorted(m for m, k in kinds.items() if k != "qnn")
    for rec in best.itertuples(index=False):
        n, qnn = rec.n_train, rec.best
        at_n = runs[runs["n_train"] == n].set_index(["model", "fold", "seed"])["test_rmse"]
        for base in baselines:
            row: dict[str, Any] = {
                "readout": rec.readout,
                "n_train": n,
                "qnn": qnn,
                "baseline": base,
            }
            if qnn in at_n.index.get_level_values(0) and base in at_n.index.get_level_values(0):
                pairs = pd.concat([at_n.loc[qnn], at_n.loc[base]], axis=1, join="inner")
                diffs = pairs.iloc[:, 0] - pairs.iloc[:, 1]
                row["wilcoxon_pairs"] = len(pairs)
                row["qnn_minus_baseline_rmse_mean"] = float(diffs.mean()) if len(pairs) else np.nan
                if len(pairs) >= 2 and (diffs != 0).any():
                    w = wilcoxon_signed_rank(
                        pairs.iloc[:, 0].to_numpy(), pairs.iloc[:, 1].to_numpy()
                    )
                    row.update(wilcoxon_statistic=w["statistic"], wilcoxon_p=w["p_value"])
            stats_per_seed = []
            p_n = pooled[pooled["n_train"] == n]
            for seed in sorted(set(p_n.loc[p_n["model"] == qnn, "seed"])):
                a = p_n[(p_n["model"] == qnn) & (p_n["seed"] == seed)].sort_values(
                    "row", kind="stable"
                )
                b = p_n[(p_n["model"] == base) & (p_n["seed"] == seed)].sort_values(
                    "row", kind="stable"
                )
                if b.empty:
                    continue
                dm = diebold_mariano(
                    a["error_kw"].to_numpy(),
                    b["error_kw"].to_numpy(),
                    horizon=int(dm_cfg["horizon"]),
                    lag=dm_cfg["hac_lag"],
                )
                stats_per_seed.append(dm)
            row["dm_seeds"] = len(stats_per_seed)
            if stats_per_seed:
                row["dm_statistic_median"] = float(
                    np.median([d["statistic"] for d in stats_per_seed])
                )
                row["dm_p_median"] = float(np.median([d["p_value"] for d in stats_per_seed]))
                row["dm_lag"] = stats_per_seed[0]["lag"]
            rows.append(row)
    columns = [
        "readout", "n_train", "qnn", "baseline", "wilcoxon_pairs",
        "qnn_minus_baseline_rmse_mean", "wilcoxon_statistic", "wilcoxon_p", "wilcoxon_p_holm",
        "dm_seeds", "dm_statistic_median", "dm_p_median", "dm_p_holm", "dm_lag",
    ]  # fmt: skip
    tests = pd.DataFrame(rows).reindex(columns=columns)
    # Holm: the baseline comparisons at one (readout, N) form one family, per test type.
    for _, index in tests.groupby(["readout", "n_train"]).groups.items():
        tests.loc[index, "wilcoxon_p_holm"] = holm(tests.loc[index, "wilcoxon_p"].to_numpy(float))
        tests.loc[index, "dm_p_holm"] = holm(tests.loc[index, "dm_p_median"].to_numpy(float))
    return tests


def protocol_comparison(main_by_protocol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """T9: test RMSE and R² under blocked and random evaluation, and random minus blocked."""
    cols = ["model", "readout", "n_train", "test_rmse_mean", "test_rmse_std", "test_r2_mean",
            "test_r2_std", "runs"]  # fmt: skip
    blocked = main_by_protocol["blocked"][cols]
    random = main_by_protocol["random"][cols]
    merged = blocked.merge(
        random, on=["model", "readout", "n_train"], suffixes=("_blocked", "_random")
    )
    merged["rmse_random_minus_blocked"] = (
        merged["test_rmse_mean_random"] - merged["test_rmse_mean_blocked"]
    )
    merged["r2_random_minus_blocked"] = (
        merged["test_r2_mean_random"] - merged["test_r2_mean_blocked"]
    )
    return merged


# --------------------------------------------------------------------------------------
# Timing, convergence, stability
# --------------------------------------------------------------------------------------


def timing_tables(
    runs: pd.DataFrame, kinds: dict[str, str], gates: dict[str, int], workers: int
) -> dict[str, pd.DataFrame]:
    """Node-load filter, fits of time per evaluation, and the same-unitary time ratios, per
    readout."""
    qnn = runs[runs["model"].map(kinds) == "qnn"].copy()
    threshold = NODE_LOAD_FRACTION * workers
    qnn["gates"] = qnn["model"].map(gates)
    qnn["included"] = qnn["node_busy_mean"].notna() & (qnn["node_busy_mean"] >= threshold)
    qnn["exclusion"] = np.where(
        qnn["node_busy_mean"].isna(),
        "no node_load",
        np.where(qnn["included"], "", f"busy_mean < {threshold:g}"),
    )
    runs_cols = [
        "readout", "model", "gates", "n_train", "fold", "seed", "time_per_evaluation",
        "total_training_time", "nfev", "node_busy_mean", "node_busy_min", "included",
        "exclusion",
    ]  # fmt: skip
    timing_runs = qnn.reindex(columns=runs_cols)

    exclusions = (
        qnn.groupby(["readout", "n_train"])
        .agg(
            runs=("included", "size"),
            included=("included", "sum"),
            no_node_load=("node_busy_mean", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    exclusions["excluded"] = exclusions["runs"] - exclusions["included"]
    exclusions["threshold_busy"] = threshold

    inc = qnn[qnn["included"]]
    fits = []
    for (readout, model), g in inc.groupby(["readout", "model"]):
        row = {
            "readout": readout,
            "model": model,
            "gates": gates[model],
            "runs": len(g),
            "sizes": g["n_train"].nunique(),
        }
        if g["n_train"].nunique() >= 2:
            row.update(
                linear_fit(g["n_train"].to_numpy(float), g["time_per_evaluation"].to_numpy())
            )
        fits.append(row)
    fits_df = pd.DataFrame(fits).reindex(
        columns=["readout", "model", "gates", "runs", "sizes", "a", "b", "r2"]
    )

    ratio_rows = []
    keyed = inc.set_index(["n_train", "fold", "seed"])
    for more, fewer in SAME_UNITARY_PAIRS:
        a = keyed[keyed["model"] == more]["time_per_evaluation"].rename("more")
        b = keyed[keyed["model"] == fewer]["time_per_evaluation"].rename("fewer")
        paired = pd.concat([a, b], axis=1, join="inner")
        for n, g in paired.groupby(level="n_train"):
            r = g["more"] / g["fewer"]
            readout = inc.loc[inc["model"] == more, "readout"].iloc[0]
            ratio_rows.append(
                {"readout": readout, "pair": f"{more} / {fewer}", "n_train": n, "pairs": len(r),
                 "ratio_mean": r.mean(), "ratio_std": r.std(ddof=1)}
            )  # fmt: skip
    ratio = pd.DataFrame(
        ratio_rows, columns=["readout", "pair", "n_train", "pairs", "ratio_mean", "ratio_std"]
    )

    by_gates = mean_std(inc, ["readout", "model", "gates", "n_train"], ["time_per_evaluation"])
    total = mean_std(qnn, ["readout", "model", "n_train"], ["total_training_time", "nfev"])
    return {
        "timing_runs": timing_runs,
        "timing_exclusions": exclusions,
        "timing_fits": fits_df,
        "timing_ratio": ratio,
        "timing_gates": by_gates,
        "timing_total": total,
    }


def convergence_table(runs: pd.DataFrame, kinds: dict[str, str]) -> pd.DataFrame:
    """Early stops, cache near hits, and best-iteration flags per (model, N)."""
    opt = runs[runs["model"].map(kinds).isin(["qnn", "mlp_pm"])].copy()
    if opt.empty:
        return pd.DataFrame()
    opt["cache_near_hits"] = pd.to_numeric(opt["cache_near_hits"]).fillna(0)
    for flag in ("stopped_before_maxiter", "best_is_initial"):
        opt[flag] = opt[flag].astype(bool).astype(int)
    return (
        opt.groupby(["readout", "model", "n_train"])
        .agg(
            runs=("seed", "size"),
            stopped_before_maxiter=("stopped_before_maxiter", "sum"),
            nit_mean=("nit", "mean"),
            nfev_mean=("nfev", "mean"),
            runs_with_near_hits=("cache_near_hits", lambda s: int((s > 0).sum())),
            cache_near_hits_total=("cache_near_hits", "sum"),
            best_is_initial=("best_is_initial", "sum"),
            messages=(
                "scipy_message",
                lambda s: "; ".join(f"{m} ({c})" for m, c in s.value_counts().items()),
            ),
        )
        .reset_index()
    )


def stability_tables(
    curves_eval: pd.DataFrame, readouts: dict[str, str], stats_cfg: dict
) -> dict[str, pd.DataFrame]:
    """Per-run SD/MS/FL over the fixed window, then SC per configuration, normalized across
    the configurations of each readout."""
    rows = []
    for (model, n, fold, seed), g in curves_eval.groupby(["model", "n_train", "fold", "seed"]):
        loss = g.sort_values("evaluation", kind="stable")["train_mse"].to_numpy()
        s = curve_stability(
            loss, int(stats_cfg["stability_start_eval"]), int(stats_cfg["stability_window"])
        )
        rows.append(
            {"readout": readouts[model], "config": model, "n_train": n, "fold": fold,
             "seed": seed, **s}
        )  # fmt: skip
    per_run = pd.DataFrame(
        rows, columns=["readout", "config", "n_train", "fold", "seed", "SD", "MS", "FL", "padded"]
    )
    parts = [
        stability_scores(g.drop(columns="readout")).assign(readout=readout)
        for readout, g in per_run.groupby("readout")
    ]
    scores = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return {"stability_runs": per_run, "stability": scores}


# --------------------------------------------------------------------------------------
# Hyperparameters and complexity
# --------------------------------------------------------------------------------------


def hyperparameter_table(source: Source) -> pd.DataFrame:
    if source.package:
        return read_package(source.results, "tuning")
    config = source.config
    rows = []
    for model in config["tuning"]["search_spaces"]:
        if model not in model_kinds(config):
            continue
        for n in config["sizes"]:
            for fold in config["folds"]["run"]:
                path = tuning_dir(source.results, model, n, fold) / "best_params.json"
                if not path.is_file():
                    continue
                content = read_json(path)
                rows.append(
                    {
                        "batch": source.name,
                        "model": model,
                        "n_train": n,
                        "fold": fold,
                        "params": json.dumps(content["params"], sort_keys=True),
                        "best_val_rmse_kw": content["best_val_rmse_kw"],
                        "trials": content["n_trials_completed"],
                    }
                )
    return pd.DataFrame(
        rows, columns=["batch", "model", "n_train", "fold", "params", "best_val_rmse_kw", "trials"]
    )


def complexity_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trainable_params",
        "tree_nodes",
        "tree_leaves",
        "support_vectors",
        "stored_training_samples",
    ]
    present = [c for c in cols if c in runs]
    numeric = runs[["model", "n_train", *present]].copy()
    for c in present:
        numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
    grouped = numeric.groupby(["model", "n_train"])
    return (
        grouped.mean()
        .add_suffix("_mean")
        .join(grouped.min().add_suffix("_min"))
        .join(grouped.max().add_suffix("_max"))
        .reset_index()
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


KIND_ORDER = ("classical", "deep", "mlp_pm", "qnn")


def model_order(sources: list[Source]) -> list[str]:
    """Every model of the sources: baselines by kind, then QNN-1..6, then QNN-1u..6u."""
    order: list[str] = []
    for kind in KIND_ORDER:
        names = [m for s in sources for m in s.config["models"].get(kind, [])]
        if kind == "qnn":
            names = sorted(names, key=lambda m: (m.endswith("u"), m))
        order += [m for m in dict.fromkeys(names) if m not in order]
    return order


def union(sources: list[Source], fn: Any) -> dict:
    out: dict = {}
    for source in sources:
        out.update(fn(source.config))
    return out


def aggregate_protocol(sources: list[Source]) -> dict[str, Any]:
    """Every aggregated table of one protocol, from one or more batches."""
    first = sources[0].config
    kinds = union(sources, model_kinds)
    readouts = union(sources, model_readouts)
    gates = union(sources, lambda c: total_gates(c["qnn"]))
    parts = [collect(s) for s in sources]
    data = {
        name: pd.concat([p[name] for p in parts], ignore_index=True)
        for name in ("runs", "completeness", "predictions", "pooled_seed", "curves_iter",
                     "curves_eval")
    }  # fmt: skip
    runs = data["runs"]
    out: dict[str, Any] = dict(data)
    out["completeness_report"] = {
        s.name: completeness_report(data["completeness"][data["completeness"]["batch"] == s.name])
        for s in sources
    }
    out["provenance"] = {
        s.name: provenance_report(runs[runs["batch"] == s.name] if not runs.empty else runs)
        for s in sources
    }
    out["meta"] = {
        "kinds": kinds,
        "readouts": readouts,
        "gates": gates,
        "order": model_order(sources),
        "batches": {s.name: {"config": s.config.source.name, "config_hash": s.config.hash}
                    for s in sources},
    }  # fmt: skip
    if runs.empty:
        return out
    n_rows = int(first["data"]["n_rows"])
    n_folds = int(first["folds"]["n_folds"])
    pooled = pooled_errors(data["predictions"], n_rows, n_folds)
    out["main"] = main_results(runs)
    out["per_fold"] = per_fold(runs)
    out["pooled"] = pooled_metrics(data["pooled_seed"])
    out["best_qnn"] = best_qnn(runs, kinds, gates)
    out["stats_tests"] = statistical_tests(
        runs, pooled, out["best_qnn"], kinds, first["stats"]["dm"]
    )
    out.update(timing_tables(runs, kinds, gates, int(first["runtime"]["workers"])))
    out["convergence"] = convergence_table(runs, kinds)
    out.update(stability_tables(data["curves_eval"], readouts, first["stats"]))
    out["hyperparameters"] = pd.concat(
        [hyperparameter_table(s) for s in sources], ignore_index=True
    )
    out["complexity"] = complexity_table(runs)
    return out


def aggregate(config: Config) -> dict[str, Any]:
    """One batch from a loaded config (results path as resolved), named after its file."""
    return aggregate_protocol([Source(config.source.stem, config, config.path("results"))])


def aggregate_all(sources: list[Source]) -> dict[str, Any]:
    """Aggregate each protocol, plus the protocol comparison when both are present."""
    protocols: dict[str, list[Source]] = {}
    for s in sources:
        protocols.setdefault(s.protocol, []).append(s)
    out: dict[str, Any] = {p: aggregate_protocol(srcs) for p, srcs in protocols.items()}
    if {"blocked", "random"} <= set(out) and all("main" in out[p] for p in ("blocked", "random")):
        out["comparison"] = protocol_comparison({p: out[p]["main"] for p in ("blocked", "random")})
    return out


CSV_TABLES = (
    "runs", "completeness", "curves_iter", "curves_eval", "main", "per_fold", "pooled",
    "best_qnn", "stats_tests", "timing_runs", "timing_exclusions", "timing_fits",
    "timing_ratio", "timing_gates", "timing_total", "convergence", "stability_runs",
    "stability", "hyperparameters", "complexity",
)  # fmt: skip


def write_outputs(root: Path, out: dict[str, Any]) -> Path:
    """Write every protocol under ``root/<protocol>`` and the comparison under ``root``."""
    for protocol, tables in out.items():
        if protocol == "comparison":
            continue
        target = root / protocol
        target.mkdir(parents=True, exist_ok=True)
        for name in CSV_TABLES:
            if name in tables:
                write_csv(target / f"{name}.csv", tables[name])
        if "predictions" in tables:
            tables["predictions"].to_csv(
                target / "predictions_test.csv.gz",
                index=False,
                lineterminator="\n",
                compression="gzip",
            )
        write_json(target / "completeness.json", tables["completeness_report"])
        write_json(target / "provenance.json", tables["provenance"])
        write_json(target / "meta.json", tables["meta"])
    if "comparison" in out:
        write_csv(root / "protocol_comparison.csv", out["comparison"])
    return root


def run(sources: list[Source], root: Path) -> dict[str, Any]:
    out = aggregate_all(sources)
    write_outputs(root, out)
    for protocol, tables in out.items():
        if protocol == "comparison":
            continue
        for batch, report in tables["completeness_report"].items():
            print(
                f"aggregate: {protocol}/{batch}: {report['found']} of {report['expected']} "
                f"expected runs -> {root / protocol}"
            )
    return out


def sources_from_args(args: argparse.Namespace) -> list[Source]:
    if args.source:
        return [make_source(name, cfg, Path(runs)) for name, cfg, runs in args.source]
    return [make_source(Path(args.config).stem, args.config, args.runs)]


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="one batch: its config")
    parser.add_argument("--runs", type=Path, help="one batch: its run directory (QNNWIND_SCRATCH)")
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("NAME", "CONFIG", "RUNS"),
        help="a batch: name, config, run directory (repeatable)",
    )


def default_root(sources: list[Source]) -> Path:
    """``<results of the first batch>/aggregated`` for one batch, else ``.../aggregated_all``."""
    name = "aggregated" if len(sources) == 1 else "aggregated_all"
    return sources[0].results / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_arguments(parser)
    parser.add_argument("--aggregated", type=Path, help="output root (default: see docs)")
    args = parser.parse_args(argv)
    use_runs_dir(args.runs)
    if not args.source and not args.config:
        parser.error("give --config (one batch) or --source (repeatable)")
    sources = sources_from_args(args)
    run(sources, args.aggregated or default_root(sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())
