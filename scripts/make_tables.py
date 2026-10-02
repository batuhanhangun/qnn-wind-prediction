"""LaTeX tables T1-T9 from the aggregated CSVs: booktabs, no vertical
rules, consistent decimals for mean +- std.

Reads the aggregation root written by scripts/aggregate.py (``blocked/*.csv``, ``random/*.csv``,
``protocol_comparison.csv``; default ``<results>/aggregated``), the dataset (T1, T1b), and the
circuit library (T2); writes ``<outputs>/tables/*.tex``. Tables T3-T8 use the blocked protocol
(the primary evaluation); T6 and T7 are also written for the random protocol when present, and
T9 compares the two protocols.

Usage: ``python scripts/make_tables.py --config configs/experiment.yaml [--aggregated DIR]
[--outputs DIR]``
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate import dataset_available  # noqa: E402
from inspect_data import descriptive, fold_table  # noqa: E402
from qnnwind.circuits import all_circuits, circuit_summary, save_entanglement_maps  # noqa: E402
from qnnwind.data import load_dataset  # noqa: E402
from qnnwind.folds import folds_from_config  # noqa: E402
from qnnwind.io import Config, load_config, write_text  # noqa: E402
from runs_dir import add_runs_argument, use_runs_dir  # noqa: E402

DASH = "--"


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def escape(text: object) -> str:
    """Escape LaTeX special characters in plain text."""
    out = str(text)
    for char, repl in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                       ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):  # fmt: skip
        out = out.replace(char, repl)
    return out


def missing(x: object) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA


def num(x: object, decimals: int) -> str:
    return DASH if missing(x) else f"{float(x):.{decimals}f}"


def sci(x: object, digits: int = 2) -> str:
    """p-values and other small numbers: fixed below 1e-3 is unreadable, use scientific."""
    if missing(x):
        return DASH
    x = float(x)
    if x == 0 or abs(x) >= 1e-3:
        return f"{x:.3f}"
    mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
    return rf"${mantissa} \times 10^{{{int(exponent)}}}$"


def pm(mean: object, std: object, decimals: int) -> str:
    """``mean ± std`` with the same decimals for both; a single run shows ``± --``."""
    if missing(mean):
        return DASH
    spread = r"\mbox{--}" if missing(std) else f"{float(std):.{decimals}f}"
    return rf"${float(mean):.{decimals}f} \pm {spread}$"


def tabular(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: str,
    caption: str,
    label: str,
    groups: Sequence[int] = (),
    long: bool = False,
    note: str = "",
) -> str:
    """A booktabs table (no vertical rules). ``groups``: row indices preceded by \\midrule."""
    head = " & ".join(header) + r" \\"
    body = []
    for i, row in enumerate(rows):
        if i in groups and i > 0:
            body.append(r"\midrule")
        # A cell starting with "[" (e.g. a readout) would be read as the optional argument of
        # the preceding row's "\\"; braces keep it literal.
        body.append(" & ".join(f"{{{c}}}" if c.startswith("[") else c for c in row) + r" \\")
    note_tex = [rf"\par\smallskip{{\footnotesize {note}}}"] if note else []
    if long:
        return "\n".join(
            [
                rf"\begin{{longtable}}{{{align}}}",
                rf"\caption{{{caption}}}\label{{{label}}} \\",
                r"\toprule", head, r"\midrule", r"\endfirsthead",
                r"\toprule", head, r"\midrule", r"\endhead",
                r"\bottomrule", r"\endlastfoot",
                *body,
                r"\end{longtable}",
                *note_tex,
                "",
            ]
        )  # fmt: skip
    return "\n".join(
        [
            r"\begin{table}[htbp]", r"\centering", r"\small",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{align}}}", r"\toprule", head, r"\midrule",
            *body,
            r"\bottomrule", r"\end{tabular}",
            *note_tex,
            r"\end{table}",
            "",
        ]
    )  # fmt: skip


def model_order(config: Config) -> list[str]:
    return [m for names in config["models"].values() for m in names]


def read(agg: Path, name: str) -> pd.DataFrame:
    path = agg / f"{name}.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def grouped_rows(frame: pd.DataFrame, key: str) -> list[int]:
    """Row indices where ``key`` changes (for \\midrule between blocks)."""
    values = frame[key].tolist()
    return [i for i in range(1, len(values)) if values[i] != values[i - 1]]


def sort_models(
    frame: pd.DataFrame, order: list[str], keys: Sequence[str] = ("n_train",)
) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(order)}
    return (
        frame.assign(_r=frame["model"].map(rank))
        .sort_values([*keys, "_r"], kind="stable")
        .drop(columns="_r")
    )


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


DATA_TABLES = ("t1_descriptive", "t1b_folds")  # computed from the dataset itself


def t1(config: Config) -> dict[str, str | None]:
    if not dataset_available(config):
        print("make_tables: T1 and T1b need the dataset (not found); skipped")
        return dict.fromkeys(DATA_TABLES)
    dataset = load_dataset(config.path("data"), config["data"])
    data_cfg = config["data"]
    columns = [*data_cfg["feature_columns"], data_cfg["target_column"]]
    frame = pd.DataFrame(np.column_stack([dataset.features, dataset.target]), columns=columns)
    stats = descriptive(frame)
    units = {"Temperature": "°C", "Pressure": "hPa", "Theta": "°", "Velocity": "m/s", "Power": "kW"}
    rows = [
        [f"{escape(r.variable)} ({units.get(r.variable, '')})"]
        + [num(getattr(r, c), 2) for c in ("mean", "median", "std", "min", "max", "range")]
        for r in stats.itertuples(index=False)
    ]
    out = {
        "t1_descriptive": tabular(
            ["Variable", "Mean", "Median", "Std", "Min", "Max", "Range"], rows, "lrrrrrr",
            f"Descriptive statistics of the dataset ({dataset.n_rows} ten-minute samples; std with "
            "ddof = 1).",
            "tab:t1",
        )
    }  # fmt: skip
    folds = folds_from_config(config["folds"], dataset.n_rows)
    fb = fold_table(frame, folds, data_cfg["target_column"], data_cfg["wind_speed_column"])
    fb = fb[fb["block"] != "buffers"]
    rows = [
        [str(r.fold), escape(r.block.replace("_", " ")), escape(r.rows).replace(" ; ", "; "),
         str(r.size), num(r.power_mean, 1), num(r.power_max, 1), num(r.speed_mean, 2),
         num(r.speed_max, 2)]
        for r in fb.itertuples(index=False)
    ]  # fmt: skip
    out["t1b_folds"] = tabular(
        ["Fold", "Block", "Rows", "Size", r"$\bar P$ (kW)", r"$P_{\max}$ (kW)",
         r"$\bar v$ (m/s)", r"$v_{\max}$ (m/s)"],
        rows, "rllrrrrr",
        rf"Blocked folds with buffers of B = {config['folds']['buffer']} rows: row ranges, "
        "and mean and maximum power and wind speed per block.",
        "tab:t1b", groups=grouped_rows(fb, "fold"),
    )  # fmt: skip
    return out


def t2(config: Config, with_unit: bool = False) -> str:
    rows = []
    for qc in all_circuits(config["qnn"]):
        s = circuit_summary(qc, qc.feature_map.num_parameters)
        g = s["gate_counts"]
        rows.append(
            [qc.name, escape(qc.entanglement)]
            + [str(g.get(k, 0)) for k in ("h", "p", "ry", "cx")]
            + [
                str(s["total_gates"]),
                str(s["depth"]),
                str(s["input_params"]),
                str(s["weight_params"]),
            ]
        )
    return tabular(
        ["Config", "Entanglement", "H", "P", "RY", "CX", "Total", "Depth", "Inputs", "Weights"],
        rows, "llrrrrrrrr",
        "QNN circuits: gate counts by type, depth, and parameter counts. QNN-1 and QNN-5 "
        "implement the same unitary."
        + (" QNN-1u to QNN-6u use the circuits of QNN-1 to QNN-6." if with_unit else ""),
        "tab:t2",
    )  # fmt: skip


PRIMARY_READOUT = "[-1, 1]"
SAME_UNITARY_PAIRS = (("QNN-1", "QNN-5"), ("QNN-1u", "QNN-5u"))
RANDOM_PREFIX = "Random cross-validation. "
SAME_UNITARY_NOTE = (
    "QNN-1 and QNN-5 implement the same unitary: their accuracy results are "
    "identical by construction; they differ only in circuit cost (T2, T5)."
)
SAME_UNITARY_NOTE_U = (
    "QNN-1 and QNN-5 implement the same unitary, and so do QNN-1u and QNN-5u: "
    "the accuracy results within each pair are identical by construction; the members of a "
    "pair differ only in circuit cost (T2, T5)."
)
UNIT_NOTE = (
    "QNN-1u to QNN-6u use the circuits of QNN-1 to QNN-6 with the target min-max scaled to "
    "[0, 1] instead of [-1, 1]."
)
T8_NOTE = (
    "Trainable parameters are fitted scalar values. SVR: the dual coefficients (one per support "
    "vector) plus the intercept; the support vectors themselves are stored training samples and "
    "are listed as a complexity measure, like the stored samples of kNN (0 trainable "
    "parameters). Tree models have no parameter count; their node and leaf counts are complexity "
    "measures. F5 plots the trainable-parameter count and omits the models without one (tree "
    "models and kNN)."
)


def readout_key(readout: str) -> tuple[bool, str]:
    """The primary readout first."""
    return (readout != PRIMARY_READOUT, readout)


def is_unit(model: str) -> bool:
    return model.startswith("QNN-") and model.endswith("u")


def same_unitary_note(frame: pd.DataFrame, column: str = "model") -> str:
    """The same-unitary note (and the [0, 1] note) for the QNN models that appear in a table."""
    present = set(frame[column]) if column in frame else set()
    pairs = [set(p) <= present for p in SAME_UNITARY_PAIRS]
    notes = []
    if pairs[0] and pairs[1]:
        notes.append(SAME_UNITARY_NOTE_U)
    elif pairs[0]:
        notes.append(SAME_UNITARY_NOTE)
    elif pairs[1]:
        notes.append(SAME_UNITARY_NOTE_U.replace("QNN-1 and QNN-5 implement the same unitary, "
                                                  "and so do", "As QNN-1 and QNN-5,"))  # fmt: skip
    if any(is_unit(m) for m in present):
        notes.append(UNIT_NOTE)
    return " ".join(notes)


def multi_readout(frame: pd.DataFrame) -> bool:
    return "readout" in frame and frame["readout"].nunique() > 1


def by_readout(frame: pd.DataFrame, order: list[str], keys: Sequence[str] = ()) -> pd.DataFrame:
    """Sort by readout (primary first), then ``keys``, then model order."""
    rank = {m: i for i, m in enumerate(order)}
    column = "model" if "model" in frame else "config"
    return (
        frame.assign(_ro=frame["readout"].map(readout_key), _r=frame[column].map(rank))
        .sort_values(["_ro", *keys, "_r"], kind="stable")
        .drop(columns=["_ro", "_r"])
        .reset_index(drop=True)
    )


def t3(agg: Path, order: list[str]) -> dict[str, str]:
    main = read(agg, "main")
    out = {}
    if not main.empty:
        main = sort_models(main, order)
        rows = [
            [str(r.n_train), escape(r.model), pm(r.test_r2_mean, r.test_r2_std, 3),
             pm(r.test_rmse_mean, r.test_rmse_std, 1), pm(r.test_mae_mean, r.test_mae_std, 1),
             str(r.runs)]
            for r in main.itertuples(index=False)
        ]  # fmt: skip
        out["t3_main"] = tabular(
            ["N", "Model", r"$R^2$", "RMSE (kW)", "MAE (kW)", "Runs"], rows, "rlrrrr",
            "Test-fold results, mean $\\pm$ std over the (fold, seed) runs.",
            "tab:t3", groups=grouped_rows(main, "n_train"), long=True,
            note=same_unitary_note(main),
        )  # fmt: skip
        if "test_clip_rmse_mean" in main:
            rows = [
                [str(r.n_train), escape(r.model), pm(r.test_clip_r2_mean, r.test_clip_r2_std, 3),
                 pm(r.test_clip_rmse_mean, r.test_clip_rmse_std, 1),
                 pm(r.test_rmse_mean, r.test_rmse_std, 1),
                 pm(100 * r.test_frac_negative_mean, 100 * r.test_frac_negative_std, 1),
                 str(r.runs)]
                for r in main.itertuples(index=False)
            ]  # fmt: skip
            out["t3c_clipped"] = tabular(
                ["N", "Model", r"$R^2$ clipped", "RMSE clipped (kW)", "RMSE (kW)",
                 r"Negative (\%)", "Runs"],
                rows, "rlrrrrr",
                "Secondary metric: test-fold $R^2$ and RMSE after clipping the predictions to "
                "[0, maximum training power] of each run, mean $\\pm$ std over the (fold, seed) "
                "runs, with the unclipped (primary) RMSE and the share of negative test "
                "predictions for reference.",
                "tab:t3c", groups=grouped_rows(main, "n_train"), long=True,
                note=same_unitary_note(main),
            )  # fmt: skip
    per_fold = read(agg, "per_fold")
    if not per_fold.empty:
        wide = per_fold.pivot_table(index=["model", "n_train"], columns="fold", values="test_rmse")
        wide = sort_models(wide.reset_index(), order)
        folds = [c for c in wide.columns if c not in ("model", "n_train")]
        rows = [
            [str(r["n_train"]), escape(r["model"])] + [num(r[f], 1) for f in folds]
            for _, r in wide.iterrows()
        ]
        out["t3b_per_fold"] = tabular(
            ["N", "Model", *[f"Fold {f}" for f in folds]], rows, "rl" + "r" * len(folds),
            "Test RMSE (kW) per fold, mean over seeds.",
            "tab:t3b", groups=grouped_rows(wide, "n_train"), long=True,
            note=same_unitary_note(wide),
        )  # fmt: skip
    return out


def t4(agg: Path, config: Config, order: list[str]) -> str | None:
    st = read(agg, "stability")
    if st.empty:
        return None
    multi = multi_readout(st)
    st = by_readout(st.sort_values(["rank", "config"], kind="stable"), order, keys=["rank"])
    rows = [
        ([escape(r.readout)] if multi else [])
        + [escape(r.config), sci(r.SD), sci(r.MS), sci(r.FL), num(r.SD_norm, 3), num(r.MS_norm, 3),
           num(r.FL_norm, 3), num(r.SC, 3), str(int(r.rank)),
           str(int(r.padded_runs)) if "padded_runs" in st else DASH]
        for r in st.itertuples(index=False)
    ]  # fmt: skip
    window = config["stats"]["stability_window"]
    start = config["stats"]["stability_start_eval"]
    per_readout = (
        " SC is normalized and ranked within each readout, because the MSE scales differ."
        if multi
        else ""
    )
    return tabular(
        (["Readout"] if multi else [])
        + ["Config", "SD", "MS", "FL", r"SD$_n$", r"MS$_n$", r"FL$_n$", "SC", "Rank", "Padded"],
        rows, ("l" if multi else "") + "lrrrrrrrrr",
        f"QNN stability score on the training loss over evaluations 1 to {window} (SD and MS "
        f"over evaluations {start + 1} to {window}); lower SC is more stable. Padded: runs that "
        f"converged before evaluation {window} (curve padded with the final loss)." + per_readout,
        "tab:t4", groups=grouped_rows(st, "readout") if multi else (),
        note=same_unitary_note(st, "config"),
    )  # fmt: skip


def t5(agg: Path, order: list[str]) -> str | None:
    fits, ratio = read(agg, "timing_fits"), read(agg, "timing_ratio")
    total, excl = read(agg, "timing_total"), read(agg, "timing_exclusions")
    if fits.empty and total.empty:
        return None
    parts = []
    if not fits.empty:
        multi = multi_readout(fits)
        fits = by_readout(fits, order)
        rows = [
            ([escape(r.readout)] if multi else [])
            + [escape(r.model), str(int(r.gates)), sci(r.a), num(r.b, 2), num(r.r2, 3),
               str(int(r.runs))]
            for r in fits.itertuples(index=False)
        ]  # fmt: skip
        parts.append(
            tabular(
                (["Readout"] if multi else [])
                + ["Config", "Gates", "a (s per sample)", "b (s)", r"$R^2$", "Runs"],
                rows, ("l" if multi else "") + "lrrrrr",
                "Least-squares fits of time per objective evaluation = a N + b (runs passing the "
                "node-load filter)."
                + (" Per readout; the circuits are identical." if multi else ""),
                "tab:t5a", groups=grouped_rows(fits, "readout") if multi else (),
            )
        )  # fmt: skip
    if not ratio.empty:
        multi = ratio["pair"].nunique() > 1
        ratio = ratio.assign(_ro=ratio["readout"].map(readout_key))
        ratio = ratio.sort_values(["_ro", "n_train"], kind="stable").reset_index(drop=True)
        rows = [
            ([escape(r.pair)] if multi else [])
            + [str(r.n_train), str(int(r.pairs)), pm(r.ratio_mean, r.ratio_std, 3)]
            for r in ratio.itertuples(index=False)
        ]
        header = ["Pair", "N", "Pairs", "Ratio"] if multi else ["N", "Pairs", "QNN-1 / QNN-5"]
        parts.append(
            tabular(
                header, rows, ("l" if multi else "") + "rrr",
                "Ratio of time per objective evaluation, QNN-1 (40 gates) over QNN-5 (34 gates)"
                + (" and QNN-1u over QNN-5u (the same circuits)" if multi else "")
                + ", paired by (fold, seed).",
                "tab:t5b", groups=grouped_rows(ratio, "pair") if multi else (),
            )
        )  # fmt: skip
    if not total.empty:
        total = sort_models(total, order)
        rows = [
            [str(r.n_train), escape(r.model),
             pm(r.total_training_time_mean / 3600, r.total_training_time_std / 3600, 2),
             pm(r.nfev_mean, r.nfev_std, 1), str(int(r.runs))]
            for r in total.itertuples(index=False)
        ]  # fmt: skip
        parts.append(
            tabular(
                ["N", "Config", "Training time (h)", "Evaluations", "Runs"], rows, "rlrrr",
                "Total QNN training time and objective evaluations (descriptive; not fitted).",
                "tab:t5c", groups=grouped_rows(total, "n_train"),
                note=UNIT_NOTE if any(is_unit(m) for m in total["model"]) else "",
            )
        )  # fmt: skip
    if not excl.empty:
        multi = multi_readout(excl)
        if multi:
            excl = excl.assign(_ro=excl["readout"].map(readout_key))
            excl = excl.sort_values(["_ro", "n_train"], kind="stable").reset_index(drop=True)
        rows = [
            ([escape(r.readout)] if multi else [])
            + [str(r.n_train), str(int(r.runs)), str(int(r.included)), str(int(r.excluded)),
               str(int(r.no_node_load))]
            for r in excl.itertuples(index=False)
        ]  # fmt: skip
        threshold = excl["threshold_busy"].iloc[0]
        parts.append(
            tabular(
                (["Readout"] if multi else [])
                + ["N", "QNN runs", "Included", "Excluded", "No node load"],
                rows,
                ("l" if multi else "") + "rrrrr",
                f"Node-load filter for the timing fits: runs with mean busy workers below "
                f"{threshold:g} or without a node-load record are excluded.",
                "tab:t5d",
                groups=grouped_rows(excl, "readout") if multi else (),
            )
        )
    return "\n".join(parts)


def t6(agg: Path, order: list[str], label: str = "tab:t6", prefix: str = "") -> str | None:
    tests = read(agg, "stats_tests")
    if tests.empty:
        return None
    multi = multi_readout(tests)
    tests = by_readout(tests.rename(columns={"baseline": "model"}), order, keys=["n_train"])
    rows = [
        ([escape(r.readout)] if multi else [])
        + [str(r.n_train), escape(r.qnn), escape(r.model), num(r.qnn_minus_baseline_rmse_mean, 1),
           num(r.wilcoxon_statistic, 1), sci(r.wilcoxon_p), sci(r.wilcoxon_p_holm),
           str(int(r.wilcoxon_pairs)), num(r.dm_statistic_median, 2), sci(r.dm_p_median),
           sci(r.dm_p_holm), str(int(r.dm_seeds))]
        for r in tests.itertuples(index=False)
    ]  # fmt: skip
    holm = r"^{\mathrm{Holm}}"
    family = "at each N and readout" if multi else "at each N"
    keys = tests["readout"].astype(str) + "|" + tests["n_train"].astype(str)
    return tabular(
        (["Readout"] if multi else [])
        + ["N", "QNN", "Baseline", r"$\Delta$RMSE", "W", "$p_W$", f"$p_W{holm}$", "Pairs", "DM",
           "$p_{DM}$", f"$p_{{DM}}{holm}$", "Seeds"],
        rows, ("l" if multi else "") + "rllrrrrrrrrr",
        prefix
        + ("Best QNN of each readout" if multi else "Best QNN")
        + " (by validation) against each baseline. $\\Delta$RMSE: mean test RMSE "
        "difference (QNN minus baseline, kW). Wilcoxon signed-rank on the paired (fold, seed) "
        "test RMSEs. Diebold-Mariano on pooled out-of-fold squared errors with the "
        "Harvey-Leybourne-Newbold correction and a Bartlett HAC lag of 72: median statistic and "
        "p-value over seeds (negative: the QNN has lower loss). Holm: step-down adjusted "
        f"p-values, with the baseline comparisons {family} as one family (per test).",
        label, groups=[i for i in range(1, len(keys)) if keys[i] != keys[i - 1]], long=True,
        note=same_unitary_note(tests, "qnn") if multi else "",
    )  # fmt: skip


def t7(agg: Path, order: list[str], label: str = "tab:t7", prefix: str = "") -> str | None:
    hp = read(agg, "hyperparameters")
    if hp.empty:
        return None
    hp = sort_models(hp, order, keys=())
    hp = hp.sort_values(["n_train"], kind="stable")

    def fmt(params: str) -> str:
        items = json.loads(params).items()
        return escape(
            ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in items)
        )

    rows = [
        [escape(r.model), str(r.n_train), str(r.fold), fmt(r.params), num(r.best_val_rmse_kw, 1)]
        for r in hp.itertuples(index=False)
    ]
    return tabular(
        ["Model", "N", "Fold", "Selected hyperparameters", "Val. RMSE"], rows,
        r"lrrp{0.55\linewidth}r",
        prefix + "Hyperparameters selected by Optuna TPE (30 trials, seed-0 training subset) per "
        "model, N, and fold, with the best validation RMSE (kW).",
        label, groups=grouped_rows(hp, "n_train"), long=True,
    )  # fmt: skip


def t8(agg: Path, order: list[str], focus_n: int) -> str | None:
    cx = read(agg, "complexity")
    if cx.empty:
        return None
    at = cx[cx["n_train"] == focus_n]
    if at.empty:
        at = cx[cx["n_train"] == cx["n_train"].max()]
        focus_n = int(cx["n_train"].max())
    at = sort_models(at, order, keys=())

    def span(r: pd.Series, col: str, decimals: int = 0) -> str:
        if missing(r.get(f"{col}_mean")):
            return DASH
        lo, hi = r[f"{col}_min"], r[f"{col}_max"]
        mean = num(r[f"{col}_mean"], decimals)
        return mean if lo == hi else f"{mean} ({num(lo, 0)}--{num(hi, 0)})"

    rows = []
    for _, r in at.iterrows():
        params = span(r, "trainable_params")
        measure = DASH
        if not missing(r.get("tree_nodes_mean")):
            measure = f"nodes {span(r, 'tree_nodes')}; leaves {span(r, 'tree_leaves')}"
        elif not missing(r.get("support_vectors_mean")):
            measure = f"support vectors {span(r, 'support_vectors')}"
        elif not missing(r.get("stored_training_samples_mean")):
            measure = f"stored samples {span(r, 'stored_training_samples')}"
        rows.append([escape(r["model"]), params, escape(measure)])
    return tabular(
        ["Model", "Trainable parameters", "Complexity measure (not parameters)"], rows, "lrl",
        f"Trainable parameter counts and, for tree and kernel models, complexity measures at "
        f"N = {focus_n}: mean over runs (range in parentheses when it varies).",
        "tab:t8",
        note=" ".join(filter(None, [T8_NOTE, same_unitary_note(at)])),
    )  # fmt: skip


def t9(agg_root: Path, order: list[str]) -> str | None:
    """Blocked vs random test RMSE and R² per model and N."""
    cmp = read(agg_root, "protocol_comparison")
    if cmp.empty:
        return None
    cmp = sort_models(cmp, order)
    rows = [
        [str(r.n_train), escape(r.model),
         pm(r.test_rmse_mean_blocked, r.test_rmse_std_blocked, 1),
         pm(r.test_rmse_mean_random, r.test_rmse_std_random, 1),
         num(r.rmse_random_minus_blocked, 1),
         pm(r.test_r2_mean_blocked, r.test_r2_std_blocked, 3),
         pm(r.test_r2_mean_random, r.test_r2_std_random, 3),
         num(r.r2_random_minus_blocked, 3)]
        for r in cmp.itertuples(index=False)
    ]  # fmt: skip
    return tabular(
        ["N", "Model", "RMSE blocked", "RMSE random", r"$\Delta$RMSE", r"$R^2$ blocked",
         r"$R^2$ random", r"$\Delta R^2$"],
        rows, "rlrrrrrr",
        "Protocol comparison: test RMSE (kW) and $R^2$ under blocked cross-validation "
        "(primary) and under random cross-validation, mean $\\pm$ std over "
        "the (fold, seed) runs; $\\Delta$: random minus blocked.",
        "tab:t9", groups=grouped_rows(cmp, "n_train"), long=True,
        note=same_unitary_note(cmp),
    )  # fmt: skip


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def read_order(agg_root: Path, config: Config) -> list[str]:
    """The model order of the aggregation (meta.json), else the config's order."""
    order: list[str] = []
    for protocol in ("blocked", "random"):
        path = agg_root / protocol / "meta.json"
        if path.is_file():
            order += [m for m in json.loads(path.read_text(encoding="utf-8"))["order"]
                      if m not in order]  # fmt: skip
    return order or model_order(config)


def run(config: Config, outputs: Path, agg_root: Path | None = None) -> list[Path]:
    """Write every table. ``agg_root`` holds ``blocked/`` and/or ``random/`` (and
    ``protocol_comparison.csv``); default ``<results>/aggregated``."""
    agg_root = agg_root or config.path("results") / "aggregated"
    blocked, random = agg_root / "blocked", agg_root / "random"
    order = read_order(agg_root, config)
    focus_n = int(max(config["sizes"]))
    tables_dir = outputs / "tables"
    tables: dict[str, str | None] = {}
    tables.update(t1(config))
    tables["t2_circuits"] = t2(config, any(is_unit(m) for m in order))
    tables.update(t3(blocked, order))
    tables["t4_stability"] = t4(blocked, config, order)
    tables["t5_timing"] = t5(blocked, order)
    tables["t6_tests"] = t6(blocked, order)
    tables["t6_tests_random"] = t6(random, order, "tab:t6r", RANDOM_PREFIX)
    tables["t7_hyperparameters"] = t7(blocked, order)
    tables["t7_hyperparameters_random"] = t7(random, order, "tab:t7r", RANDOM_PREFIX)
    tables["t8_complexity"] = t8(blocked, order, focus_n)
    tables["t9_protocols"] = t9(agg_root, order)
    written = []
    for name, tex in tables.items():
        path = tables_dir / f"{name}.tex"
        if tex:
            write_text(path, tex)
            written.append(path)
        elif name in DATA_TABLES and path.is_file():  # a copy provided without the dataset
            written.append(path)
    save_entanglement_maps(tables_dir / "entanglement_maps.json", config["qnn"])
    write_preview(tables_dir, written)
    print(f"make_tables: {len(written)} tables -> {tables_dir}")
    return written


PREVIEW_PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[margin=2cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\begin{document}
"""


def write_preview(tables_dir: Path, tables: list[Path]) -> Path:
    """``tables_preview.tex``: a standalone document that inputs every table."""
    body = "".join(f"\\input{{{t.name}}}\n\\clearpage\n" for t in tables)
    path = tables_dir / "tables_preview.tex"
    write_text(path, PREVIEW_PREAMBLE + body + "\\end{document}\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--aggregated", type=Path, help="default: <results>/aggregated")
    parser.add_argument("--outputs", type=Path, help="default: paths.outputs of the config")
    add_runs_argument(parser)
    args = parser.parse_args(argv)
    use_runs_dir(args.runs)
    config = load_config(args.config)
    run(config, args.outputs or config.path("outputs"), args.aggregated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
