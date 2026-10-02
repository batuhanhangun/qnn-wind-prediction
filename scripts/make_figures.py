"""Figures F0-F6 and A1-A5 from the aggregated CSVs.

Style: vector PDF with embedded TrueType fonts (``pdf.fonttype`` 42) plus a 300 dpi PNG,
width 3.5 in (single column) or 7.2 in (double column), 8 pt base font, no in-figure titles,
one fixed colorblind-safe color and marker per model in every figure, shared axes where panels
are compared. Figures are drawn only from the aggregation root written by scripts/aggregate.py
(``blocked/``, ``random/``, ``protocol_comparison.csv``; never by re-running experiments),
except A4, which draws the circuits from the Qiskit circuit library, and F0 (evaluation
layout), which draws the folds from qnnwind.folds via the configs. F1-F5 and A1-A3, A5 use
the blocked protocol (the primary evaluation); F6 compares the two protocols.

Colors (validated categorical palette, composite color + marker encoding): QNN-1..6 take slots
1-6 (six distinct hues for the QNN-only figures); QNN-1u..6u (target in [0, 1]) take
the color and marker of their circuit with an open marker and a dashed line; the classical
models share violet and the deep models (with MLP-PM) share red, told apart by marker.

Usage: ``python scripts/make_figures.py --config configs/experiment.yaml [--aggregated DIR]
[--outputs DIR] [--random-config configs/random.yaml]``
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate import dataset_available, pooled_errors  # noqa: E402
from qnnwind.circuits import all_circuits  # noqa: E402
from qnnwind.data import load_dataset  # noqa: E402
from qnnwind.folds import Fold  # noqa: E402
from qnnwind.io import Config, load_config  # noqa: E402
from runs_dir import add_runs_argument, use_runs_dir  # noqa: E402

SINGLE, DOUBLE = 3.5, 7.2  # inches
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
QNN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLASSICAL, DEEP = "#4a3aa7", "#e34948"
MARKERS = ["o", "s", "^", "D", "v", "P"]
PRIMARY_READOUT, UNIT_READOUT = "[-1, 1]", "[0, 1]"
# F6: the two protocols (slots 1 and 2 of the validated palette).
PROTOCOL_COLORS = {"blocked": QNN_COLORS[0], "random": QNN_COLORS[1]}

STYLE = {
    **{f"QNN-{i + 1}": (QNN_COLORS[i], MARKERS[i]) for i in range(6)},
    **{
        m: (CLASSICAL, MARKERS[i])
        for i, m in enumerate(["LR", "kNN", "DTR", "SVR", "XGBoost", "LightGBM"])
    },
    **{
        m: (DEEP, mk)
        for m, mk in zip(["MLP", "LSTM", "GRU", "Transformer", "MLP-PM"], "os^DX", strict=True)
    },
}

# QNN-1 and QNN-5 implement the same unitary, and so do QNN-1u and QNN-5u.
SAME_UNITARY = ("QNN-1", "QNN-5")
MERGED = "QNN-1/QNN-5 (same unitary)"
MERGED_U = "QNN-1u/QNN-5u (same unitary)"
PAIRS = ((SAME_UNITARY, MERGED, ""), (("QNN-1u", "QNN-5u"), MERGED_U, " (u)"))
SHORT = {MERGED: "QNN-1/QNN-5", MERGED_U: "QNN-1u/QNN-5u"}
BEST = {PRIMARY_READOUT: "Best QNN, [-1, 1]", UNIT_READOUT: "Best QNN, [0, 1]"}


def is_unit(model: str) -> bool:
    """A [0, 1]-readout QNN series (QNN-ku, the merged u pair, or the best [0, 1] QNN)."""
    return model in (MERGED_U, BEST[UNIT_READOUT]) or (
        model.startswith("QNN-") and model.endswith("u")
    )


def base_style(model: str) -> tuple[str, str]:
    """Color and marker: a u model takes those of its circuit."""
    if model in (MERGED, MERGED_U) or model in BEST.values():
        return STYLE["QNN-1"]
    return STYLE[model[:-1] if is_unit(model) else model]


def short(model: str) -> str:
    return SHORT.get(model, model)


def accuracy_order(order: list[str]) -> list[str]:
    """Model order for accuracy figures: each merged same-unitary series takes the place of the
    pair's first member (the members stay listed, in case their data differ)."""
    out = []
    for m in order:
        for (first, _), merged, _ in PAIRS:
            if m == first:
                out.append(merged)
        out.append(m)
    return out


def merge_same_unitary(
    frame: pd.DataFrame, keys: list[str], values: list[str], notes: dict, figure: str
) -> pd.DataFrame:
    """Accuracy figures draw the members of a same-unitary pair as one series.

    For each pair (QNN-1/QNN-5, QNN-1u/QNN-5u): if both are present, their values are first
    checked to be identical on ``keys``; only then is the first dropped and the second
    relabelled. A single member is relabelled as well. If the data differ, both stay separate
    and the check result is recorded in ``notes`` (key: figure, plus " (u)" for the u pair).
    """
    if frame.empty or "model" not in frame:
        return frame
    for pair, merged, suffix in PAIRS:
        present = [m for m in pair if m in set(frame["model"])]
        if not present:
            continue
        if len(present) == 2:
            a, b = (frame[frame["model"] == m].set_index(keys)[values].sort_index() for m in pair)
            identical = a.index.equals(b.index) and np.allclose(
                a.to_numpy(float), b.to_numpy(float), rtol=0.0, atol=1e-9, equal_nan=True
            )
            notes.setdefault("same_unitary_identical", {})[figure + suffix] = bool(identical)
            if not identical:
                continue
            frame = frame[frame["model"] != pair[0]]
        frame = frame.assign(model=frame["model"].replace({m: merged for m in pair}))
    return frame


def merged_name(model: str, present: set[str]) -> str:
    """The name under which ``model`` appears after merging."""
    for pair, merged, _ in PAIRS:
        if model in pair and merged in present:
            return merged
    return model


def apply_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "errorbar.capsize": 1.5,
            "savefig.dpi": 300,
            "figure.dpi": 100,
        }
    )


def style(model: str, label: str | None = None) -> dict[str, Any]:
    """Color, marker, and label; u series get an open (white-filled) marker."""
    color, marker = base_style(model)
    out = {"color": color, "marker": marker, "label": model if label is None else label}
    if is_unit(model):
        out["markerfacecolor"] = "white"
    return out


def dash(model: str) -> str:
    return "--" if is_unit(model) else "-"


def save(fig: plt.Figure, figures: Path, name: str) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    paths = [figures / f"{name}.pdf", figures / f"{name}.png"]
    for path in paths:
        fig.savefig(
            path, dpi=300, metadata={"CreationDate": None} if path.suffix == ".pdf" else None
        )
    plt.close(fig)
    return paths


def read(agg: Path, name: str, **kwargs: Any) -> pd.DataFrame:
    path = agg / name
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ordered(models: list[str], present: set[str]) -> list[str]:
    return [m for m in models if m in present]


def readouts_of(frame: pd.DataFrame) -> list[str]:
    """The readouts present, the primary one first."""
    if "readout" not in frame:
        return [PRIMARY_READOUT]
    return sorted(frame["readout"].dropna().unique(), key=lambda r: (r != PRIMARY_READOUT, r))


def label_orphans(axes: np.ndarray) -> list[plt.Axes]:
    """With shared x-axes, show x tick labels on every visible panel that has no visible panel
    below it (e.g. in a partly filled last row); returns those bottom-most panels."""
    rows, cols = axes.shape
    bottom = []
    for j in range(cols):
        for i in range(rows):
            below = [axes[k][j] for k in range(i + 1, rows) if axes[k][j].get_visible()]
            if axes[i][j].get_visible() and not below:
                axes[i][j].xaxis.set_tick_params(labelbottom=True)
                bottom.append(axes[i][j])
    return bottom


def legend_below(fig: plt.Figure, handles: list, labels: list[str], ncol: int) -> None:
    fig.legend(
        handles, labels, loc="outside lower center", ncol=ncol, handlelength=1.6, columnspacing=1.0
    )


def legend_below_fit(fig: plt.Figure, handles: list, labels: list[str]) -> None:
    """Legend below the axes with the most columns whose width fits inside the figure."""
    width = fig.get_figwidth() * fig.dpi
    for ncol in range(len(labels), 0, -1):
        legend_below(fig, handles, labels, ncol=ncol)
        legend = fig.legends[-1]
        fig.canvas.draw()
        if legend.get_window_extent().width <= width or ncol == 1:
            return
        legend.remove()


def _errorbar_panels(
    axes: np.ndarray, main: pd.DataFrame, models: list[str], labels: dict[str, str]
) -> None:
    """RMSE (top) and R² (bottom) against N, one series per model."""
    for model in models:
        g = main[main["model"] == model].sort_values("n_train", kind="stable")
        for ax, metric in zip(axes, ("rmse", "r2"), strict=True):
            ax.errorbar(
                g["n_train"],
                g[f"test_{metric}_mean"],
                yerr=g[f"test_{metric}_std"],
                markersize=4,
                linewidth=1.0,
                linestyle=dash(model),
                **style(model, labels.get(model)),
            )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

# F0: role of every sample in a fold. Every categorical slot belongs to a model, so the roles
# are told apart by lightness alone (warm neutrals, light to dark; safe under every color-
# vision deficiency), and the unused buffer rows are hatched as well.
ROLES = ("training pool", "validation", "buffer (unused)", "test")
ROLE_STYLE = {
    "training pool": {"facecolor": "#d9d7cf"},
    "validation": {"facecolor": "#8f8c84"},
    "buffer (unused)": {"facecolor": "#fcfcfb", "hatch": "//////", "edgecolor": "#8f8c84"},
    "test": {"facecolor": "#2f2e2b"},
}


def fold_roles(fold: Fold, n_rows: int) -> np.ndarray:
    """Index into ROLES of every row 0..n_rows-1 in ``fold`` (each row has exactly one role)."""
    roles = np.full(n_rows, -1, dtype=np.int64)
    parts = (fold.train_pool, fold.validation, fold.buffers, fold.test)
    for index, rows in enumerate(parts):
        if np.any(roles[rows] != -1):
            raise ValueError(f"fold {fold.k}: a row has more than one role")
        roles[rows] = index
    if np.any(roles == -1):
        raise ValueError(f"fold {fold.k}: rows without a role")
    return roles


def _role_bar(ax: plt.Axes, y: float, roles: np.ndarray, gid: str) -> None:
    """One horizontal bar at ``y``: sample i covers [i - 0.5, i + 0.5), colored by its role.
    Each role is one collection of its contiguous runs, with gid ``<gid>:<role index>``."""
    starts = np.flatnonzero(np.diff(roles, prepend=-2) != 0)
    stops = np.append(starts[1:], roles.size)
    for index, role in enumerate(ROLES):
        mask = roles[starts] == index
        if not mask.any():
            continue
        spans = [(s - 0.5, e - s) for s, e in zip(starts[mask], stops[mask], strict=True)]
        bars = ax.broken_barh(
            spans, (y - 0.4, 0.8), linewidth=0, antialiased=False, **ROLE_STYLE[role]
        )
        bars.set_gid(f"{gid}:{index}")


def f0_figure(blocked: list[Fold], random_fold: Fold | None, n_rows: int) -> plt.Figure:
    """(a) blocked protocol, one bar per fold (fold 0 at the top); (b) random protocol, fold
    0 (if given). Both are drawn from Fold objects, never from hardcoded row ranges."""
    panels = [("(a) Blocked cross-validation", "blocked", blocked)]
    if random_fold is not None:
        panels.append(("(b) Random cross-validation", "random", [random_fold]))
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        squeeze=False,
        figsize=(DOUBLE, 2.6),
        layout="constrained",
        gridspec_kw={"height_ratios": [len(f) + 0.6 for _, _, f in panels]},
    )
    for ax, (label, protocol, folds) in zip(axes[:, 0], panels, strict=True):
        for y, fold in enumerate(folds):
            _role_bar(ax, y, fold_roles(fold, n_rows), f"F0:{protocol}:{fold.k}")
        ax.set_ylim(len(folds) - 0.5, -0.5)  # fold 0 at the top
        ax.set_yticks(range(len(folds)), [f"Fold {f.k}" for f in folds])
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.set_title(label, loc="left", color=INK, fontsize=8)
    axes[-1][0].set_xlim(-0.5, n_rows - 0.5)
    axes[-1][0].set_xlabel("Sample index (row order of the dataset)")
    handles = [plt.Rectangle((0, 0), 1, 1, linewidth=0, **ROLE_STYLE[r]) for r in ROLES]
    legend_below(fig, handles, list(ROLES), ncol=len(ROLES))
    return fig


def f0(config: Config, random_config: Config | None, figures: Path, notes: dict) -> list[Path]:
    """Evaluation layout (fold roles of every sample) from the fold functions of the configs."""
    n_rows = int(config["data"]["n_rows"])
    blocked = config.folds(n_rows)
    random_fold = random_config.folds(n_rows)[0] if random_config is not None else None
    notes["F0"] = {
        "blocked_config": config.source.name,
        "random_config": random_config.source.name if random_config is not None else None,
    }
    return save(f0_figure(blocked, random_fold, n_rows), figures, "F0_evaluation_layout")


def best_series(
    main: pd.DataFrame, best: pd.DataFrame, notes: dict
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pseudo-series "Best QNN, <readout>": at each N, the main row of that N's best QNN."""
    rows, labels = [], {}
    for readout in readouts_of(best):
        chosen = best[best["readout"] == readout].sort_values("n_train", kind="stable")
        if chosen.empty:
            continue
        name = BEST.get(readout, f"Best QNN, {readout}")
        for rec in chosen.itertuples(index=False):
            row = main[(main["model"] == rec.best) & (main["n_train"] == rec.n_train)]
            rows.append(row.assign(model=name))
        configs = chosen["best"].tolist()
        notes.setdefault("best_qnn_by_n", {})[readout] = dict(
            zip(chosen["n_train"].astype(int).tolist(), configs, strict=True)
        )
        detail = configs[0] if len(set(configs)) == 1 else "by N: " + ", ".join(configs)
        labels[name] = f"{name} ({detail})"
    frame = pd.concat(rows, ignore_index=True) if rows else main.iloc[0:0]
    return frame, labels


def f1(
    agg: Path, figures: Path, order: list[str], kinds: dict[str, str], notes: dict
) -> list[Path]:
    """All baselines plus the best QNN of each readout."""
    main, best = read(agg, "main.csv"), read(agg, "best_qnn.csv")
    if main.empty:
        return []
    baselines = main[main["model"].map(kinds) != "qnn"]
    series, labels = best_series(main, best, notes) if not best.empty else (main.iloc[0:0], {})
    frame = pd.concat([baselines, series], ignore_index=True)
    best_names = [m for m in BEST.values() if m in set(series["model"])]
    models = ordered(order, set(baselines["model"])) + best_names
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(DOUBLE, 5.2), layout="constrained")
    _errorbar_panels(axes, frame, models, labels)
    axes[0].set_ylabel("Test RMSE (kW)")
    axes[1].set_ylabel(r"Test $R^2$")
    axes[1].set_xlabel("Training size N")
    axes[1].set_xticks(sorted(main["n_train"].unique()))
    handles, labels_ = axes[0].get_legend_handles_labels()
    legend_below(fig, handles, labels_, ncol=5 if series.empty else 4)
    return save(fig, figures, "F1_rmse_r2_vs_n")


def a5(
    agg: Path, figures: Path, order: list[str], kinds: dict[str, str], notes: dict
) -> list[Path]:
    """All QNN configurations of both readouts, in the style of F1 (one column per readout)."""
    main = read(agg, "main.csv")
    if main.empty:
        return []
    qnn = main[main["model"].map(kinds) == "qnn"]
    if qnn.empty:
        return []
    values = ["test_rmse_mean", "test_rmse_std", "test_r2_mean", "test_r2_std"]
    qnn = merge_same_unitary(qnn, ["n_train"], values, notes, "A5")
    readouts = readouts_of(qnn)
    fig, axes = plt.subplots(
        2,
        len(readouts),
        sharex=True,
        sharey="row",
        squeeze=False,
        figsize=(DOUBLE, 5.2),
        layout="constrained",
    )
    handles, labels = [], []
    for j, readout in enumerate(readouts):
        at = qnn[qnn["readout"] == readout]
        _errorbar_panels(axes[:, j], at, ordered(order, set(at["model"])), {})
        axes[0][j].set_title(f"Readout {readout}", color=INK)
        axes[1][j].set_xlabel("Training size N")
        axes[1][j].set_xticks(sorted(main["n_train"].unique()))
        h, lab = axes[0][j].get_legend_handles_labels()
        handles += h
        labels += lab
    axes[0][0].set_ylabel("Test RMSE (kW)")
    axes[1][0].set_ylabel(r"Test $R^2$")
    legend_below(fig, handles, labels, ncol=5)
    return save(fig, figures, "A5_qnn_rmse_r2_vs_n")


BAND_QUANTILES = (0.1, 0.9)  # F2: band over (fold, seed) runs around the median line
F2_YLIM_FACTORS = (0.7, 1.3)  # F2: y-limits of a row = these x (lowest, highest band value)
F5_DODGE = 1.04  # F5: QNN x positions 12 / F5_DODGE ([-1, 1]) and 12 * F5_DODGE ([0, 1])


def _bands(
    ax: plt.Axes, curves: pd.DataFrame, model: str, column: str, dashed: bool
) -> tuple[float, float]:
    """Median line and 10th-90th percentile band over (fold, seed) runs, per iteration (a
    log axis cannot show mean - std bands that reach zero); returns the lowest and highest
    band values."""
    by_iter = curves.groupby("iteration")[column]
    lo_q, hi_q = BAND_QUANTILES
    stats = pd.DataFrame(
        {"median": by_iter.median(), "lo": by_iter.quantile(lo_q), "hi": by_iter.quantile(hi_q)}
    ).reset_index()
    color, _ = base_style(model)
    ax.plot(
        stats["iteration"],
        stats["median"],
        color=color,
        linestyle="--" if dashed else "-",
        linewidth=1.0,
        label=short(model) if not dashed else None,
    )
    ax.fill_between(
        stats["iteration"], stats["lo"], stats["hi"], color=color, alpha=0.15, linewidth=0
    )
    return float(stats["lo"].min()), float(stats["hi"].max())


def with_readout(frame: pd.DataFrame, readouts: dict[str, str]) -> pd.DataFrame:
    """Add the readout column to curve tables (from the model's readout)."""
    if frame.empty or "readout" in frame:
        return frame
    return frame.assign(readout=frame["model"].map(readouts).fillna(PRIMARY_READOUT))


def f2(
    agg: Path, figures: Path, order: list[str], readouts: dict[str, str], notes: dict
) -> list[Path]:
    """Training (solid) and validation (dashed) MSE per iteration; one row per readout."""
    curves = with_readout(read(agg, "curves_iter.csv"), readouts)
    if curves.empty:
        return []
    keys = ["n_train", "fold", "seed", "iteration"]
    curves = merge_same_unitary(curves, keys, ["train_mse", "val_mse"], notes, "F2")
    sizes = sorted(curves["n_train"].unique())
    rows = readouts_of(curves)
    fig, axes = plt.subplots(
        len(rows),
        len(sizes),
        sharex=True,
        sharey="row",
        squeeze=False,
        figsize=(DOUBLE, 2.3 * len(rows) + 0.3),
        layout="constrained",
    )
    legend: dict[str, Any] = {}
    for i, readout in enumerate(rows):
        extent = []
        for ax, n in zip(axes[i], sizes, strict=True):
            at = curves[(curves["n_train"] == n) & (curves["readout"] == readout)]
            for model in ordered(order, set(at["model"])):
                g = at[at["model"] == model]
                extent.append(_bands(ax, g, model, "train_mse", dashed=False))
                extent.append(_bands(ax, g, model, "val_mse", dashed=True))
            ax.set_yscale("log")
            if i == len(rows) - 1:
                ax.set_xlabel(f"Iteration (N = {n})")
        low, high = min(e[0] for e in extent), max(e[1] for e in extent)
        if low > 0:  # shared within the row
            axes[i][0].set_ylim(F2_YLIM_FACTORS[0] * low, F2_YLIM_FACTORS[1] * high)
        axes[i][0].set_ylabel(f"MSE, target in {readout}")
        for h, lab in zip(*axes[i][0].get_legend_handles_labels(), strict=True):
            legend.setdefault(lab.replace("u", "") if lab.startswith("QNN-") else lab, h)
    handles, labels = list(legend.values()), list(legend.keys())
    handles += [
        plt.Line2D([], [], color=INK_2, linestyle="-"),
        plt.Line2D([], [], color=INK_2, linestyle="--"),
    ]
    labels += ["training", "validation"]
    legend_below(fig, handles, labels, ncol=8)
    notes["F2_band"] = {"quantiles": list(BAND_QUANTILES), "ylim_factors": list(F2_YLIM_FACTORS)}
    return save(fig, figures, "F2_qnn_mse_vs_iteration")


def f3(agg: Path, figures: Path, order: list[str], notes: dict) -> list[Path]:
    """Timing, primary readout only (the circuits of both readouts are identical)."""
    runs = read(agg, "timing_runs.csv")
    fits = read(agg, "timing_fits.csv")
    if runs.empty:
        return []
    readout = readouts_of(runs)[0]
    notes["F3_readout"] = readout
    if "readout" in runs:
        runs = runs[runs["readout"] == readout]
        fits = fits[fits["readout"] == readout] if not fits.empty else fits
    inc = runs[runs["included"].astype(bool)]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, sharey=True, figsize=(DOUBLE, 2.8), layout="constrained")
    for model in ordered(order, set(inc["model"])):
        g = (
            inc[inc["model"] == model]
            .groupby("n_train")["time_per_evaluation"]
            .agg(["mean", "std"])
        )
        ax_a.errorbar(g.index, g["mean"], yerr=g["std"], linestyle="none", **style(model))
        fit = fits[fits["model"] == model] if not fits.empty else fits
        if not fit.empty and pd.notna(fit["a"].iloc[0]):
            xs = np.linspace(min(g.index), max(g.index), 50)
            ax_a.plot(
                xs,
                fit["a"].iloc[0] * xs + fit["b"].iloc[0],
                color=base_style(model)[0],
                linewidth=0.9,
            )
    ax_a.set_xlabel("Training size N")
    ax_a.set_ylabel("Time per objective evaluation (s)")
    per = inc.groupby(["model", "gates", "n_train"])["time_per_evaluation"].mean().reset_index()
    for model in ordered(order, set(per["model"])):
        g = per[per["model"] == model]
        color, marker = base_style(model)
        ax_b.scatter(
            g["gates"], g["time_per_evaluation"], color=color, marker=marker, s=18, zorder=3
        )
    for n, g in per.groupby("n_train"):
        g = g.sort_values("gates", kind="stable")
        ax_b.plot(g["gates"], g["time_per_evaluation"], color=GRID, linewidth=0.8, zorder=1)
        ax_b.annotate(
            f"N = {n}",
            (g["gates"].iloc[0], g["time_per_evaluation"].iloc[0]),
            xytext=(-8, 0),
            ha="right",
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=7,
        )
    ax_b.set_xlabel("Total gates")
    if not per.empty:
        ax_b.set_xticks(sorted(per["gates"].unique()))
        ax_b.set_xlim(per["gates"].min() - 3.5, per["gates"].max() + 1)  # room for N labels
    handles, labels = ax_a.get_legend_handles_labels()
    legend_below(fig, handles, labels, ncol=6)
    return save(fig, figures, "F3_qnn_timing")


def _errors_for(
    pred: pd.DataFrame, pooled: pd.DataFrame, model: str, n: int
) -> tuple[np.ndarray, bool]:
    """Pooled out-of-fold errors (all seeds) if complete, else the available test-fold errors."""
    src = pooled[(pooled["model"] == model) & (pooled["n_train"] == n)]
    complete = not src.empty
    if not complete:
        src = pred[(pred["model"] == model) & (pred["n_train"] == n)]
    return src["error_kw"].to_numpy(), complete


def _top_models(main: pd.DataFrame, best: pd.DataFrame, n: int, qnn_models: set[str]) -> list[str]:
    """The best QNN of each readout at N and the top three baselines by mean validation RMSE."""
    at = main[main["n_train"] == n]
    base = (
        at[~at["model"].isin(qnn_models)]
        .sort_values("val_rmse_mean", kind="stable")["model"]
        .tolist()[:3]
    )
    chosen = best[best["n_train"] == n]
    if "readout" in chosen:
        chosen = chosen.sort_values("readout", key=lambda s: s != PRIMARY_READOUT, kind="stable")
    return chosen["best"].tolist() + base


def _hist(ax: plt.Axes, model: str, errors: np.ndarray, bins: np.ndarray, text: str) -> None:
    color = base_style(model)[0]
    unit = is_unit(model)
    ax.hist(
        errors,
        bins=bins,
        color=color,
        alpha=0.45 if unit else 0.85,
        edgecolor=color if unit else "white",
        linewidth=0.3,
    )
    ax.axvline(0, color=INK_2, linewidth=0.6)
    ax.text(0.03, 0.95, text, transform=ax.transAxes, va="top", color=INK, fontsize=7)


def f4_a2(
    agg: Path,
    figures: Path,
    qnn_models: set[str],
    focus_n: int,
    n_rows: int,
    n_folds: int,
    notes: dict,
) -> tuple[list[Path], dict]:
    pred = read(agg, "predictions_test.csv.gz")
    main, best = read(agg, "main.csv"), read(agg, "best_qnn.csv")
    if pred.empty or main.empty or best.empty:
        return [], {}
    pred = merge_same_unitary(
        pred, ["n_train", "fold", "seed", "row"], ["error_kw"], notes, "F4/A2"
    )
    main = merge_same_unitary(main, ["n_train"], ["val_rmse_mean"], notes, "F4/A2 selection")
    present = set(pred["model"])
    best = best.assign(best=best["best"].map(lambda m: merged_name(m, present)))
    pooled = pooled_errors(pred, n_rows, n_folds)
    sizes = sorted(main["n_train"].unique())
    focus = focus_n if focus_n in sizes else max(sizes)
    notes["focus_n"] = int(focus)
    notes["F4_A2_seeds"] = sorted(int(s) for s in pred["seed"].unique())
    paths = []

    models = _top_models(main, best, focus, qnn_models)
    series = [(m, _errors_for(pred, pooled, m, focus)[0]) for m in models]
    notes["F4_models"] = models
    notes["F4_pooled"] = all(_errors_for(pred, pooled, m, focus)[1] for m in models)
    limit = max((np.nanmax(np.abs(e)) for _, e in series if e.size), default=1.0)
    bins = np.linspace(-limit, limit, 41)
    fig, axes = plt.subplots(
        1,
        len(series),
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(DOUBLE, 2.0),
        layout="constrained",
    )
    for ax, (model, errors) in zip(axes[0], series, strict=True):
        _hist(ax, model, errors, bins, short(model))
        ax.set_xlabel("Prediction error (kW)")
    axes[0][0].set_xlim(-limit, limit)
    axes[0][0].set_ylabel("Count")
    paths += save(fig, figures, "F4_error_distributions")

    all_series = {
        n: [(m, _errors_for(pred, pooled, m, n)[0]) for m in _top_models(main, best, n, qnn_models)]
        for n in sizes
    }
    cols = max(len(s) for s in all_series.values())
    fig, axes = plt.subplots(
        len(sizes),
        cols,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(DOUBLE, 1.6 * len(sizes) + 0.4),
        layout="constrained",
    )
    limit_series = [s for ss in all_series.values() for s in ss]
    limit = max((np.nanmax(np.abs(e)) for _, e in limit_series if e.size), default=1.0)
    bins = np.linspace(-limit, limit, 41)
    for row, n in zip(axes, sizes, strict=True):
        for ax, (model, errors) in zip(row, all_series[n], strict=False):
            _hist(ax, model, errors, bins, f"{short(model)}\nN = {n}")
        for ax in row[len(all_series[n]) :]:
            ax.set_visible(False)
        row[0].set_ylabel("Count")
    axes[0][0].set_xlim(-limit, limit)
    for ax in label_orphans(axes):
        ax.set_xlabel("Prediction error (kW)")
    paths += save(fig, figures, "A2_error_distributions_all_n")
    return paths, notes


def f5(
    agg: Path, figures: Path, order: list[str], focus_n: int, notes: dict
) -> tuple[list[Path], dict]:
    main, cx = read(agg, "main.csv"), read(agg, "complexity.csv")
    if main.empty or cx.empty:
        return [], {}
    main = merge_same_unitary(main, ["n_train"], ["test_rmse_mean", "test_rmse_std"], notes, "F5")
    cx = merge_same_unitary(cx, ["n_train"], ["trainable_params_mean"], notes, "F5 parameters")
    sizes = sorted(main["n_train"].unique())
    focus = focus_n if focus_n in sizes else max(sizes)
    at = main[main["n_train"] == focus].merge(cx[cx["n_train"] == focus], on=["model", "n_train"])
    at = at[at["trainable_params_mean"].fillna(0) > 0]
    unit = any(is_unit(m) for m in at["model"])
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.6), layout="constrained")
    for model in ordered(order, set(at["model"])):
        r = at[at["model"] == model].iloc[0]
        x = r["trainable_params_mean"]
        if unit and model.startswith("QNN-"):  # the two readouts share x = 12: dodge them
            x = x * F5_DODGE if is_unit(model) else x / F5_DODGE
        ax.errorbar(
            x, r["test_rmse_mean"], yerr=r["test_rmse_std"], linestyle="none", **style(model)
        )
    ax.set_xscale("log")
    ax.set_xlabel(f"Trainable parameters (N = {focus})")
    ax.set_ylabel("Test RMSE (kW)")
    handles, labels = ax.get_legend_handles_labels()
    legend_below_fit(fig, handles, labels)
    omitted = sorted(set(main.loc[main["n_train"] == focus, "model"]) - set(at["model"]))
    extra: dict[str, Any] = {"F5_omitted_no_parameter_count": omitted}
    if unit:
        extra["F5_dodge"] = F5_DODGE
    return save(fig, figures, "F5_rmse_vs_parameters"), extra


def with_predictions(pred: pd.DataFrame, config: Config) -> pd.DataFrame | None:
    """Predictions and actual values: as aggregated from raw results, or rebuilt from the
    per-sample errors of the results package and the dataset (None without the dataset)."""
    if "pred_kw" in pred:
        return pred
    if not dataset_available(config):
        return None
    actual = load_dataset(config.path("data"), config["data"]).target[pred["row"].to_numpy()]
    return pred.assign(actual_kw=actual, pred_kw=actual + pred["error_kw"].to_numpy())


def a1(
    agg: Path, figures: Path, order: list[str], focus_n: int, notes: dict, config: Config
) -> tuple[list[Path], dict]:
    pred = read(agg, "predictions_test.csv.gz")
    if pred.empty:
        return [], {}
    pred = with_predictions(pred, config)
    if pred is None:
        print("make_figures: A1 needs the dataset (not found); skipped")
        return [], {"A1_needs_dataset": True}
    pred = merge_same_unitary(pred, ["n_train", "fold", "seed", "row"], ["pred_kw"], notes, "A1")
    sizes = sorted(pred["n_train"].unique())
    focus = focus_n if focus_n in sizes else max(sizes)
    at = pred[pred["n_train"] == focus]
    seed = int(at["seed"].min())
    at = at[at["seed"] == seed]
    models = ordered(order, set(at["model"]))
    cols = 6
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(DOUBLE, 1.25 * rows + 0.5),
        layout="constrained",
    )
    lo, hi = (
        float(min(at["actual_kw"].min(), at["pred_kw"].min())),
        float(max(at["actual_kw"].max(), at["pred_kw"].max())),
    )
    for ax, model in zip(axes.flat, models, strict=False):
        g = at[at["model"] == model]
        color, _ = base_style(model)
        ax.scatter(
            g["actual_kw"],
            g["pred_kw"],
            s=1.5,
            color=color,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=0.6)
        ax.text(0.04, 0.96, short(model), transform=ax.transAxes, va="top", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    label_orphans(axes)
    axes[0][0].set_xlim(lo, hi)
    axes[0][0].set_ylim(lo, hi)
    fig.supxlabel("Actual power (kW)", fontsize=8)
    fig.supylabel("Predicted power (kW)", fontsize=8)
    folds = sorted(at["fold"].unique())
    return save(fig, figures, "A1_actual_vs_predicted"), {
        "A1_seed": seed,
        "A1_folds": [int(f) for f in folds],
    }


def a3(
    agg: Path, figures: Path, order: list[str], readouts: dict[str, str], notes: dict
) -> list[Path]:
    """Training loss per objective evaluation, every run: rows per (readout, N), a shared y
    scale within each readout."""
    curves = with_readout(read(agg, "curves_eval.csv"), readouts)
    if curves.empty:
        return []
    keys = ["n_train", "fold", "seed", "evaluation"]
    curves = merge_same_unitary(curves, keys, ["train_mse"], notes, "A3")
    sizes = sorted(curves["n_train"].unique())
    groups = readouts_of(curves)
    per_readout = {
        r: ordered(order, set(curves.loc[curves["readout"] == r, "model"])) for r in groups
    }
    cols = max(len(m) for m in per_readout.values())
    rows = [(r, n) for r in groups for n in sizes]
    fig, axes = plt.subplots(
        len(rows),
        cols,
        sharex=True,
        squeeze=False,
        figsize=(DOUBLE, 1.3 * len(rows) + 0.5),
        layout="constrained",
    )
    first: dict[str, plt.Axes] = {}
    for i, (readout, n) in enumerate(rows):
        models = per_readout[readout]
        for j in range(cols):
            ax = axes[i][j]
            if j >= len(models):
                ax.set_visible(False)
                continue
            model = models[j]
            if readout in first:
                ax.sharey(first[readout])
            else:
                first[readout] = ax
            g = curves[(curves["n_train"] == n) & (curves["model"] == model)]
            for _, run in g.groupby(["fold", "seed"]):
                ax.plot(
                    run["evaluation"],
                    run["train_mse"],
                    color=base_style(model)[0],
                    linewidth=0.4,
                    alpha=0.5,
                )
            ax.set_yscale("log")
            if j > 0:
                ax.yaxis.set_tick_params(labelleft=False)
            ax.text(
                0.95,
                0.95,
                f"{short(model)}\nN = {n}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
            )
    label_orphans(axes)
    fig.supxlabel("Objective evaluation", fontsize=8)
    fig.supylabel(
        "Training MSE (scaled target; "
        + ", ".join(f"target in {r}" for r in groups)
        + (" from top to bottom)" if len(groups) > 1 else ")"),
        fontsize=8,
    )
    return save(fig, figures, "A3_qnn_loss_curves")


def a4(config: Config, figures: Path) -> list[Path]:
    """Circuit diagrams with Qiskit's matplotlib drawer (needs pylatexenc), one style."""
    target = figures / "circuits"
    paths = []
    circuits = all_circuits(config["qnn"])
    drawings = [("feature_map", circuits[0].feature_map)]
    drawings += [(f"ansatz_{qc.name}_{qc.entanglement}", qc.ansatz) for qc in circuits]
    drawings += [(f"full_{qc.name}", qc.circuit) for qc in circuits]
    for name, circuit in drawings:
        # Measure at scale 1, then redraw with the drawer's own scale so that boxes and text
        # shrink together (resizing the figure afterwards would make labels overlap).
        probe = circuit.draw(output="mpl", style="bw", fold=-1)
        width = probe.get_size_inches()[0]
        plt.close(probe)
        fig = circuit.draw(output="mpl", style="bw", fold=-1, scale=min(1.0, DOUBLE / width))
        paths += save(fig, target, f"A4_{name}")
    return paths


def f6(agg_root: Path, figures: Path, focus_n: int, notes: dict) -> list[Path]:
    """Dumbbell at N = focus: blocked and random test RMSE per model, ordered by blocked RMSE
    ."""
    cmp = read(agg_root, "protocol_comparison.csv")
    if cmp.empty:
        return []
    values = ["test_rmse_mean_blocked", "test_rmse_mean_random"]
    cmp = merge_same_unitary(cmp, ["n_train"], values, notes, "F6")
    sizes = sorted(cmp["n_train"].unique())
    focus = focus_n if focus_n in sizes else max(sizes)
    notes["F6_n"] = int(focus)
    at = cmp[cmp["n_train"] == focus].sort_values(
        "test_rmse_mean_blocked", ascending=False, kind="stable"
    )
    y = np.arange(len(at))
    fig, ax = plt.subplots(figsize=(SINGLE, 0.17 * len(at) + 1.0), layout="constrained")
    ax.hlines(
        y, at["test_rmse_mean_blocked"], at["test_rmse_mean_random"], color=INK_2, linewidth=1.0
    )
    for protocol, label in (
        ("blocked", "Blocked (primary)"),
        ("random", "Random cross-validation"),
    ):
        ax.scatter(
            at[f"test_rmse_mean_{protocol}"],
            y,
            color=PROTOCOL_COLORS[protocol],
            s=22,
            zorder=3,
            label=label,
            edgecolors="white",
            linewidths=0.6,
        )
    ax.set_yticks(y, [short(m) for m in at["model"]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Test RMSE (kW), N = {focus}")
    handles, labels = ax.get_legend_handles_labels()
    legend_below(fig, handles, labels, ncol=2)
    return save(fig, figures, "F6_protocol_dumbbell")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def read_meta(agg_root: Path, config: Config) -> dict[str, Any]:
    """Model order, kinds, and readouts of the aggregation (meta.json of each protocol)."""
    meta: dict[str, Any] = {"order": [], "kinds": {}, "readouts": {}}
    for protocol in ("blocked", "random"):
        path = agg_root / protocol / "meta.json"
        if path.is_file():
            content = json.loads(path.read_text(encoding="utf-8"))
            meta["order"] += [m for m in content["order"] if m not in meta["order"]]
            meta["kinds"].update(content["kinds"])
            meta["readouts"].update(content["readouts"])
    if not meta["order"]:
        meta["order"] = [m for names in config["models"].values() for m in names]
        meta["kinds"] = {m: k for k, names in config["models"].items() for m in names}
    return meta


def run(
    config: Config,
    outputs: Path,
    agg_root: Path | None = None,
    random_config: Config | None = None,
) -> dict[str, Any]:
    """All figures; ``random_config`` (a random-protocol config) adds F0 panel (b)."""
    apply_style()
    agg_root = agg_root or config.path("results") / "aggregated"
    agg = agg_root / "blocked"
    figures = outputs / "figures"
    meta = read_meta(agg_root, config)
    order, kinds, readouts = meta["order"], meta["kinds"], meta["readouts"]
    qnn_models = {m for m, k in kinds.items() if k == "qnn"}
    focus_n = int(max(config["sizes"]))
    written: list[Path] = []
    notes: dict[str, Any] = {}
    acc = accuracy_order(order)  # accuracy figures: same-unitary pairs as one series
    qnn_acc = qnn_models | {MERGED, MERGED_U}
    written += f0(config, random_config, figures, notes)
    written += f1(agg, figures, order, kinds, notes)
    written += f2(agg, figures, acc, readouts, notes)
    written += f3(agg, figures, order, notes)  # timing: QNN-1 and QNN-5 separate
    n_rows, n_folds = int(config["data"]["n_rows"]), int(config["folds"]["n_folds"])
    paths, extra = f4_a2(agg, figures, qnn_acc, focus_n, n_rows, n_folds, notes)
    written += paths
    notes.update(extra)
    paths, extra = f5(agg, figures, acc, focus_n, notes)
    written += paths
    notes.update(extra)
    written += f6(agg_root, figures, focus_n, notes)
    paths, extra = a1(agg, figures, acc, focus_n, notes, config)
    written += paths
    notes.update(extra)
    written += a3(agg, figures, acc, readouts, notes)
    written += a4(config, figures)
    written += a5(agg, figures, acc, kinds, notes)
    notes["files"] = sorted(p.relative_to(outputs).as_posix() for p in written)
    print(f"make_figures: {len(written)} files -> {figures}")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--aggregated", type=Path, help="default: <results>/aggregated")
    parser.add_argument("--outputs", type=Path, help="default: paths.outputs of the config")
    parser.add_argument("--random-config", type=Path, help="random protocol: F0 panel (b)")
    add_runs_argument(parser)
    args = parser.parse_args(argv)
    use_runs_dir(args.runs)
    config = load_config(args.config)
    random_config = load_config(args.random_config) if args.random_config else None
    run(config, args.outputs or config.path("outputs"), args.aggregated, random_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
