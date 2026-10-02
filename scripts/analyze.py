"""Analysis entry point: aggregation, tables, figures, statistics, and results_summary.md.

One batch: takes the run directory (the directory that holds ``results/``) and writes the
tables, figures, and summary to the outputs directory::

    python scripts/analyze.py --config configs/experiment.yaml --runs <run directory> [--outputs outputs]

Several batches (the three batches of the paper): one ``--source NAME CONFIG RUNS`` per
batch, where RUNS is a run directory or a batch directory of the results package, and an
explicit outputs directory::

    python scripts/analyze.py \\
        --source primary configs/experiment.yaml results/primary \\
        --source blocked_unit configs/blocked_unit.yaml results/blocked_unit \\
        --source random configs/random.yaml results/random \\
        --outputs outputs/paper

Steps: scripts/aggregate.py (one directory per protocol under ``--aggregated``; default
``<runs>/results/aggregated`` for one batch, ``<outputs>/aggregated`` for several), then
scripts/make_tables.py and scripts/make_figures.py (``<outputs>/tables``, ``<outputs>/figures``),
then ``<outputs>/results_summary.md`` with the key numbers behind every table and figure.
Completeness and provenance (one commit, one config hash, clean tree) are checked per batch.
A partial grid (e.g. the smoke outputs) is handled: whatever cannot be computed is reported.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate  # noqa: E402
import make_figures  # noqa: E402
import make_tables  # noqa: E402
from qnnwind.io import Config, git_state, utc_now, write_text  # noqa: E402
from runs_dir import use_runs_dir  # noqa: E402

PROTOCOL_TITLES = {"blocked": "blocked protocol", "random": "random K-fold protocol"}

# --------------------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------------------


def fmt(x: object, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{decimals}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{decimals}f}"
    return str(x)


def pm(mean: object, std: object, decimals: int) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    s = (
        "-"
        if std is None or (isinstance(std, float) and math.isnan(std))
        else f"{std:.{decimals}f}"
    )
    return f"{mean:.{decimals}f} ± {s}"


def pms(frame: pd.DataFrame, column: str, decimals: int, scale: float = 1.0) -> list[str]:
    """``mean ± std`` strings of ``<column>_mean`` and ``<column>_std``."""
    return [
        pm(m * scale, s * scale, decimals)
        for m, s in zip(frame[f"{column}_mean"], frame[f"{column}_std"], strict=True)
    ]


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(none)_\n"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(fmt(v) if isinstance(v, float) else str(v) for v in r) + " |"
        for r in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def by_readout(frame: pd.DataFrame, *keys: str) -> pd.DataFrame:
    """Sort by readout (primary first), then ``keys``."""
    if "readout" not in frame or frame.empty:
        return frame
    rank = frame["readout"].map(make_tables.readout_key)
    return frame.assign(_ro=rank).sort_values(["_ro", *keys], kind="stable").drop(columns="_ro")


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------


def settings(config: Config, notes: dict[str, Any], multi: bool) -> list[str]:
    """The analysis thresholds and choices, printed before any result."""
    workers = int(config["runtime"]["workers"])
    cutoff = aggregate.NODE_LOAD_FRACTION * workers
    focus = notes.get("focus_n", int(max(config["sizes"])))
    f4_seeds = notes.get("F4_A2_seeds", list(config["seeds"]))
    window = config["stats"]["stability_window"]
    start = config["stats"]["stability_start_eval"]
    baselines = sum(len(v) for k, v in config["models"].items() if k != "qnn")
    family = "at each N, for one readout and one protocol," if multi else "at each N"
    return [
        "## Analysis settings",
        "",
        f"- Node-load cutoff for the QNN timing fits and the same-unitary time ratios: mean busy "
        f"workers ≥ {aggregate.NODE_LOAD_FRACTION} × W = {cutoff:g} (W = {workers}); runs without a "
        "node-load record are excluded.",
        f"- F4, F5, and A1 are drawn at N = {focus} (the largest N of the config: {max(config['sizes'])}).",
        f"- Seeds: F4 and A2 pool the out-of-fold errors of all seeds ({', '.join(map(str, f4_seeds))}); "
        f"A1 shows seed {notes.get('A1_seed', min(config['seeds']))}.",
        f"- Stability window: evaluations 1 to {window}; SD and MS over evaluations {start + 1} to {window}; "
        "shorter curves padded with their final loss. SC is normalized within each readout.",
        f"- Diebold-Mariano: HLN correction with h = {config['stats']['dm']['horizon']}, Bartlett HAC lag "
        f"{config['stats']['dm']['hac_lag']}, per seed, median over seeds. Holm adjustment: the {baselines} "
        f"baseline comparisons {family} form one family, separately for Wilcoxon and Diebold-Mariano.",
        "- Best QNN: per readout and protocol, lowest mean validation RMSE over folds and seeds; ties "
        "go to fewer total gates.",
        "- T3c (secondary): predictions clipped to [0, maximum training power of the run]; the "
        "primary metrics are unclipped.",
        "- Conclusions rest on the blocked protocol; the random protocol is a comparison.",
        "",
    ]


def provenance_lines(name: str, prov: dict[str, Any]) -> list[str]:
    if not prov.get("runs"):
        return [f"- **{name}**: no runs found."]
    check = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    if "git_commits" not in prov:  # results package
        return [
            f"- **{name}** ({prov['runs']} runs, results package):",
            f"  - One config hash: **{check(prov['single_config_hash'])}** "
            f"({', '.join(h[:12] for h in prov['config_hashes'])}).",
            f"  - Dataset SHA-256: {', '.join(s[:12] for s in prov['dataset_sha256'])}.",
        ]
    return [
        f"- **{name}** ({prov['runs']} runs):",
        f"  - One git commit: **{check(prov['single_commit'])}** ({', '.join(c[:12] for c in prov['git_commits'])}).",
        f"  - One config hash: **{check(prov['single_config_hash'])}** ({', '.join(h[:12] for h in prov['config_hashes'])}).",
        f"  - Working tree clean (dirty = false) in every run: **{check(prov['all_clean'])}** "
        f"(dirty: {prov['dirty_true']}, unknown: {prov['dirty_unknown']}; source: {', '.join(prov['git_sources'])}).",
        f"  - Dataset SHA-256: {', '.join(s[:12] for s in prov['dataset_sha256'])}; workers per node: "
        f"{', '.join(prov['workers_per_node'])}; hosts: {prov['hostnames']}; CPU: {'; '.join(prov['cpu_models'])}.",
    ]


def main_view(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": g["model"],
            "RMSE (kW)": pms(g, "test_rmse", 1),
            "R²": pms(g, "test_r2", 3),
            "MAE (kW)": pms(g, "test_mae", 1),
            "val RMSE (kW)": pms(g, "val_rmse", 1),
            "RMSE clipped": pms(g, "test_clip_rmse", 1),
            "R² clipped": pms(g, "test_clip_r2", 3),
            "negative (%)": pms(g, "test_frac_negative", 1, 100.0),
            "runs": g["runs"],
        }
    )


def tests_view(tests: pd.DataFrame) -> pd.DataFrame:
    return by_readout(tests, "n_train").assign(
        dRMSE=tests["qnn_minus_baseline_rmse_mean"].map(lambda v: fmt(v, 1)),
        W=tests["wilcoxon_statistic"].map(lambda v: fmt(v, 1)),
        p_W=tests["wilcoxon_p"].map(fmt),
        p_W_Holm=tests["wilcoxon_p_holm"].map(fmt),
        DM=tests["dm_statistic_median"].map(lambda v: fmt(v, 2)),
        p_DM=tests["dm_p_median"].map(fmt),
        p_DM_Holm=tests["dm_p_holm"].map(fmt),
    )[
        ["readout", "n_train", "qnn", "baseline", "dRMSE", "W", "p_W", "p_W_Holm",
         "wilcoxon_pairs", "DM", "p_DM", "p_DM_Holm", "dm_seeds"]
    ]  # fmt: skip


def protocol_sections(protocol: str, out: dict[str, Any], config: Config) -> list[str]:
    """Every result section of one protocol."""
    title = PROTOCOL_TITLES.get(protocol, protocol)
    lines: list[str] = []
    best = out["best_qnn"]
    lines += [
        "",
        f"## Best QNN configuration, {title} (by mean validation RMSE per readout; ties to fewer gates)",
        "",
    ]
    lines.append(
        md_table(
            by_readout(best, "n_train").assign(
                val_rmse_mean=best["val_rmse_mean"].map(lambda v: f"{v:.1f}")
            )
        )
    )

    main = out["main"]
    lines += [
        f"## Main test results, {title} (T3, T3c, F1, A5): mean ± std over (fold, seed) runs",
        "",
    ]
    for n, g in main.groupby("n_train"):
        lines += [
            f"**N = {n}**",
            "",
            md_table(main_view(g.sort_values("test_rmse_mean", kind="stable"))),
        ]

    pooled = out["pooled"]
    lines += [
        f"## Pooled out-of-fold metrics, {title} (per seed, then mean ± std over seeds)",
        "",
    ]
    if pooled.empty:
        lines.append("_Not available: needs all six test folds of a (model, N, seed)._\n")
    else:
        view = pooled.assign(RMSE=pms(pooled, "rmse", 1), R2=pms(pooled, "r2", 3))
        lines.append(md_table(view[["model", "n_train", "RMSE", "R2", "seeds"]]))

    lines += [f"## Statistical tests, {title} (T6): best QNN of each readout vs each baseline", ""]
    lines.append(md_table(tests_view(out["stats_tests"])))
    lines.append(
        "Wilcoxon needs ≥ 2 paired (fold, seed) runs with a nonzero difference; DM needs pooled "
        f"out-of-fold predictions (HLN, Bartlett lag {config['stats']['dm']['hac_lag']}).\n"
    )
    if protocol != "blocked":
        return lines

    lines += ["## Timing (T5, F3)", ""]
    lines += ["Node-load filter (mean busy workers ≥ 0.9 W; runs without node_load excluded):", ""]
    lines.append(md_table(by_readout(out["timing_exclusions"], "n_train")))
    fits = by_readout(out["timing_fits"], "model")
    lines += ["Fits of time per objective evaluation = a N + b (included runs), per readout:", ""]
    lines.append(
        md_table(fits.assign(a=fits["a"].map(fmt), b=fits["b"].map(fmt), r2=fits["r2"].map(fmt)))
    )
    ratio = out["timing_ratio"]
    lines += ["Same-unitary ratios of time per evaluation, paired by (fold, seed):", ""]
    lines.append(
        md_table(ratio.assign(ratio=pms(ratio, "ratio", 3))[["pair", "n_train", "pairs", "ratio"]])
    )
    total = out["timing_total"]
    lines += ["Total training time (descriptive):", ""]
    lines.append(
        md_table(
            total.assign(hours=pms(total, "total_training_time", 2, 1 / 3600))[
                ["model", "n_train", "hours", "runs"]
            ]
        )
    )

    lines += ["## Convergence: early stops and cache near hits", ""]
    lines.append(md_table(out["convergence"]))

    st = out["stability"]
    lines += [
        f"## Stability score (T4): window of {config['stats']['stability_window']} evaluations, "
        "per readout",
        "",
    ]
    if not st.empty:
        st = by_readout(st, "rank", "config")
        lines.append(
            md_table(
                st[["readout", "config", "SD", "MS", "FL", "SC", "rank", "padded_runs"]].assign(
                    SD=st["SD"].map(fmt),
                    MS=st["MS"].map(fmt),
                    FL=st["FL"].map(fmt),
                    SC=st["SC"].map(fmt),
                )
            )
        )
        runs_total = len(out["stability_runs"])
        lines.append(
            f"Padded runs (converged before the window's end): {int(st['padded_runs'].sum())} of {runs_total}.\n"
        )
    return lines


def figure_notes(notes: dict[str, Any]) -> list[str]:
    lines = ["", "Figure notes:", ""]
    checks = notes.get("same_unitary_identical", {})
    lines.append(
        "- QNN-1 and QNN-5 implement the same unitary, and so do QNN-1u and QNN-5u. The "
        "accuracy figures (F2, F4, F5, F6, A1, A2, A3, A5) draw each pair as one series, "
        "'QNN-1/QNN-5 (same unitary)' and 'QNN-1u/QNN-5u (same unitary)', after checking that "
        "their data are identical; F3 (timing) and every table keep them separate."
        + (
            f" Identity checks: {', '.join(f'{k}: {v}' for k, v in checks.items())}."
            if checks
            else ""
        )
    )
    if checks and not all(checks.values()):
        lines.append(
            "- **WARNING: same-unitary data differ where marked False; those figures show them separately.**"
        )
    if "F0" in notes:
        f0 = notes["F0"]
        lines.append(
            f"- F0 (evaluation layout) is drawn from the fold functions (`qnnwind.folds`) of "
            f"`{f0['blocked_config']}`: (a) every blocked fold"
            + (
                f"; (b) fold 0 of the random protocol of `{f0['random_config']}`."
                if f0["random_config"]
                else "."
            )
        )
    if "F2_band" in notes:
        lo, hi = (round(100 * q) for q in notes["F2_band"]["quantiles"])
        lines.append(
            f"- F2: lines are the median, shaded bands the {lo}th to {hi}th percentile over the "
            "(fold, seed) runs of each configuration and N, per iteration (the percentiles stay "
            "positive on the log axis). The y-axis of each row spans "
            f"{notes['F2_band']['ylim_factors'][0]:g} × its lowest to "
            f"{notes['F2_band']['ylim_factors'][1]:g} × its highest band value."
        )
    for readout, by_n in notes.get("best_qnn_by_n", {}).items():
        chosen = ", ".join(f"N = {n}: {m}" for n, m in by_n.items())
        lines.append(f"- F1 series 'Best QNN, {readout}': {chosen}.")
    if "F3_readout" in notes:
        lines.append(
            f"- F3 shows the {notes['F3_readout']} readout (the circuits of both readouts are identical)."
        )
    focus = notes.get("focus_n")
    if focus is not None:
        lines.append(f"- F4/F5/A1 are drawn at N = {focus} (the largest N present).")
    if "F4_models" in notes:
        lines.append(
            f"- F4 models (best QNN of each readout, then the top three baselines by validation RMSE): "
            f"{', '.join(notes['F4_models'])}; "
            + (
                f"pooled out-of-fold errors, all seeds ({', '.join(map(str, notes.get('F4_A2_seeds', [])))})."
                if notes.get("F4_pooled")
                else "pooled predictions not available: errors of the available test folds."
            )
        )
    if notes.get("F5_omitted_no_parameter_count"):
        lines.append(
            f"- F5 omits models without a trainable parameter count (see T8): {', '.join(notes['F5_omitted_no_parameter_count'])}."
        )
    if "F5_dodge" in notes:
        d = notes["F5_dodge"]
        lines.append(
            f"- F5: every QNN has 12 trainable parameters. For legibility the [-1, 1] QNN points are "
            f"drawn at x = 12 / {d:g} ≈ {12 / d:.2f} and the [0, 1] QNN points at x = 12 × {d:g} ≈ "
            f"{12 * d:.2f} (a multiplicative dodge on the log axis); MLP-PM is drawn at its true 13."
        )
    lines.append(
        "- F5 and T8, SVR: trainable parameters = dual coefficients (one per support vector) + intercept; the support "
        "vectors themselves are stored training samples (a complexity measure, as for kNN)."
    )
    if "F6_n" in notes:
        lines.append(f"- F6 is drawn at N = {notes['F6_n']}, models ordered by blocked test RMSE.")
    if "A1_seed" in notes:
        lines.append(f"- A1 shows seed {notes['A1_seed']}, test folds {notes['A1_folds']}.")
    if notes.get("A1_needs_dataset"):
        lines.append("- A1 (actual vs predicted) needs the dataset and was not drawn.")
    lines.append("")
    return lines


def display_path(path: Path) -> str:
    """A path relative to the repository where possible (no local absolute paths)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def summary(
    sources: list[aggregate.Source],
    config: Config,
    out: dict[str, Any],
    notes: dict[str, Any],
    tables: list[Path],
) -> str:
    commit = (git_state(config.root).get("commit") or "?")[:12]
    lines = [
        "# Results summary",
        "",
        f"Generated {utc_now()} by `scripts/analyze.py`; analysis code at commit `{commit}`. Batches:",
        "",
    ]
    lines += [
        f"- **{s.name}**: `{s.config.source.name}` (config hash `{s.config.hash[:12]}`, "
        f"{s.protocol} protocol); runs read from `{display_path(s.results)}`."
        for s in sources
    ]
    protocols = [p for p in ("blocked", "random") if p in out]
    lines += ["", "## Completeness", ""]
    for protocol in protocols:
        for name, comp in out[protocol]["completeness_report"].items():
            lines.append(
                f"- **{name}** ({protocol}): expected runs (models × N × folds × seeds of its "
                f"config): **{comp['expected']}**; found and valid: **{comp['found']}** "
                f"({'complete' if comp['complete'] else 'INCOMPLETE'})."
            )
            if comp["missing"]:
                shown = comp["missing"][:40]
                lines.append(
                    f"  - Missing ({len(comp['missing'])}): "
                    + ", ".join(shown)
                    + (" …" if len(comp["missing"]) > 40 else "")
                )
    lines += ["", "## Provenance (checked per batch)", ""]
    for protocol in protocols:
        for name, prov in out[protocol]["provenance"].items():
            lines += provenance_lines(name, prov)
    lines += ["", *settings(config, notes, len(sources) > 1)]
    for protocol in protocols:
        if "main" in out[protocol]:
            lines += protocol_sections(protocol, out[protocol], config)

    if "comparison" in out:
        cmp = out["comparison"]
        lines += ["## Protocol comparison (T9, F6): test RMSE and R², random minus blocked", ""]
        blocked_cols = cmp.rename(columns=lambda c: c.removesuffix("_blocked"))
        random_cols = cmp.rename(columns=lambda c: c.removesuffix("_random"))
        view = cmp.assign(
            blocked=pms(blocked_cols, "test_rmse", 1),
            random=pms(random_cols, "test_rmse", 1),
            dRMSE=cmp["rmse_random_minus_blocked"].map(lambda v: fmt(v, 1)),
            dR2=cmp["r2_random_minus_blocked"].map(fmt),
        )[["n_train", "model", "blocked", "random", "dRMSE", "dR2"]]
        lines.append(md_table(view.sort_values(["n_train", "model"], kind="stable")))

    blocked = out.get("blocked", {})
    hp = blocked.get("hyperparameters", pd.DataFrame())
    lines += ["## Hyperparameters (T7) and complexity (T8)", ""]
    lines.append(f"- Tuning results found, blocked protocol: {len(hp)} (model, N, fold) studies.")
    if "random" in out:
        lines.append(
            f"- Tuning results found, random protocol: {len(out['random'].get('hyperparameters', []))} studies."
        )
    lines.append(
        "- Parameter counts and complexity measures: `aggregated/blocked/complexity.csv`.\n"
    )

    lines += ["## Tables and figures", ""]
    lines += [f"- `{p.relative_to(p.parents[1]).as_posix()}`" for p in tables]
    lines += [f"- `{f}`" for f in notes.get("files", [])]
    lines += figure_notes(notes)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    aggregate.add_source_arguments(parser)
    parser.add_argument("--outputs", type=Path, help="one batch: default paths.outputs")
    parser.add_argument("--aggregated", type=Path, help="aggregation root (see the docstring)")
    args = parser.parse_args(argv)
    use_runs_dir(args.runs)
    if bool(args.source) == bool(args.config):
        parser.error("give either --config (one batch) or --source (repeatable)")
    if args.source and args.outputs is None:
        parser.error("several batches need an explicit --outputs directory")
    sources = aggregate.sources_from_args(args)
    for s in sources:
        if not s.results.is_dir():
            print(f"{s.name}: no results directory at {s.results}", file=sys.stderr)
            return 1
    blocked = [s for s in sources if s.protocol == "blocked"]
    config = (blocked or sources)[0].config  # data, circuits, and settings for T1, T2
    outputs = (args.outputs or config.path("outputs")).resolve()
    if args.aggregated:
        root = args.aggregated.resolve()
    elif len(sources) == 1:
        root = sources[0].results / "aggregated"
    else:
        root = outputs / "aggregated"

    out = aggregate.run(sources, root)
    if all(tables["runs"].empty for p, tables in out.items() if p != "comparison"):
        print("no valid runs found; nothing to analyze", file=sys.stderr)
        return 1
    tables = make_tables.run(config, outputs, root)
    random_config = next((s.config for s in sources if s.protocol == "random"), None)
    notes = make_figures.run(config, outputs, root, random_config)
    path = outputs / "results_summary.md"
    write_text(path, summary(sources, config, out, notes, tables))
    print(f"analyze: summary -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
