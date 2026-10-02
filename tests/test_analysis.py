"""Aggregation, statistics, tables, figures, and the summary."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from matplotlib import image

import aggregate
import analyze
import synthetic_runs as syn
from make_tables import pm, sci, tabular
from qnnwind.io import load_config, run_dir
from qnnwind.stats import diebold_mariano

from conftest import ROOT


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory, dataset):
    """The synthetic full grid, analyzed once for the module."""
    root = tmp_path_factory.mktemp("synthetic")
    cfg = syn.build(root, ROOT)
    mp = pytest.MonkeyPatch()
    mp.setenv("QNNWIND_SCRATCH", str(root))
    try:
        assert analyze.main(["--config", str(cfg), "--runs", str(root)]) == 0
        config = load_config(cfg)
        agg = config.path("results") / "aggregated" / "blocked"
        tables = {p.stem: pd.read_csv(p) for p in agg.glob("*.csv")}
        yield root, config, agg, tables
    finally:
        mp.undo()


def test_completeness_and_provenance(synthetic):
    _, _, agg, t = synthetic
    report = json.loads((agg / "completeness.json").read_text(encoding="utf-8"))["synthetic"]
    assert report["expected"] == report["found"] == 7 * 2 * 6 * 3 and report["complete"]
    prov = json.loads((agg / "provenance.json").read_text(encoding="utf-8"))["synthetic"]
    assert prov["single_commit"] and prov["single_config_hash"] and prov["all_clean"]
    assert prov["git_commits"] == [syn.COMMIT] and prov["workers_per_node"] == ["64"]
    assert len(t["runs"]) == 252 and t["runs"]["model"].nunique() == 7
    runs = t["runs"]
    assert (runs["protocol"] == "blocked").all() and (runs["batch"] == "synthetic").all()
    assert (runs["readout"] == "[-1, 1]").all()  # from the config: no readout key


def test_best_qnn_tie_goes_to_fewer_gates(synthetic):
    best = synthetic[3]["best_qnn"]
    assert list(best["best"]) == ["QNN-5", "QNN-5"]  # QNN-1 ties (same function), 40 > 34 gates
    assert list(best["tied_with"]) == ["QNN-1", "QNN-1"]


def test_pooled_out_of_fold(synthetic):
    _, config, agg, t = synthetic
    pooled = t["pooled"]
    assert len(pooled) == 7 * 2 and (pooled["seeds"] == 3).all()
    preds = pd.read_csv(agg / "predictions_test.csv.gz")
    one = preds[(preds["model"] == "SVR") & (preds["n_train"] == 750) & (preds["seed"] == 0)]
    assert sorted(one["row"]) == list(range(int(config["data"]["n_rows"])))


def test_statistical_tests(synthetic):
    _, _, agg, t = synthetic
    tests = t["stats_tests"]
    assert len(tests) == 2 * 4  # 2 sizes x 4 baselines (LR, MLP, MLP-PM, SVR)
    assert (tests["wilcoxon_pairs"] == 18).all() and (tests["dm_seeds"] == 3).all()
    assert (tests["dm_lag"] == 72).all()
    row = tests[(tests["n_train"] == 750) & (tests["baseline"] == "SVR")].iloc[0]
    runs = t["runs"][t["runs"]["n_train"] == 750].set_index(["model", "fold", "seed"])["test_rmse"]
    pairs = pd.concat([runs.loc["QNN-5"], runs.loc["SVR"]], axis=1, join="inner")
    expected = scipy.stats.wilcoxon(pairs.iloc[:, 0], pairs.iloc[:, 1])
    assert row["wilcoxon_p"] == pytest.approx(expected.pvalue)
    preds = pd.read_csv(agg / "predictions_test.csv.gz")
    stats = []
    for seed in syn.SEEDS:
        err = {}
        for model in ("QNN-5", "SVR"):
            g = preds[
                (preds["model"] == model) & (preds["n_train"] == 750) & (preds["seed"] == seed)
            ]
            g = g.sort_values("row")
            err[model] = (g["pred_kw"] - g["actual_kw"]).to_numpy()
        stats.append(diebold_mariano(err["QNN-5"], err["SVR"], horizon=1, lag=72)["statistic"])
    assert row["dm_statistic_median"] == pytest.approx(np.median(stats))
    assert row["dm_statistic_median"] > 0  # SVR has the lower loss in the synthetic grid


def test_timing_filter_fits_and_ratio(synthetic):
    t = synthetic[3]
    excl = t["timing_exclusions"].set_index("n_train")
    assert excl.loc[750, "excluded"] == 12 and excl.loc[750, "no_node_load"] == 3
    assert excl.loc[1500, "excluded"] == 9 and excl.loc[1500, "no_node_load"] == 0
    assert (excl["threshold_busy"] == 0.9 * 64).all()
    fits = t["timing_fits"].set_index("model")
    for model in syn.MODELS["qnn"]:
        rate = syn.RATE[model]
        assert fits.loc[model, "a"] == pytest.approx(rate)
        assert fits.loc[model, "r2"] == pytest.approx(1.0)
    ratio = t["timing_ratio"]
    assert np.allclose(ratio["ratio_mean"], 40 / 34)
    assert list(ratio["pairs"]) == [14, 15]  # included on both sides, paired by (fold, seed)


def test_convergence_and_stability(synthetic):
    t = synthetic[3]
    conv = t["convergence"].set_index(["model", "n_train"])
    assert conv.loc[("QNN-6", 750), "stopped_before_maxiter"] == 1
    assert conv.loc[("QNN-6", 750), "cache_near_hits_total"] == 11
    assert conv.loc[("QNN-1", 750), "stopped_before_maxiter"] == 0
    st = t["stability"].set_index("config")
    assert (st["padded_runs"] == 24).all()  # seeds 1 and 2 have 30 < 50 evaluations
    assert st.loc["QNN-1", "rank"] == st.loc["QNN-5", "rank"] == 1
    assert st.loc["QNN-6", "SC"] == pytest.approx(1.0) and (st["MS"] > 0).all()


def test_tables_and_figures(synthetic):
    root, _, _, _ = synthetic
    tables = root / "outputs" / "tables"
    names = {p.stem for p in tables.glob("*.tex")}
    assert {"t1_descriptive", "t1b_folds", "t2_circuits", "t3_main", "t3b_per_fold",
            "t3c_clipped", "t4_stability", "t5_timing", "t6_tests", "t7_hyperparameters",
            "t8_complexity"} <= names  # fmt: skip
    assert not {"t6_tests_random", "t9_protocols"} & names  # one blocked batch only
    for tex in tables.glob("t[0-9]*.tex"):  # the tables (not tables_preview.tex)
        text = tex.read_text(encoding="utf-8")
        assert r"\toprule" in text and r"\bottomrule" in text
        for line in text.splitlines():
            if line.startswith((r"\begin{tabular}", r"\begin{longtable}")):
                assert "|" not in line, tex.name  # no vertical rules
    figures = root / "outputs" / "figures"
    for name in ("F0_evaluation_layout", "F1_rmse_r2_vs_n", "F2_qnn_mse_vs_iteration",
                 "F3_qnn_timing", "F4_error_distributions", "F5_rmse_vs_parameters",
                 "A1_actual_vs_predicted", "A2_error_distributions_all_n", "A3_qnn_loss_curves",
                 "A5_qnn_rmse_r2_vs_n"):  # fmt: skip
        pdf = (figures / f"{name}.pdf").read_bytes()
        assert b"/FontFile2" in pdf and b"/FontFile3" not in pdf  # embedded TrueType (type 42)
        png = image.imread(figures / f"{name}.png")
        assert png.shape[1] / 300 == pytest.approx(7.2, abs=0.01)  # double column
    assert len(list((figures / "circuits").glob("A4_*.pdf"))) == 1 + 6 + 6
    summary = (root / "outputs" / "results_summary.md").read_text(encoding="utf-8")
    for heading in ("## Completeness", "## Provenance", "## Best QNN", "## Statistical tests",
                    "## Timing", "## Stability score", "## Tables and figures"):  # fmt: skip
        assert heading in summary
    assert "**PASS**" in summary and "FAIL" not in summary


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not available")
def test_tables_compile(synthetic):
    import os

    tables = synthetic[0] / "outputs" / "tables"
    # MiKTeX fails when PATH lists a file instead of a directory; keep only directories.
    path = os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep) if os.path.isdir(p))
    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tables_preview.tex"],
        cwd=tables,
        env={**os.environ, "PATH": path},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-3000:]


def test_partial_grid_is_handled(tmp_path, dataset, monkeypatch):
    cfg = syn.build(tmp_path, ROOT)
    monkeypatch.setenv("QNNWIND_SCRATCH", str(tmp_path))
    config = load_config(cfg)
    results = config.path("results")
    shutil.rmtree(run_dir(results, "LR", 750, 3, 1))  # one LR run missing
    for fold in range(6):
        for seed in syn.SEEDS:
            shutil.rmtree(run_dir(results, "QNN-6", 1500, fold, seed))  # a whole (model, N)
    (run_dir(results, "SVR", 750, 0, 0) / "preds_test.csv").unlink()  # an invalid run
    assert analyze.main(["--config", str(cfg), "--runs", str(tmp_path)]) == 0
    out = aggregate.aggregate(config)
    report = out["completeness_report"]["synthetic"]
    assert report["found"] == 252 - 1 - 18 - 1 and not report["complete"]
    assert "LR N=750 fold=3 seed=1" in report["missing"]
    pooled = out["pooled"].set_index(["model", "n_train"])
    assert pooled.loc[("LR", 750), "seeds"] == 2 and pooled.loc[("SVR", 750), "seeds"] == 2
    assert ("QNN-6", 1500) not in pooled.index
    summary = (tmp_path / "outputs" / "results_summary.md").read_text(encoding="utf-8")
    assert "INCOMPLETE" in summary and "Missing (20)" in summary


def test_real_runner_outputs(tmp_path, dataset, monkeypatch):
    """The aggregation reads what the runner actually writes (not only the synthetic files)."""
    from qnnwind import runner

    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    overrides = {
        "paths.results": str(tmp_path / "results"),
        "models": {"classical": ["LR"], "deep": [], "mlp_pm": ["MLP-PM"], "qnn": ["QNN-6"]},
        "sizes": [60],
        "seeds": [0],
        "folds.run": [5],
        "qnn.optimizer.maxiter": 1,
        "mlp_pm.optimizer.maxiter": 2,
    }
    config = load_config(ROOT / "configs" / "experiment.yaml", overrides)
    for model in ("LR", "MLP-PM", "QNN-6"):
        runner.execute_run(config, model, 60, 5, 0)
    out = aggregate.aggregate(config)
    runs = out["runs"].set_index("model")
    assert out["completeness_report"]["experiment"]["complete"] and len(runs) == 3
    assert (runs["protocol"] == "blocked").all() and (runs["readout"] == "[-1, 1]").all()
    assert runs["test_clip_rmse"].notna().all()
    assert runs.loc["QNN-6", "time_per_evaluation"] > 0 and runs.loc["QNN-6", "nfev"] >= 1
    assert runs.loc["LR", "trainable_params"] == 5 and runs.loc["MLP-PM", "trainable_params"] == 13
    assert (out["timing_runs"]["exclusion"] == "no node_load").all()  # not run by the launcher
    assert out["best_qnn"]["best"].tolist() == ["QNN-6"]


def test_latex_formatting():
    assert pm(12.345, 1.2, 1) == r"$12.3 \pm 1.2$"
    assert pm(0.5, float("nan"), 3) == r"$0.500 \pm \mbox{--}$"
    assert sci(0.0123) == "0.012" and sci(2.5e-7) == r"$2.5 \times 10^{-7}$"
    tex = tabular(["A", "B"], [["1", "2"], ["3", "4"]], "lr", "cap", "tab:x", groups=[1])
    assert tex.count(r"\midrule") == 2 and r"\toprule" in tex and "|" not in tex


def test_holm_adjustment():
    from qnnwind.stats import holm

    p = [0.01, 0.04, 0.03, float("nan"), 0.005]
    adj = holm(p)
    # m = 4 available: sorted 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, 0.06 (monotone)
    assert adj[4] == pytest.approx(0.02) and adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])
    assert holm([0.5, 0.9])[1] == 1.0  # capped at 1


def test_holm_columns_per_n_family(synthetic):
    from qnnwind.stats import holm

    tests = synthetic[3]["stats_tests"]
    for _, g in tests.groupby("n_train"):
        assert np.allclose(g["wilcoxon_p_holm"], holm(g["wilcoxon_p"].to_numpy()))
        assert np.allclose(g["dm_p_holm"], holm(g["dm_p_median"].to_numpy()))
        assert (g["dm_p_holm"] >= g["dm_p_median"] - 1e-15).all()
    t6 = (synthetic[0] / "outputs" / "tables" / "t6_tests.tex").read_text(encoding="utf-8")
    assert r"p_W^{\mathrm{Holm}}" in t6 and r"p_{DM}^{\mathrm{Holm}}" in t6


def test_merge_same_unitary():
    from make_figures import MERGED, merge_same_unitary

    frame = pd.DataFrame(
        {"model": ["QNN-1", "QNN-5", "LR"], "n_train": [750, 750, 750], "v": [1.0, 1.0, 2.0]}
    )
    notes: dict = {}
    merged = merge_same_unitary(frame, ["n_train"], ["v"], notes, "X")
    assert (
        sorted(merged["model"]) == sorted([MERGED, "LR"]) and notes["same_unitary_identical"]["X"]
    )
    different = frame.assign(v=[1.0, 1.5, 2.0])
    kept = merge_same_unitary(different, ["n_train"], ["v"], notes, "Y")
    assert (
        sorted(kept["model"]) == ["LR", "QNN-1", "QNN-5"]
        and not notes["same_unitary_identical"]["Y"]
    )
    single = merge_same_unitary(frame[frame["model"] != "QNN-1"], ["n_train"], ["v"], {}, "Z")
    assert MERGED in set(single["model"])  # a single one of them is labelled as the pair


def test_summary_settings_and_notes(synthetic):
    root = synthetic[0]
    summary = (root / "outputs" / "results_summary.md").read_text(encoding="utf-8")
    assert "## Analysis settings" in summary
    assert "0.9 × W = 57.6 (W = 64)" in summary
    assert "F4, F5, and A1 are drawn at N = 1500" in summary
    assert (
        "F4 and A2 pool the out-of-fold errors of all seeds (0, 1, 2); A1 shows seed 0" in summary
    )
    assert "Stability window: evaluations 1 to 50" in summary
    assert "the 4 baseline comparisons at each N form one family" in summary
    assert "'QNN-1/QNN-5 (same unitary)'" in summary and "A5: True" in summary
    assert "F1 series 'Best QNN, [-1, 1]': N = 750: QNN-5, N = 1500: QNN-5." in summary
    assert "WARNING" not in summary
    tables = root / "outputs" / "tables"
    t8 = (tables / "t8_complexity.tex").read_text(encoding="utf-8")
    assert "dual coefficients (one per support vector) plus the intercept" in t8
    for name in ("t3_main", "t3b_per_fold", "t4_stability", "t8_complexity"):
        text = (tables / f"{name}.tex").read_text(encoding="utf-8")
        assert "identical by construction" in text, name
