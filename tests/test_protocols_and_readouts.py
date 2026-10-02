"""The random K-fold protocol, the readout variants QNN-1u..QNN-6u, the batch
configs, and the multi-batch analysis (T3c, T6 per readout and protocol, T9, F6, A5)."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import aggregate
import analyze
import synthetic_runs as syn
import test_slurm as slurm
from qnnwind.circuits import build_circuit, circuit_for, total_gates
from qnnwind.folds import check_random_fold, folds_for, make_random_folds
from qnnwind.io import load_config, read_json, run_dir
from qnnwind.runner import execute_run, readout_label, target_scaling
from qnnwind.stats import holm
from qnnwind.tasks import build_tasks
from test_readout import EXPERIMENT_CONFIG_HASH

from conftest import ROOT

N_ROWS = 4464


def cfg(name: str):
    return load_config(ROOT / "configs" / f"{name}.yaml")


# --------------------------------------------------------------------------------------
# Configs and task counts
# --------------------------------------------------------------------------------------


def test_experiment_config_and_task_counts(monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    config = cfg("experiment")
    assert config.hash == EXPERIMENT_CONFIG_HASH and config.protocol == "blocked"
    assert "variants" not in config["qnn"] and "random_split" not in config.raw
    counts = {g: len(t) for g, t in build_tasks(config).items()}
    assert counts == {"tuning": 216, "qnn": 720, "classical": 1320}


def test_blocked_unit_and_random_configs(monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    unit, random = cfg("blocked_unit"), cfg("random")
    assert unit.protocol == "blocked" and random.protocol == "random"
    assert len({unit.hash, random.hash, EXPERIMENT_CONFIG_HASH}) == 3
    assert {g: len(t) for g, t in build_tasks(unit).items()} == {
        "tuning": 0,
        "qnn": 720,
        "classical": 0,
    }
    assert {g: len(t) for g, t in build_tasks(random).items()} == {
        "tuning": 216,
        "qnn": 1440,
        "classical": 1320,
    }
    assert random["random_split"]["partition_seed"] == 20260928
    # Separate subtrees per batch; nothing else changes.
    for name, config in (("blocked_unit", unit), ("random", random)):
        for key in ("results", "logs", "tasks"):
            assert config.path(key) == (ROOT / key / name).resolve()
    full = cfg("experiment")
    for key in ("folds", "sizes", "seeds", "scaling", "tuning", "stats", "runtime", "data"):
        assert unit[key] == full[key] and random[key] == full[key], key
    base_qnn = {k: v for k, v in random["qnn"].items() if k != "variants"}
    assert base_qnn == full["qnn"]


def test_smoke_configs(monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    counts = {
        name: {g: len(t) for g, t in build_tasks(cfg(name)).items()}
        for name in ("smoke_blocked_unit", "smoke_random")
    }
    assert counts["smoke_blocked_unit"] == {"tuning": 0, "qnn": 6, "classical": 0}
    assert counts["smoke_random"] == {"tuning": 9, "qnn": 12, "classical": 11}
    assert cfg("smoke_random").protocol == "random"


# --------------------------------------------------------------------------------------
# Random K-fold protocol
# --------------------------------------------------------------------------------------


def test_random_folds_structure_and_seeds():
    folds = make_random_folds(N_ROWS, 6, 744, 464, 20260928)
    tests = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(tests), np.arange(N_ROWS))  # the test folds partition all rows
    permutation = np.random.default_rng(20260928).permutation(N_ROWS)
    for fold in folds:
        check_random_fold(fold, N_ROWS)
        assert (fold.test.size, fold.validation.size, fold.train_pool.size) == (744, 464, 3256)
        assert fold.buffers.size == 0
        assert np.array_equal(fold.test, np.sort(permutation[744 * fold.k : 744 * (fold.k + 1)]))
        non_test = np.setdiff1d(np.arange(N_ROWS), fold.test)
        rng = np.random.default_rng(20260928 + fold.k)
        expected = np.sort(rng.choice(non_test, size=464, replace=False))
        assert np.array_equal(fold.validation, expected)
    again = make_random_folds(N_ROWS, 6, 744, 464, 20260928)
    assert all(
        np.array_equal(a.validation, b.validation) for a, b in zip(folds, again, strict=True)
    )
    other = make_random_folds(N_ROWS, 6, 744, 464, 1)
    assert not np.array_equal(folds[0].test, other[0].test)


def test_folds_for_protocols(monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    random, full = cfg("random"), cfg("experiment")
    blocked = full.folds(N_ROWS)
    assert blocked[0].buffers.size > 0  # the blocked protocol keeps its buffers
    assert all(f.buffers.size == 0 for f in random.folds(N_ROWS))
    with pytest.raises(ValueError):
        folds_for(full["folds"], N_ROWS, "random", None)
    with pytest.raises(ValueError):
        folds_for(full["folds"], N_ROWS, "shuffled", {"partition_seed": 1})


# --------------------------------------------------------------------------------------
# Readout variants
# --------------------------------------------------------------------------------------


def structure(circuit) -> list:
    """Gates, qubits, and parameter names (Parameter objects differ between two builds)."""
    return [
        (inst.operation.name, [circuit.find_bit(q).index for q in inst.qubits],
         [str(p) for p in inst.operation.params])
        for inst in circuit.data
    ]  # fmt: skip


def test_variants_share_circuits_and_gates(monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    qnn_cfg = cfg("random")["qnn"]
    gates = total_gates(qnn_cfg)
    for k in range(1, 7):
        base = build_circuit(f"QNN-{k}", qnn_cfg["entanglement"][f"QNN-{k}"], qnn_cfg)
        variant = circuit_for(qnn_cfg, f"QNN-{k}u")
        assert variant.name == f"QNN-{k}u" and variant.entanglement == base.entanglement
        assert structure(variant.circuit) == structure(base.circuit)
        assert gates[f"QNN-{k}u"] == gates[f"QNN-{k}"]
    assert gates["QNN-1u"] == 40 and gates["QNN-5u"] == 34


def test_variant_target_scaling_and_labels(monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    random = cfg("random")
    assert target_scaling(random, "QNN-3u")["target_range"] == [0.0, 1.0]
    assert target_scaling(random, "QNN-3")["target_range"] == [-1.0, 1.0]
    assert target_scaling(random, "SVR")["target_range"] == [-1.0, 1.0]
    assert readout_label([0.0, 1.0]) == "[0, 1]" and readout_label([-1, 1]) == "[-1, 1]"


@pytest.mark.parametrize(("name", "model", "protocol", "readout"), [
    ("random", "QNN-6u", "random", "[0, 1]"),
    ("random", "LR", "random", "[-1, 1]"),
    ("blocked_unit", "QNN-6u", "blocked", "[0, 1]"),
])  # fmt: skip
def test_result_records_protocol_and_readout(tmp_path, dataset, monkeypatch, name, model,
                                             protocol, readout):  # fmt: skip
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    overrides = {"paths.results": str(tmp_path), "qnn.optimizer.maxiter": 1}
    config = load_config(ROOT / "configs" / f"{name}.yaml", overrides)
    execute_run(config, model, 60, 5, 0)
    directory = run_dir(tmp_path, model, 60, 5, 0)
    result = read_json(directory / "result.json")
    assert result["protocol"] == protocol and result["readout"] == readout
    fold = config.folds(dataset.n_rows)[5]
    test = pd.read_csv(directory / "preds_test.csv")
    assert np.array_equal(test["row"].to_numpy(), fold.test)  # this protocol's test fold
    if readout == "[0, 1]":
        assert result["scaling"]["target_range"] == [0.0, 1.0]


# --------------------------------------------------------------------------------------
# render.sh with a batch config
# --------------------------------------------------------------------------------------


@slurm.needs_bash
def test_render_with_batch_config(tmp_path):
    bash, _path = slurm.bash, slurm._path
    scratch = tmp_path.as_posix()
    env = {"PATH": _path(), "SCRATCH": scratch, "QNNWIND_CONFIG": "configs/random.yaml"}
    out = bash("slurm/render.sh slurm/qnn.sbatch m1234 regular 14:00:00 6", env)
    assert out.returncode == 0, out.stderr
    rendered = tmp_path / "qnn-wind-runs" / "jobs" / "random-qnn.sbatch"
    text = rendered.read_text(encoding="utf-8")
    assert 'CONFIG="${QNNWIND_CONFIG:-configs/random.yaml}"' in text
    assert "#SBATCH -N 6" in text and "#SBATCH -A m1234" in text
    plain = bash("slurm/render.sh slurm/qnn.sbatch m1234 regular 14:00:00 6",
                 {k: v for k, v in env.items() if k != "QNNWIND_CONFIG"})  # fmt: skip
    text = (tmp_path / "qnn-wind-runs" / "jobs" / "qnn.sbatch").read_text(encoding="utf-8")
    assert plain.returncode == 0 and "configs/experiment.yaml" in text


# --------------------------------------------------------------------------------------
# Multi-batch analysis
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def combined(tmp_path_factory, dataset):
    """Primary, blocked_unit, and random synthetic batches, analyzed together."""
    root = tmp_path_factory.mktemp("batches")
    configs = syn.build_all(root, ROOT)
    outputs = root / "combined"
    argv = []
    for name, path in configs.items():
        argv += ["--source", name, str(path), str(root)]
    assert analyze.main([*argv, "--outputs", str(outputs)]) == 0
    agg = outputs / "aggregated"
    tables = {
        protocol: {p.stem: pd.read_csv(p) for p in (agg / protocol).glob("*.csv")}
        for protocol in ("blocked", "random")
    }
    return root, outputs, agg, tables


def test_runs_record_batch_protocol_readout(combined):
    _, _, agg, t = combined
    blocked, random = t["blocked"]["runs"], t["random"]["runs"]
    assert set(blocked["batch"]) == {"primary", "blocked_unit"} and len(blocked) == 252 + 108
    assert (blocked["protocol"] == "blocked").all() and (random["protocol"] == "random").all()
    readout = blocked.set_index("model")["readout"]
    assert set(readout.loc[["QNN-1u", "QNN-5u", "QNN-6u"]]) == {"[0, 1]"}
    assert set(readout.loc[["QNN-1", "LR", "MLP-PM"]]) == {"[-1, 1]"}
    assert len(random) == 10 * 2 * 6 * 3 and set(random["batch"]) == {"random"}


def test_provenance_per_batch(combined):
    _, outputs, agg, _ = combined
    blocked = json.loads((agg / "blocked" / "provenance.json").read_text(encoding="utf-8"))
    random = json.loads((agg / "random" / "provenance.json").read_text(encoding="utf-8"))
    assert set(blocked) == {"primary", "blocked_unit"} and set(random) == {"random"}
    for prov in (*blocked.values(), *random.values()):
        assert prov["single_commit"] and prov["single_config_hash"] and prov["all_clean"]
    commits = {p["git_commits"][0] for p in (*blocked.values(), *random.values())}
    assert len(commits) == 3  # one commit per batch, not across batches
    summary = (outputs / "results_summary.md").read_text(encoding="utf-8")
    assert summary.count("One git commit: **PASS**") == 3 and "FAIL" not in summary
    completeness = json.loads((agg / "blocked" / "completeness.json").read_text(encoding="utf-8"))
    assert completeness["blocked_unit"]["expected"] == completeness["blocked_unit"]["found"] == 108


def test_best_qnn_per_readout_and_protocol(combined):
    t = combined[3]
    for protocol in ("blocked", "random"):
        best = t[protocol]["best_qnn"].set_index(["readout", "n_train"])
        for n in syn.SIZES:
            assert best.loc[("[-1, 1]", n), "best"] == "QNN-5"
            assert best.loc[("[0, 1]", n), "best"] == "QNN-5u"  # tie with QNN-1u: fewer gates
            assert best.loc[("[0, 1]", n), "tied_with"] == "QNN-1u"
            assert best.loc[("[0, 1]", n), "configs_compared"] == 3


def test_tests_holm_family_per_readout_and_protocol(combined):
    t = combined[3]
    for protocol in ("blocked", "random"):
        tests = t[protocol]["stats_tests"]
        assert len(tests) == 2 * 2 * 4  # readouts x sizes x baselines
        assert set(tests["qnn"]) == {"QNN-5", "QNN-5u"}
        for _, g in tests.groupby(["readout", "n_train"]):
            assert len(g) == 4
            assert np.allclose(g["wilcoxon_p_holm"], holm(g["wilcoxon_p"].to_numpy()))
            assert np.allclose(g["dm_p_holm"], holm(g["dm_p_median"].to_numpy()))


def test_timing_and_stability_per_readout(combined):
    t = combined[3]["blocked"]
    ratio = t["timing_ratio"]
    assert set(ratio["pair"]) == {"QNN-1 / QNN-5", "QNN-1u / QNN-5u"}
    assert np.allclose(ratio["ratio_mean"], 40 / 34)
    fits = t["timing_fits"].set_index(["readout", "model"])
    assert fits.loc[("[0, 1]", "QNN-6u"), "a"] == pytest.approx(syn.RATE["QNN-6u"])
    st = t["stability"].set_index(["readout", "config"])
    assert st.loc[("[-1, 1]", "QNN-6"), "SC"] == pytest.approx(1.0)
    assert st.loc[("[0, 1]", "QNN-6u"), "SC"] == pytest.approx(1.0)  # normalized per readout
    assert st.loc[("[0, 1]", "QNN-1u"), "rank"] == st.loc[("[0, 1]", "QNN-5u"), "rank"] == 1
    conv = t["convergence"]
    assert set(conv["readout"]) == {"[-1, 1]", "[0, 1]"}


def test_clipped_metrics(combined):
    root, _, _, t = combined
    runs = t["blocked"]["runs"]
    row = runs[(runs["model"] == "QNN-1u") & (runs["n_train"] == 750)].iloc[0]
    config = load_config(root / "blocked_unit.yaml")
    directory = run_dir(root / "results" / "blocked_unit", "QNN-1u", 750, row["fold"], row["seed"])
    test = pd.read_csv(directory / "preds_test.csv")
    clipped = test["pred_kw"].clip(0.0, row["y_max_kw"])
    rmse = float(np.sqrt(np.mean((clipped - test["actual_kw"]) ** 2)))
    assert row["test_clip_rmse"] == pytest.approx(rmse)
    assert row["test_frac_negative"] > 0 and config.protocol == "blocked"
    main = t["blocked"]["main"]
    assert {"test_clip_rmse_mean", "test_clip_r2_mean", "readout"} <= set(main.columns)


def test_protocol_comparison(combined):
    _, _, agg, t = combined
    cmp = pd.read_csv(agg / "protocol_comparison.csv")
    assert len(cmp) == 10 * 2  # models in both protocols x sizes
    assert np.allclose(
        cmp["rmse_random_minus_blocked"],
        cmp["test_rmse_mean_random"] - cmp["test_rmse_mean_blocked"],
    )
    assert (cmp["rmse_random_minus_blocked"] < 0).mean() > 0.8  # the random split is optimistic
    main = t["blocked"]["main"].set_index(["model", "n_train"])
    row = cmp.set_index(["model", "n_train"]).loc[("SVR", 1500)]
    assert row["test_rmse_mean_blocked"] == pytest.approx(main.loc[("SVR", 1500), "test_rmse_mean"])


def test_combined_tables_and_figures(combined):
    root, outputs, _, _ = combined
    tables = outputs / "tables"
    names = {p.stem for p in tables.glob("*.tex")}
    assert {"t3_main", "t3c_clipped", "t4_stability", "t5_timing", "t6_tests",
            "t6_tests_random", "t7_hyperparameters", "t7_hyperparameters_random",
            "t9_protocols"} <= names  # fmt: skip
    t3 = (tables / "t3_main.tex").read_text(encoding="utf-8")
    assert "QNN-1u" in t3 and "and so do QNN-1u and QNN-5u" in t3 and "[0, 1]" in t3
    t4 = (tables / "t4_stability.tex").read_text(encoding="utf-8")
    assert "Readout" in t4 and "within each readout" in t4
    t6 = (tables / "t6_tests.tex").read_text(encoding="utf-8")
    assert "at each N and readout as one family" in t6
    assert "Random cross-validation" in (tables / "t6_tests_random.tex").read_text(encoding="utf-8")
    t5 = (tables / "t5_timing.tex").read_text(encoding="utf-8")
    assert "QNN-1u / QNN-5u" in t5
    figures = outputs / "figures"
    for name in ("F1_rmse_r2_vs_n", "F2_qnn_mse_vs_iteration", "F3_qnn_timing",
                 "F4_error_distributions", "F5_rmse_vs_parameters", "F6_protocol_dumbbell",
                 "A1_actual_vs_predicted", "A2_error_distributions_all_n",
                 "A3_qnn_loss_curves", "A5_qnn_rmse_r2_vs_n"):  # fmt: skip
        assert (figures / f"{name}.pdf").is_file() and (figures / f"{name}.png").is_file(), name
    summary = (outputs / "results_summary.md").read_text(encoding="utf-8")
    assert "F1 series 'Best QNN, [0, 1]': N = 750: QNN-5u, N = 1500: QNN-5u." in summary
    assert "A5 (u): True" in summary and "WARNING" not in summary
    assert "F4 models (best QNN of each readout" in summary and "QNN-1u/QNN-5u" in summary
    assert "## Protocol comparison (T9, F6)" in summary
    assert "for one readout and one protocol" in summary
    assert "F3 shows the [-1, 1] readout" in summary
    # The primary outputs directory of the synthetic batches is not touched.
    assert not (root / "outputs").exists()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not available")
def test_combined_tables_compile(combined):
    import os

    tables = combined[1] / "tables"
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


def test_protocol_mismatch_is_refused(tmp_path, dataset, monkeypatch):
    configs = {"random": syn.build_batch(tmp_path, ROOT, "random")}
    directory = run_dir(tmp_path / "results" / "random", "LR", 750, 0, 0)
    result = read_json(directory / "result.json")
    (directory / "result.json").write_text(json.dumps({**result, "protocol": "blocked"}))
    source = aggregate.make_source("random", configs["random"], tmp_path)
    with pytest.raises(ValueError, match="protocol 'blocked' in a random batch"):
        aggregate.collect(source)


def test_several_batches_need_outputs(tmp_path):
    with pytest.raises(SystemExit):
        analyze.main(["--source", "a", "configs/experiment.yaml", str(tmp_path)])
    with pytest.raises(SystemExit):
        analyze.main([])
