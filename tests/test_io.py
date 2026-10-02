"""Config loading, LF-only writing, and run-directory handling."""

from __future__ import annotations

import pandas as pd
import pytest

from qnnwind.io import (
    add_run_metadata,
    finalize_dir,
    load_config,
    parse_override,
    read_json,
    sqlite_url,
    staging_dir,
    validate_run_dir,
    write_csv,
    write_json,
    write_text,
)

from conftest import ROOT


def test_smoke_inherits_experiment(monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)  # set inside the cluster container
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    full = load_config(ROOT / "configs" / "experiment.yaml")
    assert smoke["folds"]["run"] == [5] and smoke["sizes"] == [100] and smoke["seeds"] == [0]
    assert smoke["qnn"]["optimizer"] == {"maxiter": 3, "gtol": 1e-12}
    assert smoke["tuning"]["n_trials"] == 2
    assert smoke["tuning"]["search_spaces"] == full["tuning"]["search_spaces"]
    assert smoke["folds"]["buffer"] == full["folds"]["buffer"]
    assert smoke.hash != full.hash
    assert smoke.path("results") == (ROOT / "results" / "smoke").resolve()
    monkeypatch.setenv("QNNWIND_SCRATCH", str(ROOT / "scratch"))
    assert smoke.path("results") == ROOT / "scratch" / "results" / "smoke"
    assert full.path("logs") == ROOT / "scratch" / "logs"


def test_overrides():
    cfg = load_config(
        ROOT / "configs" / "experiment.yaml", dict([parse_override("qnn.optimizer.maxiter=5")])
    )
    assert cfg["qnn"]["optimizer"]["maxiter"] == 5
    with pytest.raises(KeyError):
        load_config(ROOT / "configs" / "experiment.yaml", {"qnn.optimizer.nope": 1})


def test_env_var_paths(monkeypatch):
    monkeypatch.setenv("QNNWIND_TEST_DIR", str(ROOT / "somewhere"))
    cfg = load_config(
        ROOT / "configs" / "experiment.yaml", {"paths.results": "${QNNWIND_TEST_DIR}/r"}
    )
    assert cfg.path("results") == ROOT / "somewhere" / "r"


def test_writers_use_lf(tmp_path):
    write_text(tmp_path / "a.txt", "one\ntwo\n")
    write_json(tmp_path / "b.json", {"x": [1, 2]})
    write_csv(tmp_path / "c.csv", pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]}))
    for name in ("a.txt", "b.json", "c.csv"):
        content = (tmp_path / name).read_bytes()
        assert b"\r\n" not in content and b"\n" in content


def test_sqlite_url_is_posix(tmp_path):
    url = sqlite_url(tmp_path / "study.db")
    assert url.startswith("sqlite:///") and "\\" not in url


def test_run_dir_lifecycle(tmp_path):
    final = tmp_path / "seed0"
    files = ("result.json", "preds_val.csv")
    assert not validate_run_dir(final, files)
    stage = staging_dir(final)
    write_json(stage / "result.json", {"status": "complete"})
    write_text(stage / "preds_val.csv", "row\n")
    finalize_dir(stage, final)
    assert validate_run_dir(final, files)
    # An incomplete directory is kept (renamed), never overwritten.
    write_json(final / "result.json", {"status": "running"})
    assert not validate_run_dir(final, files)
    stage = staging_dir(final)
    write_json(stage / "result.json", {"status": "complete"})
    write_text(stage / "preds_val.csv", "row\n")
    finalize_dir(stage, final)
    assert validate_run_dir(final, files)
    assert len(list(tmp_path.glob("seed0.invalid-*"))) == 1


def test_add_run_metadata(tmp_path):
    result_file = tmp_path / "result.json"
    write_json(result_file, {"status": "complete", "metadata": {"a": 1}})
    add_run_metadata(result_file, "node_load", {"busy_mean": 63.5})
    content = result_file.read_bytes()
    assert b"\r\n" not in content
    assert read_json(result_file)["metadata"] == {"a": 1, "node_load": {"busy_mean": 63.5}}
    with pytest.raises(KeyError):
        add_run_metadata(result_file, "a", 2)  # never replaces an existing key
    assert list(tmp_path.iterdir()) == [result_file]  # no temporary file left behind
