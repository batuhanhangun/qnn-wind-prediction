"""The results package: built from raw results, it yields the same tables and figures, holds
no per-sample data except signed errors, and is reproducible byte for byte."""

from __future__ import annotations

import gzip
import io
import json

import numpy as np
import pandas as pd
import pytest

import aggregate
import analyze
import build_results_package as pkg
import compare_outputs
import synthetic_runs as syn
import verify_provenance

from conftest import ROOT


@pytest.fixture(scope="module")
def built(tmp_path_factory, dataset):
    """Synthetic raw batches, their package, and the analyses of both."""
    root = tmp_path_factory.mktemp("package")
    configs = syn.build_all(root, ROOT)
    raw_args = [
        a for name, path in configs.items() for a in ("--source", name, str(path), str(root))
    ]
    assert analyze.main([*raw_args, "--outputs", str(root / "from_raw")]) == 0
    sources = [aggregate.make_source(name, path, root) for name, path in configs.items()]
    manifest = pkg.build(sources, root / "package", check_hashes=False)
    pkg_args = [
        a
        for name, path in configs.items()
        for a in ("--source", name, str(path), str(root / "package" / name))
    ]
    assert analyze.main([*pkg_args, "--outputs", str(root / "from_package")]) == 0
    return root, configs, sources, manifest


def _read(path):
    return pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())), float_precision="round_trip")


def test_package_gives_identical_tables_and_figures(built):
    root, *_ = built
    identical, differences, _ = compare_outputs.compare(root / "from_package", root / "from_raw")
    assert not differences, differences
    assert any(p.startswith("tables/t6_tests") for p in identical)  # DM from errors only
    assert "figures/A1_actual_vs_predicted.png" in identical  # rebuilt with the dataset


def test_package_holds_only_signed_errors_per_sample(built):
    root, configs, _, manifest = built
    for f in manifest["files"]:
        columns = set(_read(root / "package" / f).columns)
        assert not columns & pkg.FORBIDDEN, (f, columns & pkg.FORBIDDEN)
        assert not columns & {"Temperature", "Pressure", "Theta", "Velocity", "Power"}
    errors = _read(root / "package" / "primary" / "errors_test.csv.gz")
    assert list(errors.columns) == ["model", "n_train", "fold", "seed", "row", "error_kw"]
    # The errors are exactly prediction minus actual of the raw runs.
    raw = aggregate.collect(aggregate.make_source("primary", configs["primary"], root))
    merged = raw["predictions"].merge(errors, on=["model", "n_train", "fold", "seed", "row"])
    assert len(merged) == len(errors) == len(raw["predictions"])
    assert np.array_equal(merged["error_kw_x"].to_numpy(), merged["error_kw_y"].to_numpy())


def test_package_is_reproducible(built, tmp_path):
    root, _, sources, manifest = built
    again = pkg.build(sources, tmp_path / "package", check_hashes=False)
    assert again["files"] == manifest["files"]
    assert (tmp_path / "package" / "manifest.json").read_bytes() == (
        root / "package" / "manifest.json"
    ).read_bytes()


def test_build_rejects_runs_of_another_configuration(built, tmp_path):
    _, _, sources, _ = built
    with pytest.raises(SystemExit):  # synthetic runs record a different config hash
        pkg.build(sources[:1], tmp_path / "package")


def test_verify_provenance_of_the_results_package(capsys):
    assert verify_provenance.main([]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    assert out.count("[ OK ]") >= 2 * len(manifest["batches"]) + 1
    assert "all checks passed" in out
