"""Reproduction helpers: run directory default, output comparison, deterministic figures,
and the batch lookup of single reruns."""

from __future__ import annotations

import os
import zlib

import matplotlib.pyplot as plt
import pytest

import compare_outputs
import make_figures as mf
import rerun_single
import runs_dir

from conftest import ROOT


def test_runs_dir_default_env_and_argument(tmp_path, monkeypatch):
    monkeypatch.delenv("QNNWIND_SCRATCH", raising=False)
    assert runs_dir.use_runs_dir() == ROOT / "runs"
    monkeypatch.setenv("QNNWIND_SCRATCH", str(tmp_path / "env"))
    assert runs_dir.use_runs_dir() == tmp_path / "env"
    assert runs_dir.use_runs_dir(tmp_path / "arg") == (tmp_path / "arg").resolve()
    assert os.environ["QNNWIND_SCRATCH"] == str((tmp_path / "arg").resolve())


def _figure(path_dir, name):
    mf.apply_style()
    fig, ax = plt.subplots(figsize=(mf.SINGLE, 2.0))
    ax.plot([0, 1, 2], [1, 3, 2], label="series")
    ax.legend()
    return mf.save(fig, path_dir, name)


def test_figures_are_deterministic(tmp_path):
    first = _figure(tmp_path / "a", "fig")
    second = _figure(tmp_path / "b", "fig")
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name


def test_compare_outputs_reports_each_kind(tmp_path):
    new, ref = tmp_path / "new", tmp_path / "ref"
    for root in (new, ref):
        (root / "tables").mkdir(parents=True)
        (root / "tables" / "same.tex").write_text("a\nb\n", encoding="utf-8")
    (new / "tables" / "changed.tex").write_text("a\nx\n", encoding="utf-8")
    (ref / "tables" / "changed.tex").write_text("a\nb\n", encoding="utf-8")
    (ref / "tables" / "only_ref.tex").write_text("", encoding="utf-8")
    (new / "tables" / "only_new.tex").write_text("", encoding="utf-8")
    _figure(new / "figures", "f")
    mf.apply_style()
    fig, ax = plt.subplots(figsize=(mf.SINGLE, 2.0))
    ax.plot([0, 1, 2], [1, 3, 3], label="series")  # one point moved
    ax.legend()
    mf.save(fig, ref / "figures", "f")

    identical, differences, equivalent = compare_outputs.compare(new, ref)
    assert identical == ["tables/same.tex"]
    kinds = {d.path: (d.kind, d.detail) for d in differences}
    assert kinds["tables/only_ref.tex"][0] == "missing"
    assert kinds["tables/only_new.tex"][0] == "extra"
    assert kinds["tables/changed.tex"] == ("differs", "line 2: 'x' (reference: 'b')")
    assert "of pixels differ" in kinds["figures/f.png"][1]
    assert kinds["figures/f.pdf"][0] == "differs"
    assert compare_outputs.main([str(new), str(new)]) == 0


@pytest.mark.parametrize(
    ("protocol", "model", "config"),
    [
        ("blocked", "QNN-3", "experiment.yaml"),
        ("blocked", "LSTM", "experiment.yaml"),
        ("blocked", "QNN-3u", "blocked_unit.yaml"),
        ("random", "QNN-3u", "random.yaml"),
        ("random", "SVR", "random.yaml"),
    ],
)
def test_rerun_uses_the_batch_config(protocol, model, config):
    assert rerun_single.config_for(protocol, model).name == config


def test_rerun_rejects_unknown_model():
    with pytest.raises(SystemExit):
        rerun_single.config_for("blocked", "QNN-7")


def test_missing_dataset_stops_with_a_clear_message(tmp_path):
    from aggregate import require_dataset
    from qnnwind.io import load_config

    config = load_config(
        ROOT / "configs" / "smoke.yaml", {"paths.data": str(tmp_path / "missing.csv")}
    )
    with pytest.raises(SystemExit, match="data/README.md"):
        require_dataset(config)


def _pdf(content: bytes) -> bytes:
    stream = zlib.compress(content)
    head = b"%PDF-1.4\n1 0 obj\n<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
    return head + stream + b"\nendstream\nendobj\nxref\n0 2\ntrailer\n<< >>\nstartxref\n42\n%%EOF\n"


def test_cross_platform_equivalent_figures(tmp_path):
    """Same pixels or same drawing content (up to negative zero) is equivalent; a real change
    is not."""
    from PIL import Image

    new, ref = tmp_path / "new" / "figures", tmp_path / "ref" / "figures"
    new.mkdir(parents=True)
    ref.mkdir(parents=True)
    pixels = Image.new("RGB", (40, 30), (200, 10, 10))
    pixels.save(new / "a.png", compress_level=1)
    pixels.save(ref / "a.png", compress_level=9)
    (new / "b.pdf").write_bytes(_pdf(b"q\n-0 -2.828427 m\n1 0 l\nS\nQ\n"))
    (ref / "b.pdf").write_bytes(_pdf(b"q\n0 -2.828427 m\n1 0 l\nS\nQ\n"))
    (new / "c.pdf").write_bytes(_pdf(b"q\n0 -2.8 m\n1 0 l\nS\nQ\n"))
    (ref / "c.pdf").write_bytes(_pdf(b"q\n0 -2.828427 m\n1 0 l\nS\nQ\n"))
    # Anti-aliasing: 2/255 in a few pixels is equivalent only when the figure's PDF matches.
    shaded = pixels.copy()
    shaded.putpixel((5, 5), (198, 10, 10))
    for name, pdf_new in (("d", b"q\n0 0 m\nQ\n"), ("e", b"q\n1 1 m\nQ\n")):
        shaded.save(new / f"{name}.png")
        pixels.save(ref / f"{name}.png")
        (new / f"{name}.pdf").write_bytes(_pdf(pdf_new))
        (ref / f"{name}.pdf").write_bytes(_pdf(b"q\n0 0 m\nQ\n"))
    assert (new / "a.png").read_bytes() != (ref / "a.png").read_bytes()
    identical, differences, equivalent = compare_outputs.compare(new.parent, ref.parent)
    assert identical == ["figures/d.pdf"]
    assert equivalent == ["figures/a.png", "figures/b.pdf", "figures/d.png"]
    assert [d.path for d in differences] == ["figures/c.pdf", "figures/e.pdf", "figures/e.png"]


def _pdf_indirect(content: bytes, length_value: int) -> bytes:
    stream = zlib.compress(content)
    return (
        b"%PDF-1.4\n1 0 obj\n<< /Length 2 0 R >>\nstream\n" + stream + b"\nendstream\nendobj\n"
        b"2 0 obj\n" + str(length_value).encode() + b"\nendobj\nxref\n0 3\n%%EOF\n"
    )


def test_pdf_indirect_stream_lengths():
    same = compare_outputs._pdf_content(_pdf_indirect(b"0 0 m\n", 11))
    assert same == compare_outputs._pdf_content(_pdf_indirect(b"-0 0 m\n", 12))
    assert same != compare_outputs._pdf_content(_pdf_indirect(b"1 0 m\n", 11))
