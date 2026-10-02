"""F0 (evaluation layout): the plotted role of every sample equals the fold assignment."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import make_figures as mf
from qnnwind.io import load_config

from conftest import ROOT


def plotted_roles(fig: plt.Figure, n_rows: int) -> dict[str, np.ndarray]:
    """Role index of every sample per bar, rebuilt from the drawn collections (gid
    ``F0:<protocol>:<fold>:<role index>``); -1 where nothing was drawn."""
    bars: dict[str, np.ndarray] = {}
    for ax in fig.axes:
        for coll in ax.collections:
            gid = coll.get_gid() or ""
            if not gid.startswith("F0:"):
                continue
            key, role = gid.rsplit(":", 1)
            roles = bars.setdefault(key, np.full(n_rows, -1, dtype=np.int64))
            for path in coll.get_paths():
                x = path.vertices[:, 0]
                start, stop = int(round(x.min() + 0.5)), int(round(x.max() + 0.5))
                assert np.all(roles[start:stop] == -1), f"{key}: rows {start}..{stop} drawn twice"
                roles[start:stop] = int(role)
    return bars


def expected(fold, n_rows: int) -> np.ndarray:
    out = np.full(n_rows, -1, dtype=np.int64)
    for role, rows in (
        ("training pool", fold.train_pool),
        ("validation", fold.validation),
        ("buffer (unused)", fold.buffers),
        ("test", fold.test),
    ):
        out[rows] = mf.ROLES.index(role)
    return out


@pytest.mark.parametrize("with_random", [True, False])
def test_f0_plotted_roles_equal_fold_assignments(with_random):
    blocked_cfg = load_config(ROOT / "configs" / "experiment.yaml")
    random_cfg = load_config(ROOT / "configs" / "random.yaml")
    n_rows = int(blocked_cfg["data"]["n_rows"])
    blocked = blocked_cfg.folds(n_rows)
    random_fold = random_cfg.folds(n_rows)[0] if with_random else None
    mf.apply_style()
    fig = mf.f0_figure(blocked, random_fold, n_rows)
    try:
        bars = plotted_roles(fig, n_rows)
        want = {f"F0:blocked:{f.k}": expected(f, n_rows) for f in blocked}
        if with_random:
            want["F0:random:0"] = expected(random_fold, n_rows)
        assert set(bars) == set(want)
        for key, roles in bars.items():
            assert np.all(roles >= 0), f"{key}: samples without a drawn role"
            np.testing.assert_array_equal(roles, want[key], err_msg=key)
        # the random bar has no buffer rows and interleaves its roles
        if with_random:
            random_roles = bars["F0:random:0"]
            assert mf.ROLES.index("buffer (unused)") not in set(random_roles)
            assert np.count_nonzero(np.diff(random_roles)) > 1000
        # fold 0 is the top bar of panel (a)
        ax = fig.axes[0]
        assert ax.get_ylim()[0] > ax.get_ylim()[1]
        assert ax.get_yticklabels()[0].get_text() == "Fold 0"
        assert fig.get_size_inches()[0] == pytest.approx(mf.DOUBLE)
    finally:
        plt.close(fig)


def test_fold_roles_rejects_overlaps():
    cfg = load_config(ROOT / "configs" / "experiment.yaml")
    n_rows = int(cfg["data"]["n_rows"])
    fold = cfg.folds(n_rows)[2]
    np.testing.assert_array_equal(mf.fold_roles(fold, n_rows), expected(fold, n_rows))
    broken = type(fold)(fold.k, fold.test, fold.test[:5], fold.buffers, fold.train_pool)
    with pytest.raises(ValueError):
        mf.fold_roles(broken, n_rows)
