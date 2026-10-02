"""Blocked folds with buffers and seeded training subsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """Row indices (0-based, row order of the file) of one blocked fold.

    Attributes:
        k: Fold number.
        test: Test rows (contiguous).
        validation: Validation rows (contiguous).
        buffers: Unused rows around the test fold and on the far side of validation.
        train_pool: All remaining rows, in increasing order.
    """

    k: int
    test: np.ndarray
    validation: np.ndarray
    buffers: np.ndarray
    train_pool: np.ndarray


def _span(start: int, stop: int) -> np.ndarray:
    """Rows ``start`` to ``stop - 1``, clipped to be non-empty only when ``stop > start``."""
    return np.arange(start, max(start, stop), dtype=np.int64)


def make_fold(
    k: int, n_rows: int, n_folds: int, fold_size: int, buffer: int, val_size: int
) -> Fold:
    """Build blocked fold ``k``.

    The test fold is rows ``fold_size*k`` to ``fold_size*(k+1) - 1``, with ``buffer`` rows on
    each side. Validation is the ``val_size`` rows immediately before the leading buffer (for
    fold 0: immediately after the trailing buffer), followed on its far side by another
    ``buffer`` rows. Every other row is in the training pool.
    """
    if n_folds * fold_size != n_rows:
        raise ValueError(f"{n_folds} folds of {fold_size} rows do not cover {n_rows} rows")
    if not 0 <= k < n_folds:
        raise ValueError(f"Fold {k} out of range 0..{n_folds - 1}")

    test_start, test_stop = fold_size * k, fold_size * (k + 1)
    test = _span(test_start, test_stop)
    buffer_parts = [
        _span(max(0, test_start - buffer), test_start),
        _span(test_stop, min(n_rows, test_stop + buffer)),
    ]
    if k == 0:
        val_start = test_stop + buffer
        val_stop = val_start + val_size
        buffer_parts.append(_span(val_stop, val_stop + buffer))
    else:
        val_stop = test_start - buffer
        val_start = val_stop - val_size
        buffer_parts.append(_span(val_start - buffer, val_start))
    if val_start - buffer < 0 or val_stop + buffer > n_rows:
        raise ValueError(f"Fold {k}: validation block and its buffer do not fit in the data")
    validation = _span(val_start, val_stop)
    buffers = np.sort(np.concatenate(buffer_parts))

    used = np.zeros(n_rows, dtype=bool)
    for part in (test, validation, buffers):
        used[part] = True
    train_pool = np.flatnonzero(~used).astype(np.int64)
    return Fold(k=k, test=test, validation=validation, buffers=buffers, train_pool=train_pool)


def make_folds(n_rows: int, n_folds: int, fold_size: int, buffer: int, val_size: int) -> list[Fold]:
    """All ``n_folds`` folds."""
    return [make_fold(k, n_rows, n_folds, fold_size, buffer, val_size) for k in range(n_folds)]


def folds_from_config(folds_cfg: dict, n_rows: int) -> list[Fold]:
    """All folds from the ``folds`` config section."""
    return make_folds(
        n_rows=n_rows,
        n_folds=folds_cfg["n_folds"],
        fold_size=folds_cfg["fold_size"],
        buffer=folds_cfg["buffer"],
        val_size=folds_cfg["validation_size"],
    )


def make_random_folds(
    n_rows: int, n_folds: int, fold_size: int, val_size: int, partition_seed: int
) -> list[Fold]:
    """Random K-fold protocol: no blocks, no buffers.

    * Test folds: one permutation of all rows drawn with ``default_rng(partition_seed)``; fold
      k is entries ``fold_size*k`` to ``fold_size*(k+1) - 1`` of that permutation.
    * Validation of fold k: ``val_size`` rows drawn without replacement from the non-test rows
      with ``default_rng(partition_seed + k)``.
    * Training pool: the remaining rows. Every array is returned sorted.
    """
    if n_folds * fold_size != n_rows:
        raise ValueError(f"{n_folds} folds of {fold_size} rows do not cover {n_rows} rows")
    permutation = np.random.default_rng(partition_seed).permutation(n_rows)
    folds = []
    for k in range(n_folds):
        test = np.sort(permutation[fold_size * k : fold_size * (k + 1)]).astype(np.int64)
        non_test = np.setdiff1d(np.arange(n_rows), test)
        rng = np.random.default_rng(partition_seed + k)
        validation = np.sort(rng.choice(non_test, size=val_size, replace=False)).astype(np.int64)
        train_pool = np.setdiff1d(non_test, validation).astype(np.int64)
        folds.append(
            Fold(k=k, test=test, validation=validation, buffers=np.empty(0, np.int64),
                 train_pool=train_pool)
        )  # fmt: skip
    return folds


def folds_for(
    folds_cfg: dict, n_rows: int, protocol: str = "blocked", random_cfg: dict | None = None
) -> list[Fold]:
    """The folds of an evaluation protocol: ``blocked`` or ``random``."""
    if protocol == "blocked":
        return folds_from_config(folds_cfg, n_rows)
    if protocol == "random":
        if not random_cfg or "partition_seed" not in random_cfg:
            raise ValueError("The random protocol needs random_split.partition_seed")
        return make_random_folds(
            n_rows,
            folds_cfg["n_folds"],
            folds_cfg["fold_size"],
            folds_cfg["validation_size"],
            int(random_cfg["partition_seed"]),
        )
    raise ValueError(f"Unknown protocol {protocol!r}")


def check_random_fold(fold: Fold, n_rows: int) -> None:
    """Structural guarantees of the random protocol: disjoint blocks covering every row."""
    parts = (fold.train_pool, fold.validation, fold.test)
    everything = np.concatenate(parts)
    assert fold.buffers.size == 0, f"fold {fold.k}: the random protocol has no buffers"
    assert everything.size == n_rows, f"fold {fold.k}: blocks overlap or miss rows"
    assert np.array_equal(np.sort(everything), np.arange(n_rows)), f"fold {fold.k}: not a cover"


def training_subset(fold: Fold, n_train: int, seed: int) -> np.ndarray:
    """The training rows for size ``n_train`` and ``seed``.

    One permutation of the fold's training pool is drawn with
    ``np.random.default_rng(seed)``; the training set is its first ``n_train`` entries, in
    permutation order. Subsets are therefore nested across ``n_train`` for a fixed seed.
    """
    if not 0 < n_train <= fold.train_pool.size:
        raise ValueError(f"N={n_train} not in 1..{fold.train_pool.size} for fold {fold.k}")
    permutation = np.random.default_rng(seed).permutation(fold.train_pool)
    return permutation[:n_train]


def ranges(rows: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (first, last) ranges of the contiguous runs in sorted ``rows``."""
    if rows.size == 0:
        return []
    rows = np.sort(rows)
    breaks = np.flatnonzero(np.diff(rows) != 1)
    starts = np.concatenate(([rows[0]], rows[breaks + 1]))
    ends = np.concatenate((rows[breaks], [rows[-1]]))
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def check_fold(fold: Fold, n_rows: int, buffer: int) -> None:
    """Assert the structural guarantees of the blocked protocol for one fold.

    Raises:
        AssertionError: if blocks overlap, do not cover all rows, or a training row lies
            within ``buffer`` rows of a test or validation row, or validation and test are
            closer than ``buffer`` rows.
    """
    parts = (fold.train_pool, fold.validation, fold.test, fold.buffers)
    everything = np.concatenate(parts)
    assert everything.size == n_rows, f"fold {fold.k}: blocks overlap or miss rows"
    assert np.array_equal(np.sort(everything), np.arange(n_rows)), f"fold {fold.k}: not a cover"

    guarded = np.concatenate((fold.test, fold.validation))
    distance = min_distance(fold.train_pool, guarded)
    assert distance > buffer, f"fold {fold.k}: training row within {buffer} rows of val/test"
    gap = min_distance(fold.validation, fold.test)
    assert gap > buffer, f"fold {fold.k}: validation and test separated by < {buffer} rows"


def min_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Smallest ``|i - j|`` over rows ``i`` in ``a`` and ``j`` in ``b``."""
    b = np.sort(b)
    pos = np.searchsorted(b, a)
    left = b[np.clip(pos - 1, 0, b.size - 1)]
    right = b[np.clip(pos, 0, b.size - 1)]
    return int(np.minimum(np.abs(a - left), np.abs(a - right)).min())


def assert_disjoint_from_test(rows: np.ndarray, fold: Fold, what: str) -> None:
    """Guard used by every fit, tune, scale, and select entry point.

    Raises:
        AssertionError: if any of ``rows`` belongs to the fold's test block.
    """
    overlap = np.intersect1d(rows, fold.test)
    assert overlap.size == 0, f"{what}: {overlap.size} test-fold rows of fold {fold.k} passed in"
