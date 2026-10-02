"""Blocked folds with buffers and seeded training subsets."""

from __future__ import annotations

import numpy as np
import pytest

from qnnwind.folds import (
    assert_disjoint_from_test,
    check_fold,
    make_folds,
    min_distance,
    ranges,
    training_subset,
)

N_ROWS, K, SIZE, B, V = 4464, 6, 744, 72, 464
SIZES = (750, 1500, 2250, 3000)

# Expected row ranges (inclusive), B = 72.
EXPECTED = {
    0: {
        "test": [(0, 743)],
        "validation": [(816, 1279)],
        "buffers": [(744, 815), (1280, 1351)],
        "train_pool": [(1352, 4463)],
        "pool_size": 3112,
    },
    1: {
        "test": [(744, 1487)],
        "validation": [(208, 671)],
        "buffers": [(136, 207), (672, 743), (1488, 1559)],
        "train_pool": [(0, 135), (1560, 4463)],
        "pool_size": 3040,
    },
    2: {
        "test": [(1488, 2231)],
        "validation": [(952, 1415)],
        "buffers": [(880, 951), (1416, 1487), (2232, 2303)],
        "train_pool": [(0, 879), (2304, 4463)],
        "pool_size": 3040,
    },
    3: {
        "test": [(2232, 2975)],
        "validation": [(1696, 2159)],
        "buffers": [(1624, 1695), (2160, 2231), (2976, 3047)],
        "train_pool": [(0, 1623), (3048, 4463)],
        "pool_size": 3040,
    },
    4: {
        "test": [(2976, 3719)],
        "validation": [(2440, 2903)],
        "buffers": [(2368, 2439), (2904, 2975), (3720, 3791)],
        "train_pool": [(0, 2367), (3792, 4463)],
        "pool_size": 3040,
    },
    5: {
        "test": [(3720, 4463)],
        "validation": [(3184, 3647)],
        "buffers": [(3112, 3183), (3648, 3719)],
        "train_pool": [(0, 3111)],
        "pool_size": 3112,
    },
}


@pytest.fixture(scope="module")
def all_folds():
    return make_folds(N_ROWS, K, SIZE, B, V)


def test_config_matches_spec(config):
    folds_cfg = config["folds"]
    assert (folds_cfg["n_folds"], folds_cfg["fold_size"], folds_cfg["buffer"]) == (K, SIZE, B)
    assert folds_cfg["validation_size"] == V
    assert config["sizes"] == list(SIZES)
    assert max(SIZES) <= min(t["pool_size"] for t in EXPECTED.values())
    assert config["seeds"] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("k", range(K))
def test_row_ranges_equal_spec_table(all_folds, k):
    fold, expected = all_folds[k], EXPECTED[k]
    for block in ("test", "validation", "buffers", "train_pool"):
        assert ranges(getattr(fold, block)) == expected[block], block
    assert fold.train_pool.size == expected["pool_size"]


def test_test_folds_partition_all_rows(all_folds):
    tests = np.concatenate([f.test for f in all_folds])
    assert np.array_equal(np.sort(tests), np.arange(N_ROWS))
    assert tests.size == N_ROWS


@pytest.mark.parametrize("k", range(K))
def test_blocks_disjoint_cover_and_buffered(all_folds, k):
    fold = all_folds[k]
    check_fold(fold, N_ROWS, B)  # raises on any violation
    blocks = [fold.train_pool, fold.validation, fold.test, fold.buffers]
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.intersect1d(blocks[i], blocks[j]).size == 0
    guarded = np.concatenate([fold.test, fold.validation])
    assert min_distance(fold.train_pool, guarded) > B
    assert min_distance(fold.validation, fold.test) > B
    assert fold.validation.size == V and fold.test.size == SIZE


def test_check_fold_detects_violation(all_folds):
    fold = all_folds[2]
    leaky = type(fold)(
        k=fold.k,
        test=fold.test,
        validation=fold.validation,
        buffers=fold.buffers[1:],
        train_pool=np.sort(np.append(fold.train_pool, fold.buffers[0])),
    )
    with pytest.raises(AssertionError):
        check_fold(leaky, N_ROWS, B)


def test_min_distance():
    assert min_distance(np.array([0, 10]), np.array([4, 30])) == 4
    assert min_distance(np.array([13]), np.array([0, 25])) == 12


@pytest.mark.parametrize("k", range(K))
def test_training_subsets_nested_seeded_and_from_pool(all_folds, k):
    fold = all_folds[k]
    for seed in range(5):
        permutation = np.random.default_rng(seed).permutation(fold.train_pool)
        subsets = [training_subset(fold, n, seed) for n in SIZES]
        for n, subset in zip(SIZES, subsets, strict=True):
            assert subset.size == n
            assert np.unique(subset).size == n
            assert np.isin(subset, fold.train_pool).all()
            assert np.array_equal(subset, permutation[:n])
        for small, large in zip(subsets, subsets[1:], strict=False):
            assert np.array_equal(large[: small.size], small)
    assert not np.array_equal(training_subset(fold, 750, 0), training_subset(fold, 750, 1))


def test_training_subset_rejects_oversized(all_folds):
    with pytest.raises(ValueError):
        training_subset(all_folds[1], 3041, 0)


def test_disjoint_guard(all_folds):
    fold = all_folds[3]
    assert_disjoint_from_test(fold.train_pool, fold, "pool")
    with pytest.raises(AssertionError):
        assert_disjoint_from_test(np.array([fold.test[5]]), fold, "leak")
