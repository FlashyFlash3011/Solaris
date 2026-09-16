"""Tests for leakage-free dataset splitting."""

import numpy as np
import pytest

from solaris.datapipes import split_by_group


def test_groups_never_straddle_the_split():
    groups = np.repeat(np.arange(20), 5)  # 20 simulations x 5 samples each
    train, val = split_by_group(groups, val_fraction=0.2, seed=0)
    assert set(groups[train]).isdisjoint(groups[val])


def test_split_covers_every_sample_exactly_once():
    groups = np.repeat(np.arange(10), 3)
    train, val = split_by_group(groups, val_fraction=0.3, seed=1)
    assert sorted(np.concatenate([train, val])) == list(range(30))


def test_val_fraction_counts_groups():
    groups = np.repeat(np.arange(10), 4)
    _, val = split_by_group(groups, val_fraction=0.2, seed=0)
    assert len(np.unique(groups[val])) == 2


def test_split_is_deterministic_for_a_seed():
    groups = np.repeat(np.arange(8), 2)
    a = split_by_group(groups, 0.25, seed=3)
    b = split_by_group(groups, 0.25, seed=3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_at_least_one_group_on_each_side():
    groups = np.repeat(np.arange(3), 2)
    train, val = split_by_group(groups, val_fraction=0.01, seed=0)
    assert len(val) > 0 and len(train) > 0


def test_needs_two_groups():
    with pytest.raises(ValueError):
        split_by_group(np.zeros(5, dtype=int), 0.2)
