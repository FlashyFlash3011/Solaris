"""Tests for the config-keyed dataset cache."""

import numpy as np
import pytest

from solaris.datapipes import CacheMismatchError, cached_arrays


def _gen(n):
    calls = []

    def generate():
        calls.append(1)
        return {"x": np.arange(n, dtype=np.float32)}

    return generate, calls


def test_generates_then_reuses(tmp_path):
    path = tmp_path / "d.npz"
    gen, calls = _gen(3)
    a = cached_arrays(path, {"n": 3}, gen)
    b = cached_arrays(path, {"n": 3}, gen)
    assert len(calls) == 1
    np.testing.assert_array_equal(a["x"], b["x"])


def test_config_change_raises_instead_of_silently_reusing(tmp_path):
    path = tmp_path / "d.npz"
    cached_arrays(path, {"n": 3}, _gen(3)[0])
    with pytest.raises(CacheMismatchError, match="n"):
        cached_arrays(path, {"n": 5}, _gen(5)[0])


def test_overwrite_regenerates_on_mismatch(tmp_path):
    path = tmp_path / "d.npz"
    cached_arrays(path, {"n": 3}, _gen(3)[0])
    gen, calls = _gen(5)
    out = cached_arrays(path, {"n": 5}, gen, overwrite_on_mismatch=True)
    assert len(calls) == 1
    assert out["x"].shape == (5,)


def test_legacy_cache_without_config_is_used_with_warning(tmp_path):
    path = tmp_path / "d.npz"
    np.savez(path, x=np.zeros(2))
    gen, calls = _gen(2)
    with pytest.warns(UserWarning, match="no recorded config"):
        out = cached_arrays(path, {"n": 2}, gen)
    assert not calls
    assert "x" in out
