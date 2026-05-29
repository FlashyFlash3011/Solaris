# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for solaris/utils/profiling.py — CPU/no-op path."""

from solaris.utils.profiling import active_backend, mark, profile, solaris_profile


def test_active_backend_returns_string():
    backend = active_backend()
    assert backend in ("roctx", "nvtx", "none")


def test_solaris_profile_context_no_error():
    with solaris_profile("test_region"):
        result = 1 + 1
    assert result == 2


def test_solaris_profile_nested():
    with solaris_profile("outer"):
        with solaris_profile("inner"):
            pass  # must not raise


def test_profile_decorator():
    @profile("decorated_fn")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_profile_decorator_preserves_name():
    @profile("my_region")
    def my_function():
        pass

    assert my_function.__name__ == "my_function"


def test_mark_no_error():
    mark("checkpoint_event")  # must not raise


def test_solaris_profile_exception_propagates():
    """Exceptions inside the context still propagate; markers are cleaned up."""
    import pytest

    with pytest.raises(ValueError):
        with solaris_profile("error_region"):
            raise ValueError("test")
