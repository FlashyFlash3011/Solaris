# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Profiling markers for ROCm (ROCTx) and CUDA (NVTX) backends.

Provides a unified context manager and decorator that annotates GPU profiler
traces with named regions.  When used with AMD ``rocprof`` or NVIDIA Nsight
Systems, each region appears as a labelled span in the trace timeline.

Backend selection
-----------------
* **ROCm** — uses ``roctx`` via ``ctypes`` (part of the ROCm ``roctracer``
  package).  Detected automatically when PyTorch reports HIP is available.
* **CUDA** — uses ``torch.cuda.nvtx`` (bundled with PyTorch CUDA builds).
* **CPU / fallback** — all calls are no-ops; zero overhead.

Usage
-----
Context manager::

    from solaris.utils.profiling import solaris_profile

    with solaris_profile("spectral_conv"):
        out = spectral_conv(x)

Decorator::

    @solaris_profile("forward_pass")
    def forward(self, x):
        ...

CLI examples
------------
ROCm::

    HSA_OVERRIDE_GFX_VERSION=11.0.0 \\
    rocprof --roctx-trace python train.py

CUDA (Nsight Systems)::

    nsys profile --trace=cuda,nvtx python train.py
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_backend() -> str:
    """Return the active profiling backend: ``"roctx"``, ``"nvtx"``, or ``"none"``."""
    try:
        import torch

        if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip:
            return "roctx"
        if torch.cuda.is_available():
            return "nvtx"
    except ImportError:
        pass
    return "none"


_BACKEND: str = _detect_backend()


# ---------------------------------------------------------------------------
# ROCTx helpers (ctypes — no Python package needed beyond ROCm runtime)
# ---------------------------------------------------------------------------

_roctx_lib: Any = None


def _get_roctx():
    global _roctx_lib
    if _roctx_lib is not None:
        return _roctx_lib
    try:
        import ctypes

        lib = ctypes.CDLL("libroctx64.so")
        lib.roctxMarkA.argtypes = [ctypes.c_char_p]
        lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        lib.roctxRangePushA.restype = ctypes.c_int
        lib.roctxRangePop.argtypes = []
        lib.roctxRangePop.restype = ctypes.c_int
        _roctx_lib = lib
    except OSError:
        _roctx_lib = None
    return _roctx_lib


def _roctx_push(name: str) -> None:
    lib = _get_roctx()
    if lib is not None:
        lib.roctxRangePushA(name.encode())


def _roctx_pop() -> None:
    lib = _get_roctx()
    if lib is not None:
        lib.roctxRangePop()


# ---------------------------------------------------------------------------
# NVTX helpers
# ---------------------------------------------------------------------------


def _nvtx_push(name: str) -> None:
    try:
        import torch

        torch.cuda.nvtx.range_push(name)
    except Exception:  # noqa: BLE001
        pass


def _nvtx_pop() -> None:
    try:
        import torch

        torch.cuda.nvtx.range_pop()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def solaris_profile(name: str):
    """Context manager that annotates a code region in the GPU profiler trace.

    Parameters
    ----------
    name : str
        Label shown in the profiler timeline.

    Example
    -------
    ::

        with solaris_profile("spectral_conv_forward"):
            out = layer(x)
    """
    if _BACKEND == "roctx":
        _roctx_push(name)
        try:
            yield
        finally:
            _roctx_pop()
    elif _BACKEND == "nvtx":
        _nvtx_push(name)
        try:
            yield
        finally:
            _nvtx_pop()
    else:
        yield


def profile(name: str) -> Callable:
    """Decorator that wraps a function in a :func:`solaris_profile` region.

    Parameters
    ----------
    name : str
        Label shown in the profiler timeline.

    Example
    -------
    ::

        @profile("encoder_forward")
        def forward(self, x):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with solaris_profile(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def mark(name: str) -> None:
    """Emit an instant marker (point event) in the profiler trace.

    Parameters
    ----------
    name : str
        Label for the marker.
    """
    if _BACKEND == "roctx":
        lib = _get_roctx()
        if lib is not None:
            lib.roctxMarkA(name.encode())
    elif _BACKEND == "nvtx":
        try:
            import torch

            torch.cuda.nvtx.mark(name)
        except Exception:  # noqa: BLE001
            pass


def active_backend() -> str:
    """Return the active profiling backend (``"roctx"``, ``"nvtx"``, or ``"none"``)."""
    return _BACKEND
