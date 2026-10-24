# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading that does not execute code from the file by default.

``torch.load`` with ``weights_only=False`` unpickles arbitrary Python objects,
so opening a checkpoint someone else made can run any code they put in it.
Solaris checkpoints only contain tensors, numbers, strings, lists and dicts,
which PyTorch's restricted ``weights_only=True`` loader handles, so that is
the default.  Pass ``trusted=True`` only for files you created yourself.
"""

import pickle
from pathlib import Path
from typing import Any

import torch


def load_file(
    path: str | Path,
    map_location: str | torch.device | None = None,
    trusted: bool = False,
) -> Any:
    """``torch.load`` with ``weights_only=not trusted`` and a helpful error."""
    try:
        return torch.load(path, map_location=map_location, weights_only=not trusted)
    except pickle.UnpicklingError as exc:
        if trusted:
            raise
        raise pickle.UnpicklingError(
            f"{path} contains objects other than tensors and plain Python data, so it was "
            "not loaded (unpickling them could run arbitrary code). If you created this "
            "file yourself, load it with trusted=True."
            f"\n\nOriginal error: {exc}"
        ) from exc
