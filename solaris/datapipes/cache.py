# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Config-keyed ``.npz`` cache for generated datasets.

Simulation datasets are expensive to generate, so scripts cache them.  A
cache keyed only by filename silently returns stale data when the generating
parameters change (resolution, sample count, physics constants …).
:func:`cached_arrays` stores the generating config inside the file and refuses
to reuse a cache built with a different config.
"""

import json
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

_CONFIG_KEY = "__solaris_cache_config__"


class CacheMismatchError(RuntimeError):
    """The cache on disk was generated with a different config."""


def _canonical(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, default=str)


def cached_arrays(
    path: str | Path,
    config: Mapping[str, Any],
    generate: Callable[[], Mapping[str, np.ndarray]],
    overwrite_on_mismatch: bool = False,
) -> dict[str, np.ndarray]:
    """Load arrays from *path*, or generate and save them.

    Parameters
    ----------
    path :
        ``.npz`` cache file.
    config :
        JSON-serialisable parameters that determine the generated data.
    generate :
        Zero-argument callable returning a mapping of name → array.
    overwrite_on_mismatch :
        If the cache exists with a different config, regenerate and overwrite
        it instead of raising :class:`CacheMismatchError`.

    Returns
    -------
    dict[str, np.ndarray]
    """
    path = Path(path)
    wanted = _canonical(config)

    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            data = {k: d[k] for k in d.files if k != _CONFIG_KEY}
            stored = str(d[_CONFIG_KEY]) if _CONFIG_KEY in d.files else None
        if stored is None:
            warnings.warn(
                f"Cache {path} has no recorded config; assuming it matches {wanted}. "
                "Delete it to regenerate.",
                UserWarning,
                stacklevel=2,
            )
            return data
        if stored == wanted:
            return data
        if not overwrite_on_mismatch:
            raise CacheMismatchError(
                f"Cache {path} was generated with config {stored}, but {wanted} was "
                "requested. Delete the file, pass a different cache path, or "
                "regenerate explicitly."
            )

    data = {k: np.asarray(v) for k, v in generate().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data, **{_CONFIG_KEY: np.array(wanted)})
    return data
