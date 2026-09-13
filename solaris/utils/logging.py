# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities — thin wrapper around loguru for consistent formatting."""

import sys

from loguru import logger as _logger

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[name]}</cyan> | "
    "<level>{message}</level>"
)
_handler_id: int | None = None
_handler_level: str | None = None


def get_logger(name: str | None = None, level: str = "INFO"):
    """Return a loguru logger that tags every message with *name*.

    The stderr sink is configured once (and re-configured only when *level*
    changes), so loggers obtained for different names keep their own label —
    calling ``get_logger("b")`` does not relabel messages from ``get_logger("a")``.

    Parameters
    ----------
    name : str, optional
        Logger name shown in output (default ``"solaris"``).
    level : str
        Minimum level for the stderr sink (DEBUG, INFO, WARNING, ERROR).
    """
    global _handler_id, _handler_level
    if _handler_id is None or level != _handler_level:
        if _handler_id is None:
            _logger.remove()  # drop loguru's default unformatted handler
        else:
            _logger.remove(_handler_id)
        _logger.configure(extra={"name": "solaris"})
        _handler_id = _logger.add(sys.stderr, format=_FORMAT, level=level, colorize=True)
        _handler_level = level
    return _logger.bind(name=name or "solaris")
