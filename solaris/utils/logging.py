# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities — thin wrapper around loguru for consistent formatting."""

import sys

from loguru import logger as _logger


def get_logger(name: str | None = None, level: str = "INFO"):
    """Return a configured loguru logger.

    Parameters
    ----------
    name : str, optional
        Logger name prefix shown in output.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    _logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{name or 'solaris'}</cyan> | "
        "<level>{message}</level>"
    )
    _logger.add(sys.stderr, format=fmt, level=level, colorize=True)
    return _logger
