# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Solaris warning and exception hirarchy"""

class SolarisWarning(UserWarning):
    """Base class for all warnings raised by solaris"""


class ExperimentalFeatureWarning(SolarisWarning):
    """Raised when an experimental feature is used"""


class FutureFeatureWarning(SolarisWarning):
    """Behavior will change in a future release; Update your code now to avoid breakage in the future"""


class LegacyFeatureWarning(SolarisWarning):
    """Feature is deprecated and will be removed in a future release; Update your code now to avoid breakage in the future"""

    