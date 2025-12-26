"""Backward-compatible alias for the `pivpy.interfacing` module.

The original interop helpers lived in `pivpy.inter`. The canonical location is now
`pivpy.interfacing`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`pivpy.inter` has been renamed to `pivpy.interfacing`; please update imports.",
    DeprecationWarning,
    stacklevel=2,
)

from .interfacing import *  # noqa: F403
