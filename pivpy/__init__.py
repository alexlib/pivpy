# -*- coding: utf-8 -*-
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import xarray as xr

xr.set_options(keep_attrs=True, display_expand_attrs=False)

try:
	__version__ = version("pivpy")
except PackageNotFoundError:
	__version__ = "0.1.1"

from .update import check_update, UpdateCheckResult  # noqa: E402,F401
