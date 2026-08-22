# -*- coding: utf-8 -*-
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import xarray as xr

xr.set_options(keep_attrs=True, display_expand_attrs=False)

try:
	__version__ = version("pivpy")
except PackageNotFoundError:
	__version__ = "0.2.0"

from .update import check_update, UpdateCheckResult  # noqa: E402,F401

# registers the .piv xarray Dataset accessor (@xr.register_dataset_accessor) --
# importing pivpy.io / pivpy.graphics alone does not trigger this
from . import pivpy as _pivpy_accessor  # noqa: E402,F401
from . import synthetic  # noqa: E402,F401

