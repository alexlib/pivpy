from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import xarray as xr

xr.set_options(keep_attrs=True, display_expand_attrs=False)

try:
    __version__ = version("pivpy")
except PackageNotFoundError:
    __version__ = "0.2.1"

# registers the .piv xarray Dataset accessor (@xr.register_dataset_accessor) --
# importing pivpy.io / pivpy.graphics alone does not trigger this
from . import pivpy as _pivpy_accessor  # noqa: F401
from . import synthetic  # noqa: F401
from .update import UpdateCheckResult, check_update  # noqa: F401
