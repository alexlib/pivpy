"""Utility helpers for pivpy graphics.

This module is intentionally tiny: it provides helpers used by plotting functions
without pulling in heavy dependencies or creating circular imports.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def dataset_to_array(data: xr.Dataset):
    """Convert a standard PIV Dataset into 2D numpy arrays (x, y, u, v).

    Expects:
    - coords: x (1D), y (1D)
    - data vars: u, v with dims (y, x) or (y, x, t)

    Returns
    -------
    x2d, y2d, u2d, v2d : np.ndarray
    """

    ds = data
    if "t" in ds.dims:
        # pick the first frame by default
        ds = ds.isel(t=0)

    x = ds["x"].values
    y = ds["y"].values
    x2d, y2d = np.meshgrid(x, y)

    u = ds["u"].values
    v = ds["v"].values

    # handle accidental lingering singleton t dimension
    if u.ndim == 3:
        u = u[:, :, 0]
    if v.ndim == 3:
        v = v[:, :, 0]

    return x2d, y2d, u, v
