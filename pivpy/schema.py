"""pivpy.schema

The canonical xarray.Dataset schema for PIV data, shared by every reader in
pivpy.io: dims, coord/var names, attrs, plus one constructor (build_dataset)
and one validator (validate/is_valid) so readers stop hand-rolling
xr.Dataset(...) with drifting conventions.

Schema:
- dims: ('y', 'x', 't')
- coords: x (1D), y (1D), t (1D, frame index or time)
- data_vars: u, v (velocity components), chc (validity flag, 1=valid)
- attrs: delta_t, files, pivpy_schema_version
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

REQUIRED_DIMS = ("y", "x", "t")
REQUIRED_VARS = ("u", "v", "chc")

POS_UNITS: str = "pix"
TIME_UNITS: str = "frame"
VEL_UNITS: str = POS_UNITS
DELTA_T: float = 0.0

SCHEMA_VERSION = "1"

# Domain-specific standard names (not full CF compliance -- PIV isn't
# geospatial, so there's no benefit chasing CF's grid/axis/bounds vocabulary).
# Kept as plain constants so cf_xarray-style discovery is possible later
# without committing to it now.
STANDARD_NAMES = {
    "u": "u_velocity",
    "v": "v_velocity",
    "chc": "validity_flag",
}


def set_default_attrs(dataset: xr.Dataset) -> xr.Dataset:
    """Apply default units/standard_name/global attrs (sets missing only)."""
    ds = dataset

    if "x" in ds:
        ds["x"].attrs.setdefault("units", POS_UNITS)
    if "y" in ds:
        ds["y"].attrs.setdefault("units", POS_UNITS)
    if "t" in ds:
        ds["t"].attrs.setdefault("units", TIME_UNITS)
    for var in ("u", "v"):
        if var in ds:
            ds[var].attrs.setdefault("units", VEL_UNITS)
            ds[var].attrs.setdefault("standard_name", STANDARD_NAMES[var])
    if "chc" in ds:
        ds["chc"].attrs.setdefault("standard_name", STANDARD_NAMES["chc"])

    ds.attrs.setdefault("delta_t", float(DELTA_T))
    ds.attrs.setdefault("files", [])
    ds.attrs.setdefault("pivpy_schema_version", SCHEMA_VERSION)
    return ds


def _coords_from_mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected 2D mesh arrays for x and y")
    return np.asarray(x[0, :]), np.asarray(y[:, 0])


def build_dataset(
    x: ArrayLike,
    y: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    chc: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    t: Optional[ArrayLike] = None,
    frame: int = 0,
    delta_t: float = 0.0,
    files: Optional[list[str]] = None,
    **extra_vars: ArrayLike,
) -> xr.Dataset:
    """Canonical constructor for a pivpy Dataset.

    x, y are 1D coordinate vectors. u, v, chc, and any extra_vars may be
    either a single (rows, cols) frame or an already-stacked (rows, cols,
    nframes) array. mask (if given, and chc isn't) seeds chc -- there's no
    separate 'mask' variable in the canonical schema, callers that have a
    validity mask fold it into chc themselves before/via this call.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.ndim not in (2, 3) or v.ndim not in (2, 3):
        raise ValueError("u and v must be 2D (single frame) or 3D (y,x,t) arrays")
    if u.ndim == 2:
        u = u[:, :, None]
    if v.ndim == 2:
        v = v[:, :, None]

    rows, cols, nframes = u.shape

    if chc is None:
        chc = mask if mask is not None else np.ones((rows, cols, nframes))
    chc = np.asarray(chc, dtype=float)
    if chc.ndim == 2:
        chc = chc[:, :, None]

    if t is None:
        t = np.arange(frame, frame + nframes, dtype=float)
    else:
        t = np.atleast_1d(np.asarray(t, dtype=float))

    data_vars: dict[str, tuple] = {
        "u": (("y", "x", "t"), u),
        "v": (("y", "x", "t"), v),
        "chc": (("y", "x", "t"), chc),
    }
    for name, arr in extra_vars.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        data_vars[name] = (("y", "x", "t"), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"x": ("x", x), "y": ("y", y), "t": ("t", t)},
        attrs={"delta_t": float(delta_t), "files": list(files) if files else []},
    )
    return set_default_attrs(ds)


def validate(ds: xr.Dataset) -> None:
    """Raise ValueError with a clear message if ds doesn't match the schema."""
    missing_dims = [d for d in REQUIRED_DIMS if d not in ds.dims]
    if missing_dims:
        raise ValueError(f"pivpy dataset missing required dims: {missing_dims}")
    missing_vars = [v for v in REQUIRED_VARS if v not in ds.data_vars]
    if missing_vars:
        raise ValueError(f"pivpy dataset missing required variables: {missing_vars}")
    for var in REQUIRED_VARS:
        if tuple(ds[var].dims) != REQUIRED_DIMS:
            raise ValueError(f"{var} has dims {ds[var].dims}, expected {REQUIRED_DIMS}")


def is_valid(ds: xr.Dataset) -> bool:
    try:
        validate(ds)
    except ValueError:
        return False
    return True
