import numpy as np
import pytest
import xarray as xr

import pivpy.pivpy  # noqa: F401  (registers the .piv accessor)


def _make_linear_scalar_ds(*, y_desc: bool = False, with_t: bool = False) -> xr.Dataset:
    x = np.linspace(0.0, 4.0, 5)
    y = np.linspace(0.0, 3.0, 4)
    if y_desc:
        y = y[::-1]

    # Common pivpy convention is ('y','x',...) ordering.
    Y, X = np.meshgrid(y, x, indexing="ij")

    if with_t:
        t = np.array([0.0, 1.0])
        scale = (t + 1.0)[None, None, :]
        w = (X[:, :, None] + 2.0 * Y[:, :, None]) * scale
        ds = xr.Dataset(
            {"w": (("y", "x", "t"), w)},
            coords={"x": x, "y": y, "t": t},
        )
    else:
        w = X + 2.0 * Y
        ds = xr.Dataset(
            {"w": (("y", "x"), w)},
            coords={"x": x, "y": y},
        )

    ds["w"].attrs["units"] = "m/s"
    ds["x"].attrs["units"] = "m"
    ds["y"].attrs["units"] = "m"
    return ds


def test_gradientf_linear_field_components_and_units():
    ds = _make_linear_scalar_ds()
    grad = ds.piv.gradientf(variable="w")

    assert "w" not in grad
    assert "u" in grad and "v" in grad

    assert np.allclose(grad["u"].values, 1.0)
    assert np.allclose(grad["v"].values, 2.0)

    assert grad["u"].attrs.get("units") == "m/s/m"
    assert grad["v"].attrs.get("units") == "m/s/m"


def test_gradientf_descending_y_coordinate():
    ds = _make_linear_scalar_ds(y_desc=True)
    grad = ds.piv.gradientf(variable="w")

    assert np.allclose(grad["u"].values, 1.0)
    assert np.allclose(grad["v"].values, 2.0)


def test_gradientf_preserves_time_dimension():
    ds = _make_linear_scalar_ds(with_t=True)
    grad = ds.piv.gradientf(variable="w")

    assert grad["u"].dims == ("y", "x", "t")
    assert grad["v"].dims == ("y", "x", "t")

    # w = (X + 2Y) * (t+1)
    # dw/dx = 1 * (t+1), dw/dy = 2 * (t+1)
    assert np.allclose(grad["u"].sel(t=0.0).values, 1.0)
    assert np.allclose(grad["v"].sel(t=0.0).values, 2.0)
    assert np.allclose(grad["u"].sel(t=1.0).values, 2.0)
    assert np.allclose(grad["v"].sel(t=1.0).values, 4.0)


def test_gradientf_missing_variable_raises():
    ds = _make_linear_scalar_ds()
    with pytest.raises(KeyError):
        ds.piv.gradientf(variable="nope")
