import numpy as np
import xarray as xr

import pivpy.pivpy  # registers accessor


def _make_scalar_ds():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    t = np.array([0.0, 1.0, 2.0, 3.0])

    xx, yy = np.meshgrid(x, y, indexing="xy")
    w = np.stack([(xx + 10.0 * yy + 100.0 * ti) for ti in t], axis=-1)
    return xr.Dataset({"w": (("y", "x", "t"), w)}, coords={"x": x, "y": y, "t": t})


def test_probef_single_point_linear_on_grid():
    ds = _make_scalar_ds()

    out = ds.piv.probef(1.0, 2.0, variables=["w"], method="linear")
    assert "w" in out
    assert list(out["w"].dims) == ["t"]

    expected = 1.0 + 10.0 * 2.0 + 100.0 * ds["t"].values
    np.testing.assert_allclose(out["w"].values, expected)


def test_probef_multiple_points_adds_probe_dim():
    ds = _make_scalar_ds()

    out = ds.piv.probef([0.0, 2.0], [0.0, 2.0], variables=["w"], method="linear")
    assert "probe" in out["w"].dims
    assert out.sizes["probe"] == 2

    # Point 0: (0,0), Point 1: (2,2)
    expected0 = 0.0 + 10.0 * 0.0 + 100.0 * ds["t"].values
    expected1 = 2.0 + 10.0 * 2.0 + 100.0 * ds["t"].values

    np.testing.assert_allclose(out["w"].sel(probe=0).values, expected0)
    np.testing.assert_allclose(out["w"].sel(probe=1).values, expected1)


def test_probeaverf_rectangle_mean():
    ds = _make_scalar_ds()

    # Select x in [0,1], y in [0,1] -> 4 points
    out = ds.piv.probeaverf([0.0, 0.0, 1.0, 1.0], variables=["w"], skipna=True)
    assert list(out["w"].dims) == ["t"]

    # Mean of w over the four grid points:
    # w = x + 10*y + 100*t
    # mean(x) over {0,1} is 0.5, mean(y) over {0,1} is 0.5
    expected = 0.5 + 10.0 * 0.5 + 100.0 * ds["t"].values
    np.testing.assert_allclose(out["w"].values, expected)
