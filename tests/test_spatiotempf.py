import numpy as np
import xarray as xr

import pivpy.pivpy  # registers accessor


def _make_scalar_ds():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    t = np.array([0.0, 1.0, 2.0])

    xx, yy = np.meshgrid(x, y, indexing="xy")
    w = np.stack([(xx + 10.0 * yy + 100.0 * ti) for ti in t], axis=-1)
    return xr.Dataset({"w": (("y", "x", "t"), w)}, coords={"x": x, "y": y, "t": t})


def test_spatiotempf_horizontal_line_on_grid():
    ds = _make_scalar_ds()

    # Line along y=0 from x=0..2
    st = ds.piv.spatiotempf([0.0, 2.0], [0.0, 0.0], var="w", n=3, method="linear")
    assert "st" in st
    assert list(st["st"].dims) == ["t", "s"]
    assert st.sizes["t"] == 3
    assert st.sizes["s"] == 3

    # With n=3 along grid points x={0,1,2}: w = x + 100*t
    expected = np.stack([
        np.array([0.0, 1.0, 2.0]) + 100.0 * ti for ti in ds["t"].values
    ], axis=0)
    np.testing.assert_allclose(st["st"].values, expected)


def test_spatiotempf_multiple_lines_has_line_dim():
    ds = _make_scalar_ds()

    # Two lines: horizontal y=0 and vertical x=0
    X = [[0.0, 2.0], [0.0, 0.0]]
    Y = [[0.0, 0.0], [0.0, 2.0]]

    out = ds.piv.spatiotempf(X, Y, var="w", n=3, method="linear")
    assert list(out["st"].dims) == ["line", "t", "s"]
    assert out.sizes["line"] == 2

    # line 0: y=0, x=0..2 -> w = x + 100*t
    expected0 = np.stack([
        np.array([0.0, 1.0, 2.0]) + 100.0 * ti for ti in ds["t"].values
    ], axis=0)

    # line 1: x=0, y=0..2 -> w = 10*y + 100*t
    expected1 = np.stack([
        np.array([0.0, 10.0, 20.0]) + 100.0 * ti for ti in ds["t"].values
    ], axis=0)

    np.testing.assert_allclose(out["st"].sel(line=0).values, expected0)
    np.testing.assert_allclose(out["st"].sel(line=1).values, expected1)
