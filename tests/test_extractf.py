import numpy as np
import xarray as xr

import pivpy.pivpy  # registers accessor


def _make_ds(x, y):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    # u(y,x) deterministic
    u = (10.0 * yy + xx).astype(float)
    v = (100.0 + 10.0 * yy - 2.0 * xx).astype(float)
    return xr.Dataset(
        {
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
        },
        coords={"x": x, "y": y},
    )


def test_extractf_phys_expands_to_grid_points():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    ds = _make_ds(x, y)

    # Rectangle strictly inside grid cells; should expand to include x=1..2, y=1..2.
    out, mesh_rect = ds.piv.extractf([1.1, 1.1, 1.9, 1.9], "phys", return_rect=True)

    assert list(out["x"].values) == [1.0, 2.0]
    assert list(out["y"].values) == [1.0, 2.0]
    assert mesh_rect == [2, 2, 3, 3]


def test_extractf_mesh_is_1_based_and_inclusive():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    ds = _make_ds(x, y)

    out = ds.piv.extractf([2, 2, 3, 3], "mesh")
    assert list(out["x"].values) == [1.0, 2.0]
    assert list(out["y"].values) == [1.0, 2.0]


def test_extractf_clamps_out_of_bounds():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    ds = _make_ds(x, y)

    out = ds.piv.extractf([-100, -100, 100, 100], "phys")
    assert out.sizes["x"] == ds.sizes["x"]
    assert out.sizes["y"] == ds.sizes["y"]


def test_extractf_handles_descending_coordinates():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([3.0, 2.0, 1.0, 0.0])  # descending
    ds = _make_ds(x, y)

    out, mesh_rect = ds.piv.extractf([1.1, 1.1, 1.9, 1.9], "phys", return_rect=True)

    assert list(out["x"].values) == [1.0, 2.0]
    # Selected indices correspond to y=2 then y=1 (preserve original order)
    assert list(out["y"].values) == [2.0, 1.0]
    assert mesh_rect == [2, 2, 3, 3]
