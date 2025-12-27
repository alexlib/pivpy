import numpy as np
import xarray as xr


import pivpy.pivpy  # noqa: F401 (register accessor)


def test_bwfilterf_filtsize_zero_is_identity():
    y = np.arange(6)
    x = np.arange(8)
    t = np.array([0.0])
    u = np.random.default_rng(0).standard_normal((6, 8, 1))
    v = np.random.default_rng(1).standard_normal((6, 8, 1))
    ds = xr.Dataset(
        data_vars={"u": (("y", "x", "t"), u), "v": (("y", "x", "t"), v)},
        coords={"y": ("y", y), "x": ("x", x), "t": ("t", t)},
    )

    out = ds.piv.bwfilterf(filtsize=0.0, order=8)
    np.testing.assert_allclose(out["u"].values, u, rtol=0, atol=0)
    np.testing.assert_allclose(out["v"].values, v, rtol=0, atol=0)


def test_bwfilterf_highpass_removes_constant_field():
    y = np.arange(8)
    x = np.arange(8)
    t = np.array([0.0])
    u = np.ones((8, 8, 1)) * 3.0
    v = np.ones((8, 8, 1)) * -2.0
    ds = xr.Dataset(
        data_vars={"u": (("y", "x", "t"), u), "v": (("y", "x", "t"), v)},
        coords={"y": ("y", y), "x": ("x", x), "t": ("t", t)},
    )
    out = ds.piv.bwfilterf(filtsize=3.0, order=8.0, mode="high")
    # High-pass should zero out the DC component.
    assert float(np.max(np.abs(out["u"].values))) < 1e-10
    assert float(np.max(np.abs(out["v"].values))) < 1e-10


def test_bwfilterf_trims_odd_sizes_to_even():
    y = np.arange(7)
    x = np.arange(9)
    t = np.array([0.0])
    u = np.random.default_rng(2).standard_normal((7, 9, 1))
    v = np.random.default_rng(3).standard_normal((7, 9, 1))
    ds = xr.Dataset(
        data_vars={"u": (("y", "x", "t"), u), "v": (("y", "x", "t"), v)},
        coords={"y": ("y", y), "x": ("x", x), "t": ("t", t)},
    )
    out = ds.piv.bwfilterf(filtsize=2.0, order=4.0)
    assert out.sizes["y"] == 6
    assert out.sizes["x"] == 8


def test_bwfilterf_pm_high_and_trunc_options():
    y = np.arange(10)
    x = np.arange(10)
    t = np.array([0.0])
    u = np.ones((10, 10, 1)) * 1.0
    v = np.ones((10, 10, 1)) * 1.0
    ds = xr.Dataset(
        data_vars={"u": (("y", "x", "t"), u), "v": (("y", "x", "t"), v)},
        coords={"y": ("y", y), "x": ("x", x), "t": ("t", t)},
    )

    out = ds.piv.bwfilterf_pm(3.0, 8.0, "high", "trunc")
    # trunc removes floor(3)=3 cells from each border => 10-6 = 4
    assert out.sizes["y"] == 4
    assert out.sizes["x"] == 4
    # high-pass removes DC
    assert float(np.max(np.abs(out["u"].values))) < 1e-10
