import numpy as np
import xarray as xr


import pivpy.pivpy  # noqa: F401  (register accessor)


def _make_linear_field_dataset():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    X, Y = np.meshgrid(x, y)
    u = X.copy()
    v = Y.copy()
    w = X + Y
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "w": (("y", "x"), w),
        },
        coords={"x": ("x", x), "y": ("y", y)},
    )
    return ds


def test_azprofile_vector_radial_field_constant_ur_zero_ut():
    ds = _make_linear_field_dataset()
    angle, ur, ut = ds.piv.azprofile(x0=0.0, y0=0.0, r=1.0, na=64)
    assert angle.shape == (64,)
    assert ur.shape == (64,)
    assert ut.shape == (64,)
    np.testing.assert_allclose(ur, np.ones_like(ur), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ut, np.zeros_like(ut), rtol=0, atol=1e-12)


def test_azprofile_scalar_matches_expected_trig_combo():
    ds = _make_linear_field_dataset()
    angle, p = ds.piv.azprofile(x0=0.0, y0=0.0, r=2.0, na=32, var="w")
    expected = 2.0 * (np.cos(angle) + np.sin(angle))
    np.testing.assert_allclose(p, expected, rtol=0, atol=1e-12)
