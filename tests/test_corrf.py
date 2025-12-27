import numpy as np
import pytest
import xarray as xr

import pivpy.pivpy  # registers accessor
from pivpy.compute_funcs import _corrf_scales


def _expected_corr_along_x(values: np.ndarray) -> np.ndarray:
    """Brute-force expected corrf along x for nonzero fields.

    values shape: (y, x, t)
    returns shape: (x,)
    """
    y, x, t = values.shape
    out = np.zeros(x, dtype=float)
    for lag in range(x):
        prod = values[:, : (x - lag), :] * values[:, lag:, :]
        out[lag] = float(prod.mean())
    return out


def _expected_corr_along_y(values: np.ndarray) -> np.ndarray:
    """Brute-force expected corrf along y for nonzero fields.

    values shape: (y, x, t)
    returns shape: (y,)
    """
    y, x, t = values.shape
    out = np.zeros(y, dtype=float)
    for lag in range(y):
        prod = values[: (y - lag), :, :] * values[lag:, :, :]
        out[lag] = float(prod.mean())
    return out


def test_corrf_matches_bruteforce_along_x_and_y():
    y = np.array([0.0, 1.0])
    x = np.array([0.0, 0.5, 1.0, 1.5])
    t = np.array([0, 1, 2])

    # Nonzero deterministic field.
    yy, xx, tt = np.meshgrid(y, x, t, indexing="ij")
    w = 1.0 + 2.0 * xx + 3.0 * yy + 5.0 * tt

    ds = xr.Dataset({"w": (("y", "x", "t"), w)}, coords={"y": y, "x": x, "t": t})

    corx = ds.piv.corrf(variable="w", dim="x", nowarning=True)
    expected_x = _expected_corr_along_x(w)
    np.testing.assert_allclose(corx["f"].values, expected_x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(corx["r"].values, np.arange(len(x)) * 0.5)

    cory = ds.piv.corrf(variable="w", dim="y", nowarning=True)
    expected_y = _expected_corr_along_y(w)
    np.testing.assert_allclose(cory["f"].values, expected_y, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cory["r"].values, np.arange(len(y)) * 1.0)


def test_corrf_normalize_sets_f0_to_one():
    y = np.array([0.0, 1.0])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    t = np.array([0, 1])

    yy, xx, tt = np.meshgrid(y, x, t, indexing="ij")
    w = 2.0 + xx + 10.0 * yy + 100.0 * tt

    ds = xr.Dataset({"w": (("y", "x", "t"), w)}, coords={"y": y, "x": x, "t": t})

    cor = ds.piv.corrf(variable="w", dim="x", normalize=True, nowarning=True)
    assert float(cor["f"].values[0]) == pytest.approx(1.0)

    expected = _expected_corr_along_x(w)
    expected = expected / expected[0]
    np.testing.assert_allclose(cor["f"].values, expected, rtol=0, atol=1e-12)


def test_corrf_emits_warnings_when_thresholds_undefined():
    y = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    t = np.array([0, 1])

    w = np.ones((len(y), len(x), len(t)), dtype=float)
    ds = xr.Dataset({"w": (("y", "x", "t"), w)}, coords={"y": y, "x": x, "t": t})

    with pytest.warns(UserWarning) as rec:
        cor = ds.piv.corrf(variable="w", dim="x")

    # Constant positive correlation never crosses 0.5, 0.2, 0.1, or 0.
    assert len(rec) == 4
    assert np.isfinite(float(cor["isinf"].values))
    assert np.isnan(float(cor["r5"].values))
    assert np.isnan(float(cor["r2"].values))
    assert np.isnan(float(cor["r1"].values))
    assert np.isnan(float(cor["r0"].values))

    # Suppressed warnings.
    cor2 = ds.piv.corrf(variable="w", dim="x", nowarning=True)
    assert np.isnan(float(cor2["r5"].values))


def test__corrf_scales_interpolates_crossings_and_integrals():
    r = np.array([0.0, 1.0, 2.0, 3.0])
    f = np.array([10.0, 8.0, 4.0, -1.0])

    s = _corrf_scales(r, f, nowarning=True)

    # isinf = sum(f/f0) * dr
    assert s["isinf"] == pytest.approx(((10 + 8 + 4 - 1) / 10.0) * 1.0)

    # 0.5 crossing between 8 (r=1) and 4 (r=2)
    assert s["r5"] == pytest.approx(1.75)
    assert s["is5"] == pytest.approx(((10 + 8 + 4) / 10.0) * 1.0)

    # 0.2 crossing between 4 (r=2) and -1 (r=3)
    assert s["r2"] == pytest.approx(2.4)

    # zero crossing between 4 (r=2) and -1 (r=3)
    assert s["r0"] == pytest.approx(2.8)
