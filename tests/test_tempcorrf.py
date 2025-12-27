import numpy as np
import xarray as xr

import pivpy.pivpy  # registers accessor


def test_tempcorrf_scalar_matches_hand_calc():
    # w(t) = [1,2,3,4]
    w = np.array([1.0, 2.0, 3.0, 4.0])
    ds = xr.Dataset({"w": ("t", w)}, coords={"t": np.arange(w.size, dtype=float)})

    out = ds.piv.tempcorrf(variables=["w"], normalize=False)
    assert list(out["f"].dims) == ["t"]

    # lag 0: mean of squares
    expected0 = (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2) / 4.0
    expected1 = (1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0) / 3.0
    expected2 = (1.0 * 3.0 + 2.0 * 4.0) / 2.0
    expected3 = (1.0 * 4.0) / 1.0

    np.testing.assert_allclose(out["f"].values, [expected0, expected1, expected2, expected3])


def test_tempcorrf_normalize_sets_c0_to_1():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    ds = xr.Dataset({"w": ("t", w)}, coords={"t": np.arange(w.size, dtype=float)})

    out = ds.piv.tempcorrf(variables=["w"], normalize=True)
    assert np.isclose(out["f"].values[0], 1.0)


def test_tempcorrf_vector_defaults_sum_components():
    u = np.array([1.0, 0.0, 1.0, 0.0])
    v = np.array([0.0, 2.0, 0.0, 2.0])
    ds = xr.Dataset({"u": ("t", u), "v": ("t", v)}, coords={"t": np.arange(u.size, dtype=float)})

    # Default excludes zeros, so only products where both are non-zero count.
    out_excl = ds.piv.tempcorrf(normalize=False)

    # Include zeros -> should compute the straightforward mean of products.
    out_incl = ds.piv.tempcorrf(opt="0", normalize=False)

    assert np.isfinite(out_incl["f"].values[0])
    # Excluding zeros should generally reduce usable samples and may differ.
    assert not np.allclose(out_excl["f"].values, out_incl["f"].values, equal_nan=True)
