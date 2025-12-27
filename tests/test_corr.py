import numpy as np
import pytest
import xarray as xr


from pivpy.compute_funcs import corrm, corrx


def test_corrx_autocorr_matches_expected_full_and_half():
    x = np.array([1.0, 2.0, 3.0])

    expected_full = np.array([3.0, 4.0, 14.0 / 3.0, 4.0, 3.0])
    got_full = corrx(x)
    np.testing.assert_allclose(got_full, expected_full, rtol=0, atol=1e-12)

    expected_half = np.array([14.0 / 3.0, 4.0, 3.0])
    got_half = corrx(x, half=True)
    np.testing.assert_allclose(got_half, expected_half, rtol=0, atol=1e-12)


def test_corrx_is_symmetric_for_autocorr():
    x = np.array([1.0, 0.0, 2.0, 0.0])
    c = corrx(x)
    np.testing.assert_allclose(c, c[::-1], rtol=0, atol=1e-12)


def test_corrx_nan_as_zero_changes_result():
    x = np.array([1.0, np.nan, 3.0])
    c1 = corrx(x, nan_as_zero=True)
    c2 = corrx(x, nan_as_zero=False)
    # With NaN preserved, sums propagate NaN at some lags.
    assert np.any(np.isnan(c2))
    assert not np.any(np.isnan(c1))


def test_corrm_numpy_dim2_rows_match_expected():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = corrm(a, dim=2)
    expected = np.array(
        [
            [3.0, 4.0, 14.0 / 3.0, 4.0, 3.0],
            [24.0, 25.0, 77.0 / 3.0, 25.0, 24.0],
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_corrm_numpy_dim1_columns_match_expected_and_half():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got_full = corrm(a, dim=1)
    expected_full = np.array(
        [
            [4.0, 10.0, 18.0],
            [8.5, 14.5, 22.5],
            [4.0, 10.0, 18.0],
        ]
    )
    np.testing.assert_allclose(got_full, expected_full, rtol=0, atol=1e-12)

    got_half = corrm(a, dim=1, half=True)
    expected_half = np.array(
        [
            [8.5, 14.5, 22.5],
            [4.0, 10.0, 18.0],
        ]
    )
    np.testing.assert_allclose(got_half, expected_half, rtol=0, atol=1e-12)


def test_corrm_xarray_dimname_preserves_other_dims_and_lag_coord():
    a = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        dims=("y", "x"),
        coords={"y": [10, 20], "x": [0.1, 0.2, 0.3]},
        name="u",
    )
    out = corrm(a, dim="x", lag_dim="lag")
    assert out.dims == ("y", "lag")
    np.testing.assert_array_equal(out["y"].values, np.array([10, 20]))
    np.testing.assert_array_equal(out["lag"].values, np.array([-2, -1, 0, 1, 2]))

    expected = np.array(
        [
            [3.0, 4.0, 14.0 / 3.0, 4.0, 3.0],
            [24.0, 25.0, 77.0 / 3.0, 25.0, 24.0],
        ]
    )
    np.testing.assert_allclose(out.values, expected, rtol=0, atol=1e-12)


def test_corrm_xarray_half_selects_nonnegative_lags():
    a = xr.DataArray(
        np.array([[1.0, 2.0, 3.0]]),
        dims=("y", "x"),
        coords={"y": [0], "x": [0, 1, 2]},
    )
    out = corrm(a, dim="x", half=True)
    np.testing.assert_array_equal(out["lag"].values, np.array([0, 1, 2]))
    np.testing.assert_allclose(out.values, np.array([[14.0 / 3.0, 4.0, 3.0]]), rtol=0, atol=1e-12)


def test_corrm_numpy_rejects_non_2d():
    with pytest.raises(ValueError):
        corrm(np.array([1.0, 2.0, 3.0]), dim=1)
