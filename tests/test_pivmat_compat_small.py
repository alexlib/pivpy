import numpy as np
import xarray as xr


from pivpy.compute_funcs import interpolat_zeros_2d, meannz
from pivpy.pivmat_compat import expandstr


def test_meannz_numpy_dim0_and_dim1():
    x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    # dim=0 means down the rows -> per-column
    got0 = meannz(x, dim=0)
    np.testing.assert_allclose(got0, np.array([3.0, 2.5, 3.5]), rtol=0, atol=1e-12)

    # dim=1 means across columns -> per-row
    got1 = meannz(x, dim=1)
    np.testing.assert_allclose(got1, np.array([1.5, 4.0]), rtol=0, atol=1e-12)


def test_meannz_xarray_dimname():
    da = xr.DataArray(
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1, 2]},
    )
    out = meannz(da, dim="y")
    assert out.dims == ("x",)
    np.testing.assert_allclose(out.values, np.array([3.0, 2.5, 3.5]), rtol=0, atol=1e-12)


def test_interpolat_zeros_2d_single_pass_center():
    m = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    out = interpolat_zeros_2d(m, fill=False)
    assert out[1, 1] == 1.0


def test_interpolat_zeros_2d_fill_completes_simple_case():
    m = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    out = interpolat_zeros_2d(m, fill=True, max_iter=10)
    assert not np.any(out == 0)
    np.testing.assert_allclose(out, np.ones_like(out), rtol=0, atol=1e-12)


def test_expandstr_basic_and_recursive():
    out = expandstr("DSC[2:2:8,4].JPG")
    assert out == ["DSC0002.JPG", "DSC0004.JPG", "DSC0006.JPG", "DSC0008.JPG"]

    out2 = expandstr("B[1:2,2]_[1 2,1]")
    assert out2 == ["B01_1", "B01_2", "B02_1", "B02_2"]
