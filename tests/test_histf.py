import numpy as np
import pytest
import xarray as xr

import pivpy.pivpy  # noqa: F401  (registers the .piv accessor)


def _scalar_ds_from_values(values: np.ndarray) -> xr.Dataset:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be 2D")

    y = np.arange(values.shape[0], dtype=float)
    x = np.arange(values.shape[1], dtype=float)
    return xr.Dataset({"w": (("y", "x"), values)}, coords={"x": x, "y": y})


def _vector_ds_from_values(u: np.ndarray, v: np.ndarray) -> xr.Dataset:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    y = np.arange(u.shape[0], dtype=float)
    x = np.arange(u.shape[1], dtype=float)
    return xr.Dataset({"u": (("y", "x"), u), "v": (("y", "x"), v)}, coords={"x": x, "y": y})


def test_histf_scalar_default_bins_symmetric_when_mean_lt_std():
    # Non-zero values: [-1, 1] => mean=0, std=1 => mean < std => symmetric bins.
    ds = _scalar_ds_from_values(np.array([[-1.0, 0.0], [0.0, 1.0]]))
    out = ds.piv.histf(variable="w")

    assert "h" in out
    assert out["bin"].size == 200

    # bins should be roughly [-20, 20]
    assert np.isclose(out["bin"].values[0], -20.0)
    assert np.isclose(out["bin"].values[-1], 20.0)

    # zeros excluded by default => only 2 samples
    assert int(out["h"].sum()) == 2


def test_histf_scalar_default_bins_centered_when_mean_ge_std():
    # Non-zero values: [10, 12] => mean=11, std=1 => mean >= std => centered at mean.
    ds = _scalar_ds_from_values(np.array([[10.0, 0.0], [0.0, 12.0]]))
    out = ds.piv.histf(variable="w")

    assert out["bin"].size == 200
    assert np.isclose(out["bin"].values[0], -9.0)
    assert np.isclose(out["bin"].values[-1], 31.0)
    assert int(out["h"].sum()) == 2


def test_histf_scalar_include_zeros_with_opt_0():
    ds = _scalar_ds_from_values(np.array([[0.0, 0.0], [0.0, 1.0]]))

    out_excl = ds.piv.histf(variable="w")
    out_incl = ds.piv.histf(variable="w", opt="0")

    assert int(out_excl["h"].sum()) == 1
    assert int(out_incl["h"].sum()) == 4


def test_histf_scalar_custom_bins_used_as_centers():
    ds = _scalar_ds_from_values(np.array([[0.0, 2.0], [0.0, 2.0]]))
    centers = np.array([0.0, 1.0, 2.0], dtype=float)
    out = ds.piv.histf(variable="w", bin=centers, opt="0")

    assert np.allclose(out["bin"].values, centers)
    # values are {0,2,0,2}
    assert int(out["h"].sum()) == 4


def test_histf_vector_mode_returns_hx_hy_and_common_bins():
    ds = _vector_ds_from_values(u=np.array([[0.0, 1.0], [0.0, 1.0]]), v=np.array([[0.0, 2.0], [0.0, 2.0]]))
    out = ds.piv.histf()

    assert "hx" in out and "hy" in out
    assert out["bin"].size == 200
    # zeros excluded by default => 2 samples per component
    assert int(out["hx"].sum()) == 2
    assert int(out["hy"].sum()) == 2


def test_histf_missing_scalar_variable_raises():
    ds = _vector_ds_from_values(u=np.ones((2, 2)), v=np.ones((2, 2)))
    with pytest.raises(KeyError):
        ds.piv.histf(variable="w")
