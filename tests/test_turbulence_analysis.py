"""Unit tests for turbulence analysis features adopted and adapted from xrturb."""
import numpy as np
import pytest
import xarray as xr

from pivpy.schema import build_dataset
from pivpy.synthetic import multivortex
import pivpy.pivpy  # noqa: F401


@pytest.fixture
def sample_turb_ds():
    """Generates a reproducible 16-frame 2D synthetic dataset with coordinates (y, x, t)."""
    return multivortex(n_frames=16, n=32, n_vortices=4, seed=42)


@pytest.fixture
def sample_3d_turb_ds(sample_turb_ds):
    """Adds a 3D out-of-plane velocity component 'w' to the dataset."""
    ds = sample_turb_ds.copy(deep=True)
    # Synthetic out-of-plane w
    w = 0.5 * ds["u"] - 0.2 * ds["v"] + 0.1 * np.sin(ds["x"])
    ds["w"] = w
    ds["w"].attrs["units"] = "m/s"
    ds["w"].attrs["standard_name"] = "w_velocity"
    return ds


def test_fluct_and_products(sample_turb_ds):
    ds = sample_turb_ds
    u_prime = ds.piv.fluct("u")
    assert isinstance(u_prime, xr.DataArray)
    assert u_prime.name == "u'"
    np.testing.assert_allclose(u_prime.mean(dim="t").values, 0.0, atol=1e-12)

    # Product calculation
    upvp = ds.piv.product("u'v'")
    assert upvp.name == "u'v'"
    expected = u_prime * ds.piv.fluct("v")
    np.testing.assert_allclose(upvp.values, expected.values)

    # Raw variable product
    uv = ds.piv.product("uv")
    expected_uv = ds["u"] * ds["v"]
    np.testing.assert_allclose(uv.values, expected_uv.values)

    # Weighted fluctuations
    weights = xr.DataArray(np.linspace(1.0, 2.0, ds.sizes["t"]), dims=("t",))
    u_prime_w = ds.piv.fluct("u", weights=weights)
    weighted_mean = (u_prime_w * weights).sum(dim="t") / weights.sum(dim="t")
    np.testing.assert_allclose(weighted_mean.values, 0.0, atol=1e-12)


def test_reynolds_stresses_2d_and_3d(sample_turb_ds, sample_3d_turb_ds):
    # 2D dataset
    rss_2d = sample_turb_ds.piv.reynolds_stresses()
    assert "u'u'" in rss_2d
    assert "v'v'" in rss_2d
    assert "u'v'" in rss_2d
    assert "w'w'" not in rss_2d
    assert np.all(rss_2d["u'u'"].values >= 0.0)
    assert np.all(rss_2d["v'v'"].values >= 0.0)

    # 3D dataset
    rss_3d = sample_3d_turb_ds.piv.reynolds_stresses()
    assert "u'u'" in rss_3d
    assert "v'v'" in rss_3d
    assert "w'w'" in rss_3d
    assert "u'v'" in rss_3d
    assert "u'w'" in rss_3d
    assert "v'w'" in rss_3d


def test_tke_computation(sample_turb_ds, sample_3d_turb_ds):
    ds = sample_turb_ds
    # Instantaneous TKE across time frames
    ds_tke = ds.piv.tke(name="tke")
    assert "tke" in ds_tke
    u_prime = ds["u"] - ds["u"].mean(dim="t")
    v_prime = ds["v"] - ds["v"].mean(dim="t")
    expected_tke = 0.5 * (u_prime**2 + v_prime**2)
    np.testing.assert_allclose(ds_tke["tke"].values, expected_tke.values, atol=1e-12)

    # Time-averaged TKE
    ds_tke_avg = ds.piv.tke(name="tke_avg", time_average=True)
    np.testing.assert_allclose(
        ds_tke_avg["tke_avg"].values, expected_tke.mean(dim="t").values, atol=1e-12
    )

    # 3D TKE includes w
    ds_3d_tke = sample_3d_turb_ds.piv.tke(name="tke_3d", time_average=True)
    w_prime = sample_3d_turb_ds["w"] - sample_3d_turb_ds["w"].mean(dim="t")
    expected_3d = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2).mean(dim="t")
    np.testing.assert_allclose(ds_3d_tke["tke_3d"].values, expected_3d.values, atol=1e-12)


def test_moments_and_all_products(sample_turb_ds):
    ds = sample_turb_ds
    # Central moments up to order 3 (includes skewness terms)
    m = ds.piv.moments(order=3, standardized=False)
    assert "u'u'" in m
    assert "v'v'" in m
    assert "u'u'u'" in m
    assert "v'v'v'" in m
    assert "u'u'v'" in m

    # Standardized moments
    sm = ds.piv.moments(order=3, standardized=True)
    assert "Mu'u'" in sm
    assert "Mu'u'u'" in sm
    # Standardized variance is identically 1.0 (where variance > 0)
    valid = np.isfinite(sm["Mu'u'"].values)
    np.testing.assert_allclose(sm["Mu'u'"].values[valid], 1.0, atol=1e-6)

    # All products added to dataset
    prods_ds = ds.piv.all_products(order=2)
    assert "u'u'" in prods_ds
    assert "u'v'" in prods_ds
    assert "v'v'" in prods_ds


def test_two_point_correlation_and_length_scales(sample_turb_ds):
    ds = sample_turb_ds
    x_mid = float(ds["x"].mean().values)
    y_mid = float(ds["y"].mean().values)

    # Two-point correlation
    corr = ds.piv.two_point_correlation(var_name="u", x_ref=x_mid, y_ref=y_mid)
    assert isinstance(corr, xr.DataArray)
    assert "lag_x" in corr.coords
    assert "lag_y" in corr.coords
    # Self-correlation at lag 0 (nearest grid point) should be 1.0
    val_at_ref = corr.sel(lag_x=0.0, lag_y=0.0, method="nearest").values
    np.testing.assert_allclose(val_at_ref, 1.0, atol=1e-6)

    # Length scale computation via fit on field
    L_fit = ds.piv.length_scale(variable="u", dim="x", method="fit", fit_model="single", x_ref=x_mid, y_ref=y_mid)
    assert isinstance(L_fit, xr.DataArray)
    val_fit = float(L_fit.sel(lag_y=0.0, method="nearest").values)
    assert np.isfinite(val_fit) and val_fit > 0.0


def test_compute_length_scale_methods():
    from pivpy.compute_funcs import compute_length_scale

    # Canonical single exponential: R(r) = exp(-r / 5.0) -> L = 5.0
    r = np.linspace(0, 25, 51)
    R_single = np.exp(-r / 5.0)
    da_single = xr.DataArray(R_single, dims=("lag_x",), coords={"lag_x": r})

    L_1e = float(compute_length_scale(da_single, dim="lag_x", method="1e").values)
    np.testing.assert_allclose(L_1e, 5.0, atol=1e-3)

    L_int = float(compute_length_scale(da_single, dim="lag_x", method="integral").values)
    np.testing.assert_allclose(L_int, 5.0, atol=0.1)

    L_fit = float(compute_length_scale(da_single, dim="lag_x", method="fit").values)
    np.testing.assert_allclose(L_fit, 5.0, atol=1e-3)

    # Bi-exponential: R(r) = 0.6 * exp(-r / 3.0) + 0.4 * exp(-r / 12.0)
    r_bi = np.linspace(0, 30, 61)
    R_bi = 0.6 * np.exp(-r_bi / 3.0) + 0.4 * np.exp(-r_bi / 12.0)
    da_bi = xr.DataArray(R_bi, dims=("lag_x",), coords={"lag_x": r_bi})

    res_bi = compute_length_scale(da_bi, dim="lag_x", method="fit", fit_model="bi")
    assert "parameter" in res_bi.coords
    params = res_bi.values
    np.testing.assert_allclose(params, [0.6, 3.0, 12.0], atol=1e-2)


def test_quadrant_analysis(sample_turb_ds):
    ds = sample_turb_ds

    # Add quadrant labels and hole mask
    q_ds = ds.piv.add_quadrants(x_var="u", y_var="v", hole_size=0.0)
    assert "quadrant" in q_ds.coords
    assert "hole" in q_ds.coords
    assert set(np.unique(q_ds["quadrant"].values)).issubset({1, 2, 3, 4})

    # Hole filtering: higher hole size should filter out more points
    q_ds_hole = ds.piv.add_quadrants(x_var="u", y_var="v", hole_size=2.0)
    assert np.sum(q_ds_hole["hole"].values == 1) < np.sum(q_ds["hole"].values == 1)

    # High-level quadrant analysis
    qa = ds.piv.quadrant_analysis(x_var="u", y_var="v")
    assert "fraction" in qa
    assert "duration" in qa
    assert "conditional_u" in qa
    assert "conditional_v" in qa
    assert qa.sizes["quadrant"] == 4

    # Total duration across 4 quadrants should equal 1.0
    total_dur = qa["duration"].sum(dim="quadrant")
    np.testing.assert_allclose(total_dur.values, 1.0, atol=1e-5)

    # Total fraction across 4 quadrants should equal 1.0
    total_frac = qa["fraction"].sum(dim="quadrant")
    np.testing.assert_allclose(total_frac.values, 1.0, atol=1e-5)


def test_smoothing_savgol_and_spline(sample_turb_ds):
    ds = sample_turb_ds

    # Savitzky-Golay 1D smoothing along x
    smoothed_sg = ds.piv.smooth_savgol(var_key="u", dim="x", window_length=5, polyorder=2)
    assert isinstance(smoothed_sg, xr.DataArray)
    assert smoothed_sg.shape == ds["u"].shape
    # Check that variance decreases due to noise suppression
    assert float(smoothed_sg.var().values) <= float(ds["u"].var().values) * 1.05

    # B-Spline smoothing with derivative nu=1 (dudx)
    dudx_spline = ds.piv.smooth_spline(var_key="u", dim="x", deriv=1)
    assert isinstance(dudx_spline, xr.DataArray)
    assert dudx_spline.shape == ds["u"].shape

    # Dispatch through ds.piv.smooth(method='savgol') and ds.piv.smooth(method='spline')
    ds_sg = ds.piv.smooth(method="savgol", window_length=5, polyorder=2)
    assert "u" in ds_sg and "v" in ds_sg
    ds_sp = ds.piv.smooth(method="spline", order=3)
    assert "u" in ds_sp and "v" in ds_sp


def test_non_uniform_spectra():
    # Irregular time signal
    rng = np.random.default_rng(123)
    dt_base = 0.01
    t = np.cumsum(dt_base * (1.0 + 0.3 * rng.uniform(-1, 1, size=200)))
    f0 = 5.0
    u = np.sin(2.0 * np.pi * f0 * t) + 0.1 * rng.normal(size=200)

    ds_1d = xr.Dataset({"u": (("t",), u)}, coords={"t": t})
    spec = ds_1d.piv.non_uniform_spectra(var_key="u", dim="t", selfproducts=True)
    assert "autocorrelation" in spec
    assert "psd" in spec
    assert "lag" in spec.coords
    assert "frequency" in spec.coords

    # Verify peak frequency matches signal frequency (5 Hz)
    pos_freq = spec.where(spec.frequency > 0, drop=True)
    peak_f = float(pos_freq.frequency[np.argmax(pos_freq.psd.values)].values)
    assert abs(peak_f - f0) < 1.0
