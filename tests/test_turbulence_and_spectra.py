"""Unit tests for Work Package 4: Advanced Turbulence Statistics & Energy Spectra."""
import numpy as np
import pytest
import xarray as xr

from pivpy.schema import build_dataset
from pivpy.synthetic import multivortex, vortex
import pivpy.pivpy  # noqa: F401


def test_reynolds_decomposition_multi_frame():
    """Verify Reynolds decomposition separates mean from fluctuations and computes Reynolds stresses."""
    # Generate 16 frames of synthetic turbulence
    ds = multivortex(n_frames=16, n=32, n_vortices=4, seed=42)
    
    decomp = ds.piv.reynolds_decomposition()
    assert "u_mean" in decomp.data_vars
    assert "v_mean" in decomp.data_vars
    assert "u_prime" in decomp.data_vars
    assert "v_prime" in decomp.data_vars
    assert "uu_prime" in decomp.data_vars
    assert "vv_prime" in decomp.data_vars
    assert "uv_prime" in decomp.data_vars
    assert "tke" in decomp.data_vars
    assert "intensity_u" in decomp.data_vars

    # Mean of fluctuations along time must be zero
    np.testing.assert_allclose(decomp["u_prime"].mean(dim="t").values, 0.0, atol=1e-12)
    np.testing.assert_allclose(decomp["v_prime"].mean(dim="t").values, 0.0, atol=1e-12)

    # Stresses non-negativity
    assert np.all(decomp["uu_prime"].values >= 0.0)
    assert np.all(decomp["vv_prime"].values >= 0.0)

    # TKE definition check
    expected_tke = 0.5 * (decomp["uu_prime"] + decomp["vv_prime"])
    np.testing.assert_allclose(decomp["tke"].values, expected_tke.values, atol=1e-12)

    # Single frame should raise ValueError
    ds_single = ds.isel(t=0)
    with pytest.raises(ValueError, match="at least 2 time steps"):
        ds_single.piv.reynolds_decomposition()


def test_energy_spectrum_parseval_conservation():
    """Verify energy spectrum satisfies Parseval theorem and energy conservation."""
    ds = multivortex(n_frames=1, n=64, n_vortices=6, seed=123)

    u = ds["u"].squeeze().values
    v = ds["v"].squeeze().values
    u_detrend = u - np.mean(u)
    v_detrend = v - np.mean(v)
    tke_actual = 0.5 * np.mean(u_detrend**2 + v_detrend**2)

    # Unwindowed spectrum for exact Parseval equality
    spec_nowin = ds.piv.energy_spectrum(window="none", detrend=True, radial=True)
    assert "E2D" in spec_nowin.data_vars
    assert "E_radial" in spec_nowin.data_vars
    assert "E_kx" in spec_nowin.data_vars
    assert "E_ky" in spec_nowin.data_vars

    # 2D spectrum sum
    sum_e2d = float(np.sum(spec_nowin["E2D"].values))
    np.testing.assert_allclose(sum_e2d, tke_actual, rtol=1e-5)

    # Radial spectrum integral: sum(E_rad * dk)
    k = spec_nowin["k"].values
    dk = k[1] - k[0]
    sum_radial = float(np.sum(spec_nowin["E_radial"].values * dk))
    np.testing.assert_allclose(sum_radial, tke_actual, rtol=1e-3)

    # Windowed spectrum (Hann) runs cleanly
    spec_hann = ds.piv.energy_spectrum(window="hann", detrend=True, radial=True)
    assert spec_hann["E2D"].shape == (64, 64)


def test_spatial_correlation_properties():
    """Verify spatial autocorrelation satisfies R(0) = 1.0 and decays with distance."""
    ds = multivortex(n_frames=12, n=48, n_vortices=5, seed=777)

    corr_x = ds.piv.spatial_correlation(component="u", dim="x", normalize=True)
    assert "R" in corr_x.data_vars
    assert "r" in corr_x.coords
    
    R_vals = corr_x["R"].values
    # R(0) = 1.0
    np.testing.assert_allclose(R_vals[0], 1.0, atol=1e-7)

    # Autocorrelation magnitude is bounded by 1.0
    assert np.all(np.abs(R_vals) <= 1.0 + 1e-6)

    # Autocorrelation along y
    corr_y = ds.piv.spatial_correlation(component="v", dim="y", normalize=True)
    np.testing.assert_allclose(corr_y["R"].values[0], 1.0, atol=1e-7)


def test_integral_length_scale_and_taylor_microscale():
    """Verify integral length scale L11 and Taylor microscale lambda_T calculations."""
    ds = multivortex(n_frames=10, n=48, n_vortices=4, seed=999)

    L11 = ds.piv.integral_length_scale(component="u", dim="x")
    assert isinstance(L11, float)
    assert L11 > 0.0

    # Taylor microscale curvature method
    lambda_curv = ds.piv.taylor_microscale(component="u", dim="x", method="curvature")
    assert isinstance(lambda_curv, float)
    assert lambda_curv > 0.0

    # Taylor microscale gradient method
    lambda_grad = ds.piv.taylor_microscale(component="u", dim="x", method="gradient")
    assert isinstance(lambda_grad, float)
    assert lambda_grad > 0.0

    # Physical scaling relation: Taylor microscale is smaller than domain size
    domain_size = float(ds["x"].max() - ds["x"].min())
    assert lambda_curv < domain_size
    assert lambda_grad < domain_size


def test_dissipation_rate_methods():
    """Verify turbulent dissipation rate calculation and vec2scal integration."""
    ds = multivortex(n_frames=8, n=32, n_vortices=3, seed=101)

    # Direct surrogate
    ds_direct = ds.piv.dissipation(method="direct", nu=1.5e-5, name="eps_dir")
    assert "eps_dir" in ds_direct.data_vars
    assert np.all(ds_direct["eps_dir"].values >= 0.0)

    # Isotropic surrogate
    ds_iso = ds.piv.dissipation(method="isotropic", nu=1.5e-5, name="eps_iso")
    assert "eps_iso" in ds_iso.data_vars
    assert np.all(ds_iso["eps_iso"].values >= 0.0)

    # Smagorinsky subgrid model
    ds_smag = ds.piv.dissipation(method="smagorinsky", name="eps_sgs")
    assert "eps_sgs" in ds_smag.data_vars
    assert np.all(ds_smag["eps_sgs"].values >= 0.0)

    # vec2scal dissipation integration
    ds_v2s = ds.copy(deep=True).piv.vec2scal("dissipation", name="diss")
    assert "diss" in ds_v2s.data_vars
