"""Comprehensive unit tests for Work Package 3: Gradient Calculus, Spatial Filtering & Masking."""

import numpy as np
import pytest
import xarray as xr
from pivpy.synthetic import vortex, channel
from pivpy.schema import build_dataset
import pivpy.pivpy  # Register accessor


def test_normalized_median_test_outlier_detection():
    """Verify that Normalized Median Test detects outliers with zero false positives on smooth flow."""
    ds = vortex(n=65, r0=8.0, vorticity=1.0, mode="lamb")
    u_clean = ds["u"].to_numpy().copy()
    v_clean = ds["v"].to_numpy().copy()

    # Inject discrete spurious outliers
    ds["u"].values[15, 20, 0] = 50.0
    ds["v"].values[40, 45, 0] = -45.0

    flagged = ds.piv.normalized_median_test(radius=1, threshold=2.0, epsilon=0.1, name_mask="mask")

    # Flagged outliers must have chc == 0
    assert flagged["chc"].values[15, 20, 0] == 0.0
    assert flagged["chc"].values[40, 45, 0] == 0.0
    assert flagged["mask"].values[15, 20, 0] == True
    assert flagged["mask"].values[40, 45, 0] == True

    # Check zero false positives
    outlier_count = int(np.sum(flagged["chc"].values == 0.0))
    assert outlier_count == 2


def test_clean_and_inpaint_outliers():
    """Verify that ds.piv.clean replaces outliers via harmonic inpainting within < 1% error."""
    ds_true = vortex(n=65, r0=8.0, vorticity=1.0, mode="lamb")
    u_true = ds_true["u"].to_numpy().copy()
    v_true = ds_true["v"].to_numpy().copy()

    ds_corrupt = ds_true.copy(deep=True)
    # Inject outliers
    ds_corrupt["u"].values[25, 25, 0] = 100.0
    ds_corrupt["v"].values[38, 38, 0] = -120.0

    # Inpaint
    ds_cleaned = ds_corrupt.piv.clean(method="normalized_median", threshold=2.0, inpaint_method=0)

    # Inpainted values should closely match true values
    u_rep = ds_cleaned["u"].values[25, 25, 0]
    v_rep = ds_cleaned["v"].values[38, 38, 0]

    np.testing.assert_allclose(u_rep, u_true[25, 25, 0], atol=0.02)
    np.testing.assert_allclose(v_rep, v_true[38, 38, 0], atol=0.02)

    # Replaced points should now be marked valid (chc == 1)
    assert np.all(ds_cleaned["chc"].values == 1.0)


def test_smooth_gaussian_median_boxcar_butterworth():
    """Verify spatial smoothing filters reduce noise variance while preserving vortex structure."""
    ds_clean = vortex(n=65, r0=8.0, vorticity=1.0, mode="lamb")
    np.random.seed(42)
    noise = np.random.normal(0.0, 0.2, size=ds_clean["u"].shape)

    ds_noisy = ds_clean.copy(deep=True)
    ds_noisy["u"].values += noise
    ds_noisy["v"].values += noise

    var_noisy = float(np.var(ds_noisy["u"].values - ds_clean["u"].values))

    # Test Gaussian smoothing
    ds_gauss = ds_noisy.piv.smooth(sigma=1.5, method="gaussian")
    var_gauss = float(np.var(ds_gauss["u"].values - ds_clean["u"].values))
    assert var_gauss < var_noisy * 0.3

    # Test Median smoothing
    ds_median = ds_noisy.piv.smooth(sigma=3, method="median")
    var_median = float(np.var(ds_median["u"].values - ds_clean["u"].values))
    assert var_median < var_noisy * 0.5

    # Test Boxcar smoothing
    ds_box = ds_noisy.piv.smooth(sigma=3, method="boxcar")
    var_box = float(np.var(ds_box["u"].values - ds_clean["u"].values))
    assert var_box < var_noisy * 0.4

    # Test Butterworth smoothing
    ds_bw = ds_noisy.piv.smooth(sigma=16.0, method="butterworth", order=4.0)
    var_bw = float(np.var(ds_bw["u"].values - ds_clean["u"].values))
    assert var_bw < var_noisy * 0.85

    # Test filter() alias
    ds_filt = ds_noisy.piv.filter(sigma=1.5, method="gaussian")
    np.testing.assert_allclose(ds_filt["u"].values, ds_gauss["u"].values)


def test_gradient_tensor_decomposition():
    """Verify strain rate tensor invariants: trace = divergence, lambda_1 >= lambda_2, max_shear."""
    ds = vortex(n=65, r0=8.0, vorticity=2.0, mode="lamb")

    tensor = ds.piv.gradient_tensor(return_components=True)
    assert "s_xx" in tensor.data_vars
    assert "s_yy" in tensor.data_vars
    assert "s_xy" in tensor.data_vars
    assert "lambda_1" in tensor.data_vars
    assert "lambda_2" in tensor.data_vars
    assert "max_shear" in tensor.data_vars
    assert "strain_angle" in tensor.data_vars

    # Trace identity: s_xx + s_yy = div(u)
    div_expected = ds["u"].differentiate("x") + ds["v"].differentiate("y")
    np.testing.assert_allclose(
        tensor["s_xx"].squeeze().values + tensor["s_yy"].squeeze().values,
        div_expected.squeeze().values,
        atol=1e-12,
    )

    # Principal strain ordering: lambda_1 >= lambda_2
    l1 = tensor["lambda_1"].squeeze().values
    l2 = tensor["lambda_2"].squeeze().values
    assert np.all(l1 >= l2 - 1e-12)

    # Max shear formula: gamma_max = (lambda_1 - lambda_2)/2
    gamma_max = tensor["max_shear"].squeeze().values
    np.testing.assert_allclose(gamma_max, (l1 - l2) / 2.0, atol=1e-12)

    # vec2scal max_shear check
    ds_shear = ds.copy(deep=True).piv.vec2scal("max_shear", name="shear")
    np.testing.assert_allclose(ds_shear["shear"].squeeze().values, gamma_max, atol=1e-12)


def test_material_acceleration_exactness_lamb_vortex():
    """Verify convective material acceleration on steady Lamb vortex equals centripetal acceleration."""
    ds = vortex(n=65, r0=8.0, vorticity=2.0, mode="lamb")
    ds = ds.piv.acceleration(name="accel")

    a_computed = ds["accel"].squeeze().values

    # Analytical centripetal acceleration: a_c = u_theta^2 / r
    u = ds["u"].squeeze().values
    v = ds["v"].squeeze().values
    u_theta = np.sqrt(u**2 + v**2)

    x0, y0 = 32.0, 32.0
    X, Y = np.meshgrid(ds["x"].to_numpy(), ds["y"].to_numpy())
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        a_centripetal = np.where(r > 0, u_theta**2 / np.where(r == 0, 1e-12, r), 0.0)

    # Validate in core annular region away from singularity / boundary
    mask_annulus = (r > 3.0) & (r < 15.0)
    rel_err = np.abs(a_computed[mask_annulus] - a_centripetal[mask_annulus]) / a_centripetal[mask_annulus]
    assert np.max(rel_err) < 0.01  # < 1% relative error


def test_material_acceleration_unsteady_and_vector():
    """Verify unsteady acceleration component d(u)/dt and return_vector option."""
    # Create 3-frame dataset with linear velocity ramp: u(t) = (1 + t)*u0
    ds1 = vortex(n=33, r0=6.0, vorticity=1.0, mode="lamb")
    x = ds1["x"].values
    y = ds1["y"].values
    t = np.array([0.0, 1.0, 2.0])

    u3d = np.zeros((33, 33, 3))
    v3d = np.zeros((33, 33, 3))
    for i, ti in enumerate(t):
        u3d[:, :, i] = (1.0 + ti) * ds1["u"].squeeze().values
        v3d[:, :, i] = (1.0 + ti) * ds1["v"].squeeze().values

    ds_ramp = build_dataset(x=x, y=y, t=t, u=u3d, v=v3d)

    # Vector acceleration
    ds_accel_vec = ds_ramp.piv.acceleration(unsteady=True, return_vector=True)
    assert "ax" in ds_accel_vec.data_vars
    assert "ay" in ds_accel_vec.data_vars

    # At t=1 (middle frame), du/dt = u0
    u0 = ds1["u"].squeeze().values
    # ax should have local part u0 + convective part
    du_dt = ds_accel_vec["ax"].isel(t=1).values - (
        ds_ramp["u"].isel(t=1) * ds_ramp["u"].isel(t=1).differentiate("x")
        + ds_ramp["v"].isel(t=1) * ds_ramp["u"].isel(t=1).differentiate("y")
    ).values

    np.testing.assert_allclose(du_dt, u0, atol=1e-12)


def test_multi_frame_wp3_pipeline():
    """Verify complete filtering, outlier cleaning, gradient tensor on multi-frame datasets."""
    from pivpy.synthetic import multivortex
    ds = multivortex(n_frames=4, n=33, n_vortices=3)
    u_orig = ds["u"].values[10, 10, 2]
    
    # Inject outlier in frame 2
    ds["u"].values[10, 10, 2] = 200.0

    # Pipeline
    ds_cleaned = ds.piv.clean(threshold=2.0)
    assert ds_cleaned["u"].shape == (33, 33, 4)
    # Check outlier was replaced
    np.testing.assert_allclose(ds_cleaned["u"].values[10, 10, 2], u_orig, atol=0.2)

    # Smooth
    ds_smooth = ds_cleaned.piv.smooth(sigma=1.0)
    assert ds_smooth["u"].shape == (33, 33, 4)

    # Gradient tensor
    ds_tensor = ds_smooth.piv.gradient_tensor(return_components=True)
    assert ds_tensor["s_xx"].shape == (33, 33, 4)
    assert ds_tensor["max_shear"].shape == (33, 33, 4)
