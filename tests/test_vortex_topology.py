"""
Tests for Work Package 2: Vortex Topology and Identification Criteria.
"""

import numpy as np
import pytest
import xarray as xr

from pivpy.synthetic import vortex
from pivpy.compute_funcs import (
    gamma1,
    gamma2,
    vorticity_circulation,
    q_criterion,
    okubo_weiss,
    subsbr,
)
import pivpy.pivpy  # noqa: F401


def test_gamma1_lamb_vortex():
    """Gamma1 should have a sharp peak (|Gamma1| ~ 1.0) at the vortex center."""
    ds = vortex(n=65, r0=8.0, vorticity=1.0, mode="lamb")
    ds = ds.piv.gamma1(radius=3, name="g1")

    assert "g1" in ds.data_vars
    assert ds["g1"].attrs["standard_name"] == "Gamma 1"
    assert ds["g1"].attrs["units"] == "dimensionless"

    # Peak must be at domain center (x=32, y=32)
    g1 = ds["g1"].squeeze()
    center_val = float(g1.sel(x=32.0, y=32.0, method="nearest"))
    assert center_val > 0.95  # Peak near 1.0 at vortex core center
    assert center_val >= 2.0 / np.pi  # Exceeds vortex identification threshold


def test_gamma2_galilean_invariance():
    """Gamma2 must be invariant under addition of uniform convection."""
    ds_pure = vortex(n=65, r0=8.0, vorticity=1.0, mode="lamb")
    ds_conv = ds_pure.copy(deep=True)
    ds_conv["u"] = ds_conv["u"] + 15.0
    ds_conv["v"] = ds_conv["v"] - 7.5

    g2_pure = gamma2(ds_pure, radius=3, name="g2")["g2"].squeeze().to_numpy()
    g2_conv = gamma2(ds_conv, radius=3, name="g2")["g2"].squeeze().to_numpy()

    # Interior values (away from borders) should be identical within floating precision
    r = 3
    np.testing.assert_allclose(
        g2_conv[r:-r, r:-r],
        g2_pure[r:-r, r:-r],
        atol=1e-12,
        err_msg="Gamma2 is not Galilean invariant under uniform background flow!",
    )


def test_gamma2_core_boundary():
    """Gamma2 identifies the vortex core region where |Gamma2| >= 2/pi."""
    ds = vortex(n=65, r0=8.0, vorticity=1.0, mode="lamb")
    ds = ds.piv.gamma2(radius=3, name="g2")

    g2 = ds["g2"].squeeze()
    center_val = float(g2.sel(x=32.0, y=32.0, method="nearest"))
    assert center_val > 0.90  # High at center

    # Outer flow (r >> r0, near grid borders) should have Gamma2 below threshold 2/pi
    outer_val = float(g2.sel(x=5.0, y=5.0, method="nearest"))
    assert abs(outer_val) < 2.0 / np.pi


def test_vorticity_circulation_vs_diff():
    """Circulation-based vorticity closely matches finite-difference on smooth data."""
    ds = vortex(n=65, r0=10.0, vorticity=2.0, mode="lamb")
    ds_diff = ds.piv.vorticity(method="differentiation", name="w_diff")
    ds_circ = ds.piv.vorticity(method="circulation", radius=1, name="w_circ")

    w_diff = ds_diff["w_diff"].squeeze().to_numpy()
    w_circ = ds_circ["w_circ"].squeeze().to_numpy()

    # Compare interior region
    r = 2
    rmse = np.sqrt(np.mean((w_diff[r:-r, r:-r] - w_circ[r:-r, r:-r]) ** 2))
    peak = np.max(np.abs(w_diff))
    assert (rmse / peak) < 0.05  # Relative difference < 5%


def test_vorticity_circulation_noise_reduction():
    """Circulation vorticity should exhibit significantly higher SNR / lower noise than 2-point diff."""
    np.random.seed(42)
    ds_clean = vortex(n=65, r0=8.0, vorticity=1.0, mode="lamb")
    ds_noisy = ds_clean.copy(deep=True)
    noise_u = np.random.normal(0, 0.05, ds_noisy["u"].shape)
    noise_v = np.random.normal(0, 0.05, ds_noisy["v"].shape)
    ds_noisy["u"] = ds_noisy["u"] + noise_u
    ds_noisy["v"] = ds_noisy["v"] + noise_v

    w_true = ds_clean.piv.vorticity(method="differentiation", name="w_true")["w_true"].squeeze().to_numpy()
    w_diff_noisy = ds_noisy.piv.vorticity(method="differentiation", name="w_d")["w_d"].squeeze().to_numpy()
    w_circ_noisy = ds_noisy.piv.vorticity(method="circulation", radius=2, name="w_c")["w_c"].squeeze().to_numpy()

    r = 3
    err_diff = np.sqrt(np.mean((w_diff_noisy[r:-r, r:-r] - w_true[r:-r, r:-r]) ** 2))
    err_circ = np.sqrt(np.mean((w_circ_noisy[r:-r, r:-r] - w_true[r:-r, r:-r]) ** 2))

    # Circulation-based method reduces noise error by > 50%
    assert err_circ < 0.5 * err_diff


def test_q_criterion_and_okubo_weiss():
    """Q-criterion and Okubo-Weiss parameters should identify vortex core and satisfy Q_ow ~ -4Q."""
    ds = vortex(n=65, r0=8.0, vorticity=2.0, mode="lamb")
    ds = ds.piv.q_criterion(name="Q")
    ds = ds.piv.okubo_weiss(name="Q_ow")

    Q = ds["Q"].squeeze().to_numpy()
    Q_ow = ds["Q_ow"].squeeze().to_numpy()

    # Core center should have positive Q and negative Q_ow
    center_idx = 32
    assert Q[center_idx, center_idx] > 0
    assert Q_ow[center_idx, center_idx] < 0

    # Theoretical identity: Okubo-Weiss W = -4 * Q in 2D incompressible flow
    np.testing.assert_allclose(Q_ow[5:-5, 5:-5], -4.0 * Q[5:-5, 5:-5], atol=1e-5)


def test_subsbr():
    """Solid body rotation subtraction should nullify a pure solid-body rotation field."""
    # Synthetic rigid body rotation: u = -omega * y, v = omega * x
    x = np.linspace(-10, 10, 21)
    y = np.linspace(-10, 10, 21)
    X, Y = np.meshgrid(x, y)
    omega = 0.5
    u = -omega * Y
    v = omega * X

    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
        },
        coords={"x": x, "y": y},
    )

    ds_sub = ds.piv.subsbr(r0=[0.0, 0.0])
    np.testing.assert_allclose(ds_sub["u"].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(ds_sub["v"].to_numpy(), 0.0, atol=1e-12)


def test_vec2scal_wp2_properties():
    """vec2scal should properly compute WP2 properties with custom names."""
    ds = vortex(n=33, r0=5.0, vorticity=1.0, mode="lamb")

    ds = ds.piv.vec2scal("gamma1", name="g1")
    assert "g1" in ds.data_vars
    assert ds["g1"].attrs["standard_name"] == "Gamma 1"

    ds = ds.piv.vec2scal("gamma2", name="g2")
    assert "g2" in ds.data_vars
    assert ds["g2"].attrs["standard_name"] == "Gamma 2"

    ds = ds.piv.vec2scal("q_criterion", name="Q")
    assert "Q" in ds.data_vars

    ds = ds.piv.vec2scal("okubo_weiss", name="OW")
    assert "OW" in ds.data_vars


def test_multi_frame_vortex_methods():
    """Vortex criteria should seamlessly support multi-frame (3D: y, x, t) datasets."""
    f1 = vortex(n=33, r0=5.0, vorticity=1.0, mode="lamb")
    f2 = vortex(n=33, r0=5.0, vorticity=2.0, mode="lamb")
    ds = xr.concat([f1, f2], dim="t")
    ds["t"] = [0.0, 1.0]

    ds = ds.piv.gamma1(radius=2, name="g1")
    ds = ds.piv.gamma2(radius=2, name="g2")
    ds = ds.piv.vorticity(method="circulation", radius=1, name="w_circ")
    ds = ds.piv.q_criterion(name="Q")
    ds = ds.piv.okubo_weiss(name="OW")

    for var in ["g1", "g2", "w_circ", "Q", "OW"]:
        assert ds[var].shape == (33, 33, 2)
        assert "t" in ds[var].dims
