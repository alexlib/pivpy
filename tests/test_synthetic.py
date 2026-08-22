"""tests/test_synthetic.py

Comprehensive test suite for pivpy.synthetic analytical flow field generators.
"""

import numpy as np
import pytest
import xarray as xr

from pivpy import synthetic
from pivpy.schema import validate, is_valid


def test_vortex_lamb_schema_and_center():
    ds = synthetic.vortex(n=65, r0=10.0, vorticity=2.0, mode="lamb", dx=1.0, dy=1.0)
    assert is_valid(ds)
    validate(ds)

    assert ds.sizes["x"] == 65
    assert ds.sizes["y"] == 65
    assert ds.sizes["t"] == 1
    assert "u" in ds.data_vars
    assert "v" in ds.data_vars
    assert "chc" in ds.data_vars

    # Center is at x=32, y=32 (index 32, 32)
    u_center = float(ds["u"].sel(x=32.0, y=32.0).isel(t=0))
    v_center = float(ds["v"].sel(x=32.0, y=32.0).isel(t=0))
    assert abs(u_center) < 1e-10
    assert abs(v_center) < 1e-10

    # Test peak vorticity using piv accessor
    ds_vort = ds.piv.vorticity(name="vort")
    vort_center = float(ds_vort["vort"].sel(x=32.0, y=32.0).isel(t=0))
    # Expected peak vorticity is ~ 2.0 s^-1
    assert abs(vort_center - 2.0) < 0.05


def test_vortex_rankine():
    ds = synthetic.vortex(n=65, r0=10.0, vorticity=2.0, mode="rankine", dx=1.0, dy=1.0)
    validate(ds)

    # Inside core (e.g. r=5 on the positive x-axis from center (32, 32)):
    # x = 37, y = 32 => rx = 5, ry = 0 => u = 0, v = omega * rx = 1.0 * 5 = 5.0
    u_in = float(ds["u"].sel(x=37.0, y=32.0).isel(t=0))
    v_in = float(ds["v"].sel(x=37.0, y=32.0).isel(t=0))
    assert abs(u_in) < 1e-10
    assert abs(v_in - 5.0) < 1e-10

    # Outside core (e.g. r=20 on positive x-axis => x = 52, y = 32):
    # v = omega * r0^2 / r = 1.0 * 100 / 20 = 5.0
    v_out = float(ds["v"].sel(x=52.0, y=32.0).isel(t=0))
    assert abs(v_out - 5.0) < 1e-10


def test_vortex_vatistas():
    ds = synthetic.vortex(n=65, r0=10.0, vorticity=2.0, mode="vatistas", n_vatistas=2.0)
    validate(ds)

    # Center velocity is 0
    u_center = float(ds["u"].sel(x=32.0, y=32.0).isel(t=0))
    v_center = float(ds["v"].sel(x=32.0, y=32.0).isel(t=0))
    assert abs(u_center) < 1e-10
    assert abs(v_center) < 1e-10


def test_vortex_invalid_mode():
    with pytest.raises(ValueError, match="Unknown vortex mode"):
        synthetic.vortex(mode="invalid_mode")


def test_vortex_divergence():
    ds = synthetic.vortex(n=65, r0=10.0, vorticity=2.0, mode="burgers", diver=2.0)
    validate(ds)

    # On positive x-axis, divergence adds positive u component
    u_pt = float(ds["u"].sel(x=42.0, y=32.0).isel(t=0))
    assert u_pt > 0.0


def test_multivortex_reproducibility_and_frames():
    ds1 = synthetic.multivortex(n_frames=2, n=48, n_vortices=4, seed=42)
    ds2 = synthetic.multivortex(n_frames=2, n=48, n_vortices=4, seed=42)
    validate(ds1)

    assert ds1.sizes["t"] == 2
    assert ds1.sizes["x"] == 48
    assert ds1.sizes["y"] == 48

    np.testing.assert_allclose(ds1["u"].values, ds2["u"].values)
    np.testing.assert_allclose(ds1["v"].values, ds2["v"].values)


def test_randvec_divergence_free():
    ds = synthetic.randvec(n=64, n_frames=2, seed=123, dx=1.0, dy=1.0)
    validate(ds)
    assert ds.sizes["t"] == 2

    # Compute divergence using central differences
    u = ds["u"].isel(t=0).values
    v = ds["v"].isel(t=0).values
    dudx = np.gradient(u, 1.0, axis=1)
    dvdy = np.gradient(v, 1.0, axis=0)
    div = dudx + dvdy

    # In periodic Fourier space, interior divergence is zero within FFT accuracy
    interior_div = div[8:-8, 8:-8]
    assert np.max(np.abs(interior_div)) < 1e-2


def test_channel_poiseuille():
    ds = synthetic.channel(rows=33, cols=20, u_max=10.0, dy=1.0)
    validate(ds)

    y_mid = float(ds.coords["y"][16])
    u_mid = float(ds["u"].sel(y=y_mid).isel(x=0, t=0))
    assert abs(u_mid - 10.0) < 1e-10

    # Walls at y[0] and y[-1]
    u_wall_bot = float(ds["u"].isel(y=0, x=0, t=0))
    u_wall_top = float(ds["u"].isel(y=-1, x=0, t=0))
    assert abs(u_wall_bot) < 1e-10
    assert abs(u_wall_top) < 1e-10
    assert np.all(ds["v"].values == 0.0)


def test_shear_layer():
    ds = synthetic.shear_layer(rows=65, cols=32, u0=5.0, delta=8.0, perturbation=0.0)
    validate(ds)

    yc = float(ds.coords["y"][32])
    assert abs(float(ds["u"].sel(y=yc).isel(x=0, t=0))) < 1e-10
    assert float(ds["u"].isel(y=-1, x=0, t=0)) > 4.5
    assert float(ds["u"].isel(y=0, x=0, t=0)) < -4.5
