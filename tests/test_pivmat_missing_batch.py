import numpy as np
import pytest
import xarray as xr

from pivpy.compute_funcs import (
    setoriginf,
    shiftf,
    smoothf,
    timederivativef,
    truncf,
    zeropadf,
    zerotonanfield,
    statf,
    stresstensor,
    subsbr,
    tempfilterf,
    surfheight,
    spatiotempcorrf,
)
from pivpy.io import vec2mat


def _make_vec(ny=8, nx=10, nt=5, units="mm"):
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-5, 5, ny)
    t = np.arange(nt)
    u = np.zeros((ny, nx, nt), float)
    v = np.zeros((ny, nx, nt), float)
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x", "t"), u, {"units": "m/s"}),
            "v": (("y", "x", "t"), v, {"units": "m/s"}),
            "chc": (("y", "x", "t"), np.ones((ny, nx, nt), int)),
        },
        coords={
            "x": ("x", x, {"units": units}),
            "y": ("y", y, {"units": units}),
            "t": ("t", t, {"units": "frame"}),
        },
    )
    return ds


def _make_scal(ny=8, nx=10, nt=5, units="mm"):
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-5, 5, ny)
    t = np.arange(nt)
    w = np.zeros((ny, nx, nt), float)
    ds = xr.Dataset(
        data_vars={
            "w": (("y", "x", "t"), w, {"units": "au"}),
            "chc": (("y", "x", "t"), np.ones((ny, nx, nt), int)),
        },
        coords={
            "x": ("x", x, {"units": units}),
            "y": ("y", y, {"units": units}),
            "t": ("t", t, {"units": "frame"}),
        },
    )
    return ds


def test_setoriginf_shifts_coords():
    ds = _make_vec()
    out = setoriginf(ds, [1.5, -2.0])
    assert float(out.x.values[0]) == pytest.approx(float(ds.x.values[0]) - 1.5)
    assert float(out.y.values[0]) == pytest.approx(float(ds.y.values[0]) + 2.0)


def test_shiftf_center_sets_center_to_zero():
    ds = _make_vec()
    out = shiftf(ds, "center")
    assert float(out.x.values[0] + out.x.values[-1]) == pytest.approx(0.0)
    assert float(out.y.values[0] + out.y.values[-1]) == pytest.approx(0.0)


def test_smoothf_length_and_average():
    ds = _make_vec(ny=4, nx=4, nt=7)
    # u(t) = t
    for k in range(ds.sizes["t"]):
        ds["u"].values[:, :, k] = k
        ds["v"].values[:, :, k] = 2 * k
    out = smoothf(ds, 3)
    assert out.sizes["t"] == 5
    # first output is average of t=0,1,2 => 1
    assert float(out["u"].isel(t=0).mean()) == pytest.approx(1.0)
    assert float(out["v"].isel(t=0).mean()) == pytest.approx(2.0)


def test_timederivativef_order1_and_order2():
    ds = _make_vec(ny=2, nx=2, nt=5)
    for k in range(ds.sizes["t"]):
        ds["u"].values[:, :, k] = k**2
    d1 = timederivativef(ds, order=1)
    assert d1.sizes["t"] == 4
    assert float(d1["u"].isel(t=0).mean()) == pytest.approx(1.0)  # 1-0
    d2 = timederivativef(ds, order=2)
    assert d2.sizes["t"] == 5
    # centered at t=2: (9-1)/2 = 4
    assert float(d2["u"].isel(t=2).mean()) == pytest.approx(4.0)


def test_zerotonanfield():
    ds = _make_vec(ny=2, nx=2, nt=1)
    ds["u"].values[...] = 0.0
    ds["v"].values[...] = 3.0
    out = zerotonanfield(ds)
    assert np.isnan(out["u"].values).all()
    assert np.isfinite(out["v"].values).all()


def test_zeropadf_makes_square():
    ds = _make_vec(ny=6, nx=10, nt=1)
    out = zeropadf(ds)
    assert out.sizes["x"] == out.sizes["y"]
    assert out.sizes["x"] == 10


def test_truncf_center_square_and_nonzero_crop():
    ds = _make_vec(ny=6, nx=10, nt=1)
    out = truncf(ds)
    assert out.sizes["x"] == out.sizes["y"]
    assert out.sizes["x"] == 6

    ds2 = _make_vec(ny=6, nx=10, nt=1)
    ds2["u"].values[...] = 0.0
    ds2["v"].values[...] = 0.0
    ds2["u"].values[2:4, 3:7, 0] = 1.0
    out2 = truncf(ds2, "nonzero")
    assert out2.sizes["y"] == 2
    assert out2.sizes["x"] == 4


def test_statf_scalar_ignores_zeros():
    ds = _make_scal(ny=2, nx=2, nt=1)
    ds["w"].values[:, :, 0] = np.array([[0.0, 1.0], [2.0, 0.0]])
    st = statf(ds, maxorder=4)
    assert st["n"] == 2
    assert st["zeros"] == 2
    assert st["mean"] == pytest.approx(1.5)


def test_stresstensor_simple():
    ds = _make_vec(ny=2, nx=2, nt=1)
    ds["u"].values[...] = 1.0
    ds["v"].values[...] = 2.0
    t, b = stresstensor(ds)
    assert t[0, 0] == pytest.approx(1.0)
    assert t[1, 1] == pytest.approx(4.0)
    assert t[0, 1] == pytest.approx(2.0)
    assert t[1, 0] == pytest.approx(2.0)
    assert b.shape == (2, 2)


def test_subsbr_removes_solid_body_rotation():
    ds = _make_vec(ny=32, nx=32, nt=1, units="mm")
    x = np.asarray(ds.x.values)
    y = np.asarray(ds.y.values)
    xx, yy = np.meshgrid(x, y)
    omega = 1.0  # rad/s
    # x,y in mm -> m
    xm = xx / 1000.0
    ym = yy / 1000.0
    ds["u"].values[:, :, 0] = -omega * ym
    ds["v"].values[:, :, 0] = +omega * xm
    out = subsbr(ds, r0=[0.0, 0.0])
    assert float(np.nanmean(np.abs(out["u"].values))) == pytest.approx(0.0, abs=5e-3)
    assert float(np.nanmean(np.abs(out["v"].values))) == pytest.approx(0.0, abs=5e-3)


def test_tempfilterf_keeps_dc_for_constant():
    ds = _make_scal(ny=4, nx=4, nt=16)
    ds["w"].values[...] = 3.0
    out_dc = tempfilterf(ds, [1])
    assert float(np.mean(out_dc["w"].values)) == pytest.approx(3.0, rel=1e-6)
    out_not_dc = tempfilterf(ds, [2])
    assert float(np.mean(np.abs(out_not_dc["w"].values))) == pytest.approx(0.0, abs=1e-6)


def test_surfheight_reconstructs_periodic_surface():
    dr = _make_vec(ny=32, nx=32, nt=1, units="mm")
    x = np.asarray(dr.x.values)
    y = np.asarray(dr.y.values)
    xx, yy = np.meshgrid(x, y)

    h0 = 10.0
    A = 0.5
    Lx = float(x[-1] - x[0])
    Ly = float(y[-1] - y[0])
    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    h = h0 + A * np.sin(kx * (xx - x[0])) * np.sin(ky * (yy - y[0]))
    dhdx = A * kx * np.cos(kx * (xx - x[0])) * np.sin(ky * (yy - y[0]))
    dhdy = A * ky * np.sin(kx * (xx - x[0])) * np.cos(ky * (yy - y[0]))

    n = 1.33
    alpha = 1.0 - 1.0 / n
    factor = -1.0 / (alpha * h0)  # H=inf

    dr["u"].values[:, :, 0] = dhdx / factor
    dr["v"].values[:, :, 0] = dhdy / factor

    out = surfheight(dr, h0, np.inf, n, None, "submean")
    w = np.asarray(out["w"].isel(t=0).values)
    w = w - float(np.mean(w)) + h0
    err = np.sqrt(np.mean((w - h) ** 2))
    assert err < 0.2


def test_spatiotempcorrf_shape_and_normalization():
    ds = _make_scal(ny=6, nx=12, nt=10)
    x = np.asarray(ds.x.values)
    t = np.asarray(ds.t.values)
    # periodic pattern over x and t
    kx = 2.0 * np.pi / float(x[-1] - x[0])
    wt = 2.0 * np.pi / float(t[-1] - t[0] + 1)
    xx = x[None, :, None]
    tt = t[None, None, :]
    ds["w"].values[...] = np.cos(kx * (xx - x[0]) + wt * tt)
    out = spatiotempcorrf(ds)
    assert "cor" in out.data_vars
    # C(0,0) normalized to 1
    x0_idx = int(np.where(out["X"].values == 0)[0][0])
    assert float(out["cor"].values[x0_idx, 0]) == pytest.approx(1.0)


def test_vec2mat_writes_mat(tmp_path):
    ds = _make_vec(ny=4, nx=4, nt=2)
    outname = vec2mat(ds, str(tmp_path / "out"))
    assert outname.endswith(".mat")
    assert (tmp_path / "out.mat").exists()
