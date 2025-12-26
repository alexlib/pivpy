""" tests pivpy.pivpy methods """
import pathlib
import numpy as np
import importlib.resources
import xarray as xr
import pytest
from pivpy import io
import pivpy.pivpy  # Register the piv accessor for xarray.Dataset


FILE1 = "Run000001.T000.D000.P000.H001.L.vec"
FILE2 = "Run000002.T000.D000.P000.H001.L.vec"

# Ensure compatibility with different Python versions (3.9+ has 'files', 3.7 and 3.8 need 'path')
try:
    from importlib.resources import files
except ImportError:
    from importlib.resources import path as resource_path

# For Python 3.9+
try:
    path = files('pivpy') / 'data'
except NameError:
    # For Python 3.7 and 3.8
    with resource_path('pivpy', 'data') as data_path:
        path = data_path
        
path = path / "Insight"


_a = io.load_vec(path / FILE1)
_b = io.load_vec(path / FILE2)


def test_crop():
    """tests crop"""
    _c = _a.piv.crop([5, 15, -5, -15])
    assert _c.u.shape == (32, 32, 1)


def test_select_roi():
    """tests xarray selection option on our dataset"""
    _c = io.create_sample_Dataset(n_frames=5, rows=10, cols=10)
    _c = _c.sel(x=slice(35, 70), y=slice(30, 90))
    assert _c.u.shape == (7, 2, 5)  # note the last dimension is preserved


def test_pan():
    """test a shift by dx,dy using pan method"""
    _c = _a.copy()
    _c = _c.piv.pan(1.0, -1.0)  # note the use of .piv.
    assert np.allclose(_c.coords["x"][0], 1.312480)
    assert np.allclose(_c.coords["y"][0], -1.31248)


def test_mean():
    """tests mean or average property"""
    data = io.create_sample_Dataset(10)
    print(data.piv.average.u.median())
    assert np.allclose(data.piv.average.u.median(), 6.0)


def test_vec2scal():
    """tests vec2scal"""
    data = io.create_sample_Dataset()
    data = data.piv.vec2scal()  # default is curl
    assert data["w"].attrs["standard_name"] == "vorticity"

    data = data.piv.vec2scal(flow_property="strain")
    assert data["w"].attrs["standard_name"] == "strain"


def test_vec2scal_custom_name():
    """tests vec2scal with custom name parameter"""
    data = io.create_sample_Dataset()
    data = data.piv.vec2scal(flow_property="vorticity", name="vort")
    assert "vort" in data
    assert data["vort"].attrs["standard_name"] == "vorticity"
    
    # Add another scalar with different name
    data = data.piv.vec2scal(flow_property="strain", name="strain_field")
    assert "strain_field" in data
    assert data["strain_field"].attrs["standard_name"] == "strain"
    
    # Both should exist
    assert "vort" in data
    assert "strain_field" in data


def test_add():
    """tests addition of two datasets"""
    data = io.create_sample_Dataset()
    tmp = data + data
    assert tmp["u"][0, 0, 0] == 2.0


def test_subtract():
    """tests subtraction"""
    data = io.create_sample_Dataset()
    tmp = data - data
    assert tmp["u"][0, 0, 0] == 0.0


def test_multiply():
    """tests subtraction"""
    data = io.create_sample_Dataset()
    tmp = data * 3.5
    assert tmp["u"][0, 0, 0] == 3.5


def test_set_get_dt():
    """tests setting the new dt"""
    data = io.create_sample_Dataset()
    assert data.attrs["delta_t"] == 0.0

    data.piv.set_delta_t(2.0)
    assert data.attrs["delta_t"] == 2.0


# def test_rotate():
#     """ tests rotation """
#     data = io.create_sample_Dataset()
#     data.piv.rotate(90) # rotate by 90 deg
#     assert data['u'][0,0,0] == 2.1 # shall fail


def test_fluctuations():
    """tests fluctuations, velocity fields are replaced"""
    data = io.create_sample_field()
    with pytest.raises(ValueError):
        data.piv.fluct()

    data = io.create_sample_Dataset(100)  # enough for random
    fluct = data.piv.fluct()
    assert np.allclose(fluct["u"].mean(dim="t"), 0.0)
    assert np.allclose(fluct["v"].mean(dim="t"), 0.0)


def test_reynolds_stress():
    """tests Reynolds stress"""
    data = io.create_sample_Dataset(2, noise_sigma=0.0)
    data.isel(t=1)["u"] += 0.1
    data.isel(t=1)["v"] -= 0.1
    tmp = data.piv.reynolds_stress()
    assert np.allclose(tmp["w"], 0.0025)
    assert tmp["w"].attrs["standard_name"] == "Reynolds_stress"


def test_set_scale():
    """tests scaling the dataset by a scalar"""
    data = io.create_sample_Dataset()
    tmp = data.piv.set_scale(1.0)
    assert np.allclose(tmp["x"], data["x"])

    tmp = data.copy()
    tmp.piv.set_scale(2.0)
    tmp_mean = tmp["u"].mean(dim=("t", "x", "y")).values
    data_mean = data["u"].mean(dim=("t", "x", "y")).values
    assert np.allclose(tmp_mean / data_mean, 2.0)


def test_vorticity():
    """tests vorticity estimate"""
    data = io.create_sample_field()  # we need another flow field
    data.piv.vorticity()
    assert np.allclose(data["w"], 0.0)


def test_vorticity_custom_name():
    """tests vorticity with custom name"""
    data = io.create_sample_field()
    data.piv.vorticity(name="vort")
    assert "vort" in data
    assert data["vort"].attrs["standard_name"] == "vorticity"


def test_multiple_scalars_in_dataset():
    """tests storing multiple scalar fields in one dataset"""
    data = io.create_sample_Dataset(n_frames=5, rows=5, cols=5)
    # Add multiple scalars with different names
    data = data.piv.vorticity(name="vort")
    data = data.piv.strain(name="strain")
    data = data.piv.kinetic_energy(name="ke")
    data = data.piv.tke(name="tke")
    data = data.piv.reynolds_stress(name="rey_stress")
    
    # Check all are present
    assert "vort" in data
    assert "strain" in data
    assert "ke" in data
    assert "tke" in data
    assert "rey_stress" in data
    
    # Check attributes are correct
    assert data["vort"].attrs["standard_name"] == "vorticity"
    assert data["strain"].attrs["standard_name"] == "strain"
    assert data["ke"].attrs["standard_name"] == "kinetic_energy"
    assert data["tke"].attrs["standard_name"] == "TKE"
    assert data["rey_stress"].attrs["standard_name"] == "Reynolds_stress"


def test_strain():
    """tests shear estimate"""
    data = io.create_sample_field(rows=3, cols=3, noise_sigma=0.0)
    data = data.piv.strain()
    assert np.allclose(data["w"].values, 0.11328125, 1e-6)
    # also after scaling
    data.piv.set_scale(1/16) # 16 pixels is the grid
    data = data.piv.strain()
    assert np.allclose(data["w"].values, 0.11328125, 1e-6)


def test_tke():
    """tests TKE"""
    data = io.create_sample_Dataset()
    data = data.piv.tke()  # now defined
    assert data["w"].attrs["standard_name"] == "TKE"

def test_Γ1():
    """tests Γ1"""
    data = io.create_sample_Dataset(n_frames=4, rows=3, cols=2)
    data = data.piv.Γ1(n=1)  
    assert data["Γ1"].to_numpy().shape == (2,3,4)
    assert data["Γ1"].attrs["standard_name"] == "Gamma 1"
    assert data["Γ1"].attrs["units"] == "dimensionless"

def test_Γ2():
    """tests Γ2"""
    data = io.create_sample_Dataset(n_frames=2, rows=3, cols=4)
    data = data.piv.Γ2(n=1)  
    assert data["Γ2"].to_numpy().shape == (4,3,2)
    assert data["Γ2"].attrs["standard_name"] == "Gamma 2"
    assert data["Γ2"].attrs["units"] == "dimensionless"

def test_curl():
    """tests curl that is also vorticity"""
    _c = _a.copy()
    _c.piv.vec2scal(flow_property="curl")

    assert _c["w"].attrs["standard_name"] == "vorticity"

def test_fill_nans():
    """" tests fill_nans function """
    ds = io.create_sample_Dataset(n_frames=1,rows=7,cols=11,noise_sigma=0.5)
    ds["u"][1:4,1:4] = np.nan
    # ds.sel(t=0)["u"].plot()
    new = ds.copy(deep=True) # prepare memory for the result
    new.piv.fill_nans() # fill nans
    assert ds.dropna(dim='x')["v"].shape == (7, 8, 1)
    assert new.dropna(dim='x')["v"].shape == (7, 11, 1)

def test_filterf():
    """ tests filterf
    """
    dataset = io.create_sample_Dataset(n_frames=3,rows=5,cols=10)
    dataset = dataset.piv.filterf() # no inputs
    dataset = dataset.piv.filterf([.5, .5, 0.]) # with sigma
    # ds["mag"] = np.hypot(ds["u"], ds["v"])
    # ds.plot.quiver(x='x',y='y',u='u',v='v',hue='mag',col='t',scale=150,cmap='RdBu')


def test_addnoisef_eps_zero_no_change():
    ds = io.create_sample_Dataset(n_frames=2, rows=5, cols=6, noise_sigma=0.0)
    before_u = ds["u"].values.copy()
    before_v = ds["v"].values.copy()
    ds = ds.piv.addnoisef(eps=0.0, seed=0)
    assert np.allclose(ds["u"].values, before_u)
    assert np.allclose(ds["v"].values, before_v)


def test_addnoisef_additive_reproducible():
    ds1 = io.create_sample_Dataset(n_frames=2, rows=5, cols=6, noise_sigma=0.0)
    ds2 = io.create_sample_Dataset(n_frames=2, rows=5, cols=6, noise_sigma=0.0)
    ds1 = ds1.piv.addnoisef(eps=0.1, opt="add", nc=0.0, seed=123)
    ds2 = ds2.piv.addnoisef(eps=0.1, opt="add", nc=0.0, seed=123)
    assert np.allclose(ds1["u"].values, ds2["u"].values)
    assert np.allclose(ds1["v"].values, ds2["v"].values)
    assert not np.allclose(ds1["u"].values, io.create_sample_Dataset(n_frames=2, rows=5, cols=6, noise_sigma=0.0)["u"].values)


def test_addnoisef_multiplicative_runs():
    ds = io.create_sample_Dataset(n_frames=2, rows=5, cols=6, noise_sigma=0.0)
    ds = ds.piv.addnoisef(eps=0.05, opt="mul", nc=2.0, seed=0)
    assert ds["u"].shape == (5, 6, 2)
    assert ds["v"].shape == (5, 6, 2)


def test_averf_default_excludes_zeros():
    ds = io.create_sample_Dataset(n_frames=3, rows=2, cols=2, noise_sigma=0.0)
    # Make one sample invalid by setting both components to zero at t=1
    u_ref = float(ds["u"].values[0, 0, 0])
    v_ref = float(ds["v"].values[0, 0, 0])
    ds["u"].values[0, 0, 1] = 0.0
    ds["v"].values[0, 0, 1] = 0.0

    avg_excl = ds.piv.averf()  # default excludes zeros
    assert avg_excl["u"].shape == (2, 2, 1)
    assert np.allclose(avg_excl["u"].values[0, 0, 0], u_ref)
    assert np.allclose(avg_excl["v"].values[0, 0, 0], v_ref)


def test_averf_opt0_includes_zeros():
    ds = io.create_sample_Dataset(n_frames=3, rows=2, cols=2, noise_sigma=0.0)
    u_ref = float(ds["u"].values[0, 0, 0])
    v_ref = float(ds["v"].values[0, 0, 0])
    ds["u"].values[0, 0, 1] = 0.0
    ds["v"].values[0, 0, 1] = 0.0

    avg_incl = ds.piv.averf("0")
    assert np.allclose(avg_incl["u"].values[0, 0, 0], (2.0 * u_ref + 0.0) / 3.0)
    assert np.allclose(avg_incl["v"].values[0, 0, 0], (2.0 * v_ref + 0.0) / 3.0)


def test_averf_returns_std_and_rms():
    ds = io.create_sample_Dataset(n_frames=3, rows=3, cols=4, noise_sigma=0.0)
    ds["u"].values[0, 0, 1] = 0.0
    ds["v"].values[0, 0, 1] = 0.0
    avg, std, rms = ds.piv.averf(return_std_rms=True)
    assert avg["u"].shape == (3, 4, 1)
    assert "w" in std
    assert "w" in rms
    assert std["w"].shape == (3, 4, 1)
    assert rms["w"].shape == (3, 4, 1)
    assert np.all(std["w"].values >= 0)
    assert np.all(rms["w"].values >= 0)


def test_azaverf_vector_profiles_solid_body_rotation():
    # Solid-body rotation: u=-y, v=x around origin => ur=0, ut=r.
    x = np.linspace(-5.0, 5.0, 21)
    y = np.linspace(-5.0, 5.0, 21)
    X, Y = np.meshgrid(x, y)
    u = -Y
    v = X
    ds = io.from_arrays(X, Y, u, v, frame=0)

    r, ur, ut = ds.piv.azaverf(0.0, 0.0, return_profiles=True)
    # Profiles are (nr, nt); here nt=1
    assert ur.shape[1] == 1
    assert ut.shape[1] == 1
    # Expect near-zero radial component away from center
    if r.size:
        assert np.nanmean(np.abs(ur[:, 0])) < 1e-6
        # Tangential speed should approximately match radius
        # (ignore the first bin which may have few samples)
        if r.size > 2:
            assert np.nanmean(np.abs(ut[2:, 0] - r[2:])) < 0.2


def test_azaverf_scalar_profiles():
    # Scalar field depending only on radius should preserve that profile.
    x = np.linspace(-4.0, 4.0, 17)
    y = np.linspace(-4.0, 4.0, 17)
    X, Y = np.meshgrid(x, y)
    r0 = np.sqrt(X * X + Y * Y)
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    ds = io.from_arrays(X, Y, u, v, frame=0)
    ds["s"] = (("y", "x", "t"), (r0[:, :, None]))

    r, p = ds.piv.azaverf(0.0, 0.0, var="s", return_profiles=True, keepzero=True)
    assert p.shape[1] == 1
    if r.size > 2:
        assert np.nanmean(np.abs(p[2:, 0] - r[2:])) < 0.2


def test_phaseaverf_integer_period_groups_frames():
    ds = io.create_sample_Dataset(n_frames=6, rows=2, cols=2, noise_sigma=0.0)
    # Add a time-dependent offset so grouping matters.
    for ti in range(6):
        ds["u"].isel(t=ti).values[...] = ds["u"].isel(t=ti).values + float(ti)
        ds["v"].isel(t=ti).values[...] = ds["v"].isel(t=ti).values + float(2 * ti)

    out = ds.piv.phaseaverf(2)
    assert out["u"].shape == (2, 2, 2)
    # Phase 0 averages t=0,2,4 => mean offset is 2
    base_u = io.create_sample_Dataset(n_frames=1, rows=2, cols=2, noise_sigma=0.0)["u"].values[:, :, 0]
    base_v = io.create_sample_Dataset(n_frames=1, rows=2, cols=2, noise_sigma=0.0)["v"].values[:, :, 0]
    assert np.allclose(out["u"].isel(t=0).values, base_u + 2.0)
    assert np.allclose(out["v"].isel(t=0).values, base_v + 4.0)
    # Phase 1 averages t=1,3,5 => mean offset is 3
    assert np.allclose(out["u"].isel(t=1).values, base_u + 3.0)
    assert np.allclose(out["v"].isel(t=1).values, base_v + 6.0)


def test_phaseaverf_float_period_interpolates():
    # Build a dataset where u(t)=t (same everywhere), so interpolation is predictable.
    n = 6
    x = np.linspace(0.0, 1.0, 3)
    y = np.linspace(0.0, 1.0, 2)
    X, Y = np.meshgrid(x, y)
    u0 = np.zeros_like(X)
    v0 = np.zeros_like(X)
    ds = io.from_arrays(X, Y, u0, v0, frame=0)
    # Expand to time and set u(t)=t
    frames = []
    for ti in range(n):
        d = io.from_arrays(X, Y, u0 + float(ti), v0, frame=ti)
        frames.append(d)
    ds = xr.concat(frames, dim="t")

    out = ds.piv.phaseaverf(2.5)
    assert out.sizes["t"] == 2
    # Phase 0 samples points [0,2.5,5], but averf excludes zeros by default,
    # so the 0-sample is ignored and mean becomes (2.5+5)/2 = 3.75.
    assert np.allclose(out["u"].isel(t=0).values.mean(), 3.75)
    # Phase 1 samples points [1,3.5] => mean = 2.25
    assert np.allclose(out["u"].isel(t=1).values.mean(), 2.25)


def test_spaverf_xy_constant_includes_zeros_with_opt0():
    ds = io.create_sample_Dataset(n_frames=1, rows=3, cols=4, noise_sigma=0.0)
    out = ds.piv.spaverf("xy0")
    mu = float(ds["u"].isel(t=0).values.mean())
    mv = float(ds["v"].isel(t=0).values.mean())
    assert np.allclose(out["u"].isel(t=0).values, mu)
    assert np.allclose(out["v"].isel(t=0).values, mv)


def test_spaverf_xy_excludes_zeros_by_default():
    ds = io.create_sample_Dataset(n_frames=1, rows=3, cols=4, noise_sigma=0.0)
    ds["u"].values[0, 0, 0] = 0.0
    ds["v"].values[0, 0, 0] = 0.0
    out = ds.piv.spaverf("xy")

    u = ds["u"].isel(t=0).values
    v = ds["v"].isel(t=0).values
    mu = float(u[u != 0].mean())
    mv = float(v[v != 0].mean())
    assert np.allclose(out["u"].isel(t=0).values, mu)
    assert np.allclose(out["v"].isel(t=0).values, mv)


def test_spaverf_x_broadcasts_over_x():
    ds = io.create_sample_Dataset(n_frames=2, rows=4, cols=5, noise_sigma=0.0)
    out = ds.piv.spaverf("x0")
    # Should be uniform in x (for each y,t)
    assert np.allclose(out["u"].isel(x=0).values, out["u"].isel(x=-1).values)
    # And should match mean over x
    expected = ds["u"].mean(dim="x")
    assert np.allclose(out["u"].values, expected.broadcast_like(ds["u"]).values)


def test_spaverf_scalar_var():
    ds = io.create_sample_Dataset(n_frames=1, rows=3, cols=4, noise_sigma=0.0)
    ds = ds.piv.vorticity(name="w")
    out = ds.piv.spaverf("y0", var="w")
    # uniform in y for each x,t
    assert np.allclose(out["w"].isel(y=0).values, out["w"].isel(y=-1).values)


def test_subaverf_ensemble_removes_time_mean():
    ds = io.create_sample_Dataset(n_frames=6, rows=3, cols=4, noise_sigma=0.0)
    # Make u vary in time so ensemble subtraction is meaningful.
    for ti in range(ds.sizes["t"]):
        ds["u"].isel(t=ti).values[...] = ds["u"].isel(t=ti).values + float(ti)
        ds["v"].isel(t=ti).values[...] = ds["v"].isel(t=ti).values - float(2 * ti)

    out = ds.piv.subaverf("e")
    # Mean over time should be ~0 everywhere.
    assert np.allclose(out["u"].mean(dim="t").values, 0.0)
    assert np.allclose(out["v"].mean(dim="t").values, 0.0)


def test_subaverf_spatial_x_removes_x_mean():
    ds = io.create_sample_Dataset(n_frames=2, rows=4, cols=5, noise_sigma=0.0)
    # Force an x-dependent bias.
    bias = np.arange(ds.sizes["x"], dtype=float)[None, :, None]
    ds["u"].values[...] = ds["u"].values + bias
    out = ds.piv.subaverf("x0")
    # After subtracting x-mean, mean along x should be ~0.
    assert np.allclose(out["u"].mean(dim="x").values, 0.0)


def test_subaverf_preserves_zero_locations():
    ds = io.create_sample_Dataset(n_frames=3, rows=3, cols=4, noise_sigma=0.0)
    ds["u"].values[0, 0, 1] = 0.0
    ds["v"].values[0, 0, 1] = 0.0
    out = ds.piv.subaverf("e")
    assert out["u"].values[0, 0, 1] == 0.0
    assert out["v"].values[0, 0, 1] == 0.0


def test_resamplef_linear_interpolation():
    # Build dataset with u(t)=t everywhere so interpolation is predictable.
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 1.0, 3)
    X, Y = np.meshgrid(x, y)
    frames = []
    for ti in range(5):
        frames.append(io.from_arrays(X, Y, np.full_like(X, float(ti)), np.zeros_like(X), frame=ti))
    ds = xr.concat(frames, dim="t")

    tini = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    tfin = np.array([0.5, 1.5, 2.5, 3.5])
    out = ds.piv.resamplef(tini, tfin)
    assert out.sizes["t"] == 4
    assert np.allclose(out["u"].isel(t=0).values.mean(), 0.5)
    assert np.allclose(out["u"].isel(t=1).values.mean(), 1.5)
    assert np.allclose(out["u"].isel(t=2).values.mean(), 2.5)
    assert np.allclose(out["u"].isel(t=3).values.mean(), 3.5)


def test_resamplef_validation_errors():
    ds = io.create_sample_Dataset(n_frames=3, rows=2, cols=2)
    with pytest.raises(ValueError):
        ds.piv.resamplef([0.0, 1.0], [0.5])  # length mismatch
    with pytest.raises(ValueError):
        ds.piv.resamplef([0.0, 0.0, 1.0], [0.5])  # not strictly increasing
    with pytest.raises(ValueError):
        ds.piv.resamplef([0.0, 1.0, 2.0], [-1.0])  # out of bounds


def test_clip_no_by():
    """Tests clip without 'by' parameter - clips all variables independently."""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    # Sample data has u, v values around 1-5
    clipped = data.piv.clip(min=2.0, max=4.0)
    
    # Check that values are clipped
    assert clipped["u"].min() >= 2.0
    assert clipped["u"].max() <= 4.0
    assert clipped["v"].min() >= 2.0
    assert clipped["v"].max() <= 4.0
    
    # Check attributes are preserved by default
    assert clipped.attrs == data.attrs


def test_clip_by_u():
    """Tests clip by U component - masks locations where U is out of range."""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    
    # Clip based on u component
    clipped = data.piv.clip(min=2.0, max=4.0, by='u')
    
    # Check that locations where u was out of range are now NaN in all variables
    u_out_of_range = (data["u"] < 2.0) | (data["u"] > 4.0)
    
    # Where u was out of range, both u and v should be NaN
    assert np.all(np.isnan(clipped["u"].values[u_out_of_range.values]))
    assert np.all(np.isnan(clipped["v"].values[u_out_of_range.values]))


def test_clip_by_v():
    """Tests clip by V component - masks locations where V is out of range."""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    
    # Clip based on v component
    clipped = data.piv.clip(min=2.0, max=4.0, by='v')
    
    # Check that locations where v was out of range are now NaN in all variables
    v_out_of_range = (data["v"] < 2.0) | (data["v"] > 4.0)
    
    # Where v was out of range, both u and v should be NaN
    assert np.all(np.isnan(clipped["u"].values[v_out_of_range.values]))
    assert np.all(np.isnan(clipped["v"].values[v_out_of_range.values]))


def test_clip_by_magnitude():
    """Tests clip by velocity magnitude - masks locations where magnitude is out of range."""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    
    # Compute magnitude for comparison
    magnitude = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    
    # Clip based on magnitude
    clipped = data.piv.clip(max=6.0, by='magnitude')
    
    # Check that locations where magnitude was > 6.0 are now NaN
    mag_out_of_range = magnitude > 6.0
    
    # Where magnitude was too large, both u and v should be NaN
    assert np.all(np.isnan(clipped["u"].values[mag_out_of_range.values]))
    assert np.all(np.isnan(clipped["v"].values[mag_out_of_range.values]))


def test_clip_by_scalar_property():
    """Tests clip by a computed scalar property (vorticity)."""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    
    # Compute vorticity first
    data = data.piv.vorticity(name='w')
    
    # Clip based on vorticity
    clipped = data.piv.clip(min=-10, max=10, by='w')
    
    # Check that locations where vorticity was out of range are now NaN
    w_out_of_range = (data["w"] < -10) | (data["w"] > 10)
    
    # Where vorticity was out of range, all variables should be NaN
    assert np.all(np.isnan(clipped["u"].values[w_out_of_range.values]))
    assert np.all(np.isnan(clipped["v"].values[w_out_of_range.values]))
    assert np.all(np.isnan(clipped["w"].values[w_out_of_range.values]))


def test_clip_min_only():
    """Tests clip with only min parameter."""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    clipped = data.piv.clip(min=3.0)
    
    # Check that values below min are clipped
    assert clipped["u"].min() >= 3.0
    assert clipped["v"].min() >= 3.0


def test_clip_max_only():
    """Tests clip with only max parameter."""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    clipped = data.piv.clip(max=3.0)
    
    # Check that values above max are clipped
    assert clipped["u"].max() <= 3.0
    assert clipped["v"].max() <= 3.0


def test_clip_error_no_params():
    """Tests that clip raises error when neither min nor max is provided."""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    
    with pytest.raises(ValueError, match="At least one of 'min' or 'max' must be provided"):
        data.piv.clip()


def test_clip_error_invalid_by():
    """Tests that clip raises error when 'by' variable doesn't exist."""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        data.piv.clip(min=0, by='nonexistent')


def test_clip_keep_attrs_false():
    """Tests clip with keep_attrs=False."""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    data.attrs['test_attr'] = 'test_value'
    data['u'].attrs['u_attr'] = 'u_value'
    
    clipped = data.piv.clip(min=2.0, max=4.0, keep_attrs=False)
    
    # Attributes should be removed
    assert len(clipped.attrs) == 0
    assert len(clipped['u'].attrs) == 0
