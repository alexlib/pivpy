""" tests pivpy.pivpy methods """
import pathlib
import numpy as np
import importlib.resources
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


def test_clip_no_by():
    """tests clip without 'by' parameter - clips all variables independently"""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    # Sample data has u,v values around 1-5
    clipped = data.piv.clip(min=2.0, max=4.0)
    
    # Check that values are clipped
    assert clipped["u"].min() >= 2.0
    assert clipped["u"].max() <= 4.0
    assert clipped["v"].min() >= 2.0
    assert clipped["v"].max() <= 4.0
    
    # Check attributes are preserved by default
    assert clipped.attrs == data.attrs


def test_clip_by_u():
    """tests clip by U component - masks locations where U is out of range"""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    
    # Clip based on u component
    clipped = data.piv.clip(min=2.0, max=4.0, by='u')
    
    # Check that locations where u was out of range are now NaN in all variables
    u_out_of_range = (data["u"] < 2.0) | (data["u"] > 4.0)
    
    # Where u was out of range, both u and v should be NaN
    assert np.all(np.isnan(clipped["u"].values[u_out_of_range.values]))
    assert np.all(np.isnan(clipped["v"].values[u_out_of_range.values]))


def test_clip_by_v():
    """tests clip by V component - masks locations where V is out of range"""
    data = io.create_sample_Dataset(n_frames=2, rows=5, cols=5)
    
    # Clip based on v component
    clipped = data.piv.clip(min=2.0, max=4.0, by='v')
    
    # Check that locations where v was out of range are now NaN in all variables
    v_out_of_range = (data["v"] < 2.0) | (data["v"] > 4.0)
    
    # Where v was out of range, both u and v should be NaN
    assert np.all(np.isnan(clipped["u"].values[v_out_of_range.values]))
    assert np.all(np.isnan(clipped["v"].values[v_out_of_range.values]))


def test_clip_by_magnitude():
    """tests clip by velocity magnitude - masks locations where magnitude is out of range"""
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
    """tests clip by a computed scalar property (vorticity)"""
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
    """tests clip with only min parameter"""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    clipped = data.piv.clip(min=3.0)
    
    # Check that values below min are clipped
    assert clipped["u"].min() >= 3.0
    assert clipped["v"].min() >= 3.0


def test_clip_max_only():
    """tests clip with only max parameter"""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    clipped = data.piv.clip(max=3.0)
    
    # Check that values above max are clipped
    assert clipped["u"].max() <= 3.0
    assert clipped["v"].max() <= 3.0


def test_clip_error_no_params():
    """tests that clip raises error when neither min nor max is provided"""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    
    with pytest.raises(ValueError, match="At least one of 'min' or 'max' must be provided"):
        data.piv.clip()


def test_clip_error_invalid_by():
    """tests that clip raises error when 'by' variable doesn't exist"""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        data.piv.clip(min=0, by='nonexistent')


def test_clip_keep_attrs_false():
    """tests clip with keep_attrs=False"""
    data = io.create_sample_Dataset(n_frames=1, rows=5, cols=5)
    data.attrs['test_attr'] = 'test_value'
    data['u'].attrs['u_attr'] = 'u_value'
    
    clipped = data.piv.clip(min=2.0, max=4.0, keep_attrs=False)
    
    # Attributes should be removed
    assert len(clipped.attrs) == 0
    assert len(clipped['u'].attrs) == 0
