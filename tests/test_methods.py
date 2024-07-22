""" tests pivpy.pivpy methods """
import pathlib
import numpy as np
import importlib.resources
import pytest
from pivpy import io


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
