# test_methods.py
import xarray as xr
from pivpy import io, pivpy
import numpy as np

import os
f1 = 'Run000001.T000.D000.P000.H001.L.vec'
f2 = 'Run000002.T000.D000.P000.H001.L.vec'
path = './data/'

_a = io.loadvec(os.path.join(path,f1))
_b = io.loadvec(os.path.join(path,f2))


def test_crop():
    """ tests crop """
    _c = _a.piv.crop([.6, 19.,-.6,-19.])
    assert _c.u.shape == (59,59,1)
    
    _c = io.create_sample_dataset()
    _c = _c.sel(x = slice(30,90),y=slice(60,80))
    assert _c.u.shape == (2,2,5) # note the last dimension is preserved

def test_pan():
    """ test a shift by dx,dy using pan method """
    _a = io.loadvec(os.path.join(path,f1))
    _c = _a.piv.pan(1.0,-1.0) # note the use of .piv.
    assert np.allclose(_c.coords['x'][0],1.312480)
    assert np.allclose(_c.coords['y'][0], -1.31248)


def test_mean():
    data = io.create_sample_dataset(10)
    assert data.piv.average.u.median() == 4.0

def test_vec2scal():
    data = io.create_sample_dataset()
    data.piv.vec2scal()
    # tests that first data['w'] exists and then 
    # the first value is 0.0
    assert data['w'][0,0,0] == 0.0

def test_add():
    data = io.create_sample_dataset()
    tmp = data + data
    assert tmp['u'][0,0,0] == 2.0

def test_subtract():
    """ tests subtraction """
    data = io.create_sample_dataset()
    tmp = data - data
    assert tmp['u'][0,0,0] == 0.0

def test_multiply():
    """ tests subtraction """
    data = io.create_sample_dataset()
    tmp = data * 3.5
    assert tmp['u'][0,0,0] == 3.5

def test_set_get_dt():
    """ tests setting the new dt """
    data = io.create_sample_dataset()
    assert data.attrs['dt'] == 1.0
    assert data.piv.get_dt == 1.0
    data.piv.set_dt(2.0)
    assert data.attrs['dt'] == 2.0
    
# def test_rotate():
#     """ tests rotation """
#     data = io.create_sample_dataset()
#     data.piv.rotate(90) # rotate by 90 deg
#     assert data['u'][0,0,0] == 2.1 # shall fail
    
def test_vorticity():
    """ tests vorticity estimate """
    
    data = io.create_sample_field()
    data.piv.vorticity()
    assert data['w'][0,0] == 0.0
    
def test_shear():
    """ tests shear estimate """
    
    data = io.create_sample_field()
    data.piv.shear()
    assert data['w'][0,0] == 0.0 
    
def test_vec2scal():
    """ tests vec2scal """
    
    data = io.create_sample_field()
    data.piv.vec2scal(property='curl')
    data.piv.vec2scal(property='ken')
    data.piv.vec2scal(property='tke')
    assert len(data.attrs['variables']) == 5
    assert data.attrs['variables'][-1] == 'tke'
    
    _a.piv.vec2scal(property='curl')

def test_gaussian_smooth():
    """ tests spatial filter """
    
    data = io.create_sample_field()
    data.piv.gaussian_smooth()
    assert data['u'][2,2,2] == 0.0 # shall fail first 
    
def test_spatial_filter():
    """ tests spatial filter """
    
    data = io.create_sample_field()
    data.piv.spatial_filter()
    data.piv.spatial_filter(filter='median')
    data.piv.spatial_filter(filter='median',size=5)
    data.piv.spatial_filter(filter='gaussian',sigma=2)
    _a.piv.spatial_filter(filter='gaussian')
    assert _a['u'][2,2,2] == 0.0 # shall fail first 
    