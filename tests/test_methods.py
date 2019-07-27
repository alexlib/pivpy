# test_methods.py
import xarray as xr
from pivpy import io, pivpy
import numpy as np

import os
f1 = 'Run000001.T000.D000.P000.H001.L.vec'
f2 = 'Run000002.T000.D000.P000.H001.L.vec'
path = os.path.join(os.path.dirname(__file__),'data')

_a = io.load_vec(os.path.join(path,f1))
_b = io.load_vec(os.path.join(path,f2))


def test_crop():
    """ tests crop """
    _c = _a.piv.crop([5, 15,-5,-15])
    assert _c.u.shape == (32,32,1)
    
    _c = io.create_sample_dataset()
    _c = _c.sel(x = slice(35,70),y=slice(30,90))
    assert _c.u.shape == (2,2,5) # note the last dimension is preserved

def test_pan():
    """ test a shift by dx,dy using pan method """
    _a = io.load_vec(os.path.join(path,f1))
    _c = _a.piv.pan(1.0,-1.0) # note the use of .piv.
    assert np.allclose(_c.coords['x'][0],1.312480)
    assert np.allclose(_c.coords['y'][0], -1.31248)


def test_mean():
    data = io.create_sample_dataset(10)
    assert np.allclose(data.piv.average.u.median(), 4.5)

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
    assert data.piv.dt == 1.0
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
    data.piv.vec2scal(property='ke')
    assert len(data.attrs['variables']) == 5
    assert data.attrs['variables'][-1] == 'ke'

    data = io.create_sample_dataset()
    data.piv.vec2scal(property='ke')
    data.piv.vec2scal(property='tke') # now defined
    assert data.attrs['variables'][-1] == 'tke'

    
    _a.piv.vec2scal(property='curl')
    