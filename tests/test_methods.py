# test_methods.py
from pivpy import io, pivpy
import numpy as np
import pkg_resources as pkg
import pytest

import os

f1 = "Run000001.T000.D000.P000.H001.L.vec"
f2 = "Run000002.T000.D000.P000.H001.L.vec"
path = pkg.resource_filename("pivpy", "data/Insight")

_a = io.load_vec(os.path.join(path, f1))
_b = io.load_vec(os.path.join(path, f2))


def test_crop():
    """ tests crop """
    _c = _a.piv.crop([5, 15, -5, -15])
    assert _c.u.shape == (32, 32, 1)

    _c = io.create_sample_dataset()
    _c = _c.sel(x=slice(35, 70), y=slice(30, 90))
    assert _c.u.shape == (4, 1, 5)  # note the last dimension is preserved



def test_pan():
    """ test a shift by dx,dy using pan method """
    _a = io.load_vec(os.path.join(path, f1))
    _c = _a.piv.pan(1.0, -1.0)  # note the use of .piv.
    assert np.allclose(_c.coords["x"][0], 1.312480)
    assert np.allclose(_c.coords["y"][0], -1.31248)


def test_mean():
    data = io.create_sample_dataset(10)
    print(data.piv.average.u.median())
    assert np.allclose(data.piv.average.u.median(), 6.0)


def test_vec2scal():
    data = io.create_sample_dataset()
    data.piv.vec2scal()
    data.piv.vec2scal(property="curl")
    data.piv.vec2scal(property="ke")
    assert len(data.attrs["variables"]) == 5
    assert data.attrs["variables"][-1] == "ke"


def test_add():
    data = io.create_sample_dataset()
    tmp = data + data
    assert tmp["u"][0, 0, 0] == 2.0


def test_subtract():
    """ tests subtraction """
    data = io.create_sample_dataset()
    tmp = data - data
    assert tmp["u"][0, 0, 0] == 0.0


def test_multiply():
    """ tests subtraction """
    data = io.create_sample_dataset()
    tmp = data * 3.5
    assert tmp["u"][0, 0, 0] == 3.5


def test_set_get_dt():
    """ tests setting the new dt """
    data = io.create_sample_dataset()
    assert data.attrs["dt"] == 1.0
    assert data.piv.dt == 1.0
    data.piv.set_dt(2.0)
    assert data.attrs["dt"] == 2.0


# def test_rotate():
#     """ tests rotation """
#     data = io.create_sample_dataset()
#     data.piv.rotate(90) # rotate by 90 deg
#     assert data['u'][0,0,0] == 2.1 # shall fail

def test_fluctuations():
    data = io.create_sample_field()
    with pytest.raises(ValueError):
        data.piv.fluct()
    
    data = io.create_sample_dataset(100) # enough for random
    fluct = data.piv.fluct()
    assert np.allclose(fluct['u'].mean(dim='t'), 0.0)
    assert np.allclose(fluct['v'].mean(dim='t'), 0.0)


def test_reynolds_stress():
    data = io.create_sample_dataset(2, noise_sigma=0.0)
    data.isel(t=1)['u'] += 0.1
    data.isel(t=1)['v'] -= 0.1
    # data['u'] += np.random.randn(*data.u.shape)
    tmp = data.piv.reynolds_stress()
    # print(tmp['w'].mean(dim=('x','y')))
    assert np.allclose(tmp['w'],-0.0025)

def test_set_scale():
    data = io.create_sample_dataset()
    tmp = data.piv.set_scale(1.)
    assert np.allclose(tmp['x'], data['x'])

    tmp = data.copy()
    tmp.piv.set_scale(2.)
    assert np.allclose(tmp['u'].mean(dim=('t','x','y')).values/data['u'].mean(dim=('t','x','y')).values, 2.0)

    # assert np.allclose(tmp['u'].mean(dim=('t','x','y'))/data['u'].mean(dim=('t','x','y')), 2.0)



def test_vorticity():
    """ tests vorticity estimate """
    data = io.create_sample_field() # we need another flow field
    data.piv.vorticity()
    assert np.allclose(data['w'], 0.0)


def test_strain():
    """ tests shear estimate """
    data = io.create_sample_field(noise_sigma=0.)
    data.piv.strain()
    assert np.allclose(data['w'], 0.00223713)


def test_tke():
    """ tests TKE """
    data = io.create_sample_dataset()
    data.piv.vec2scal(property="ke")
    data.piv.vec2scal(property="tke")  # now defined
    assert data.attrs["variables"][-1] == "tke"


def test_curl():
    """ tests curl that is also vorticity """
    _a = io.load_vec(os.path.join(path, f1))
    _a.piv.vec2scal(property="curl")
    assert _a.attrs["variables"][-1] == "vorticity"
