# test_methods.py
import xarray as xr
from pivpy import io
import numpy as np

import os
f1 = 'Run000001.T000.D000.P000.H001.L.vec'
f2 = 'Run000002.T000.D000.P000.H001.L.vec'
path = './data/'

_a = io.loadvec(os.path.join(path,f1))
_b = io.loadvec(os.path.join(path,f2))


def test_add():
    """ test add of two fields """
    # this operation should sum only velocities
    # xarray provides it out of the box for the variables
    # and not dimensions
    _c = _a + _b 

    assert np.allclose(_c.x,_a.x)
    assert np.allclose(_c.u, _a.u + _b.u)

def test_pan():
    """ test a shift by dx,dy using pan method """
    _c = _a.piv.pan(1.0,-1.0) # note the use of .piv.
    assert np.allclose(_c.coords['x'][0],1.312480)
    assert np.allclose(_c.coords['y'][0], -1.31248)


# def test_mean():
#     from pivpy.pivpy import PIVAccessor
#     data = io.loadvec(os.path.join(path,fname))
#     data.piv.average


