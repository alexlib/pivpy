import xarray as xr
from pivpy import io, pivpy, graphics
import numpy as np

import os
f1 = 'Run000001.T000.D000.P000.H001.L.vec'
path = os.path.join(os.path.dirname(__file__),'data')


_d = io.load_vec(os.path.join(path,f1))

def test_showscal():
    graphics.showscal(_d, property='ke')
    

def test_quiver():
    graphics.quiver(_d)


def test_xarray_plot():
    _d.piv.vec2scal(property='curl')
    _d['w'].isel(t=0).plot.pcolormesh()

def test_histogram():
    graphics.histogram(_d)

    
    
    
    
    


