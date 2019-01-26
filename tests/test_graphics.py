# test_methods.py
import xarray as xr
from pivpy import io, graphics
import numpy as np

import os
f1 = 'Run000001.T000.D000.P000.H001.L.vec'
f2 = 'Run000002.T000.D000.P000.H001.L.vec'
path = './data/'

_a = io.loadvec(os.path.join(path,f1))
_b = io.loadvec(os.path.join(path,f2))


def test_showscal():
    graphics.showscal(_a,bckgr='ken')


