# test_methods.py
import xarray as xr
from pivpy import io, process
import numpy as np

import os
f1 = 'Run000001.T000.D000.P000.H001.L.vec'
f2 = 'Run000002.T000.D000.P000.H001.L.vec'
path = './data/'

_a = io.loadvec(os.path.join(path,f1))
_b = io.loadvec(os.path.join(path,f2))


def test_filterf():
    """ test filterf """
    assert True

def test_averf():
    """ test averf """
    assert True

def test_vec2scal():
    """ tests of vec2scal """
    tmp = process.vec2scal(_a, property='curl')
    print("inside test_vec2scal")
    print(tmp)
    # assert 't' in tmp.dims
    # assert 'w' in tmp.vars






