""" Processing PIV flow fields """
import xarray as xr
import numpy as np

def averf(data):
    """ Ensemble average """
    return data.mean(dim='t')


def filterf(data):
    """Gaussian filtering of velocity """
    from scipy.ndimage.filters import gaussian_filter as gf
    data['u'] = xr.DataArray(gf(data['u'],1),dims=('x','y'))
    data['v'] = xr.DataArray(gf(data['v'],1),dims=('x','y'))
    return data


def vec2scal(data):
    data['w'] = 
