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


def vec2scal(data, property='curl'):
    """ creates a dataset of scalar values on the same 
    dimensions and coordinates as the vector dataset
    Agruments:
        data : xarray.DataSet with u,v on t,x,y grid
    Returns:
        scalar_data : xarray.Dataset w on t,x,y grid
        'w' represents one of the following properties:
            - 'curl' or 'rot' - vorticity

    """
    print("inside vec2scal")
    print(data)

    # prepare space
    d = data.copy(deep=True)
    d = d.assign(variables=r'w')
    d.attrs['variables'] = ['x','y','w']

    print("after assignment")
    print(d)

    for t in d['t']:
        tmp = d.isel(t=t)
        if property is 'curl':
            #    estimate curl
            ux,_ = np.gradient(tmp['u'])
            _,vy = np.gradient(tmp['v'])
            tmp['w'] += vy - ux
        elif property is 'ken':
            tmp['w'] = tmp['u']**2 + tmp['v']**2
    
    d = d.drop(['u','v','cnc'])
    return d
