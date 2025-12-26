"""
"inter" stands for "intefacing".
This module provides a function that allows to go convert PIVPY datasets 
to the VortexFitting datasets. 
Here is the link to the VortexFitting article:
https://www.sciencedirect.com/science/article/pii/S2352711020303174?via%3Dihub .
"""

import warnings
import numpy as np

try:
    import vortexfitting.classes as vf
except ImportError:
    warnings.warn("VortexFitting is not installed, use pip install vortexfitting")


def pivpyTOvf(field, ncFilePath):
    """Convert a PIVPy xarray Dataset into a VortexFitting ``VelocityField``.

    Notes
    -----
    VortexFitting's ``VelocityField`` expects to read from a NetCDF file, so this
    function writes a temporary NetCDF file at ``ncFilePath`` and then loads it.

    Parameters
    ----------
    field:
        PIVPy xarray Dataset containing at least ``u`` and ``v`` variables and a
        single time frame.
    ncFilePath:
        Path to the NetCDF file to write (it does not need to exist yet).

    Returns
    -------
    vortexfitting.VelocityField
        The VortexFitting velocity field loaded from the written NetCDF file.
    """

 

    # VortexFitting expects the physical system of corrdinates, but field - being obtained
    # from an OpenPIV .txt file is in the image system of coordinates. So, we have to invert
    # they y axis. The procedure that after a lot of trials and errors ended up working is
    # copied from here https://stackoverflow.com/a/70695479/10073233 and is given by:
    field = field.reindex(y = field.y[::-1]) 

    # VortexFitting expects time coordinate to go first. In practice it reads spatial
    # matrices with x as the first spatial axis for piv_netcdf, so we store as (t, x, y)
    # to match the expectations in tests/test_inter.py.
    fieldReordered = field.transpose('t','x','y') 
    fieldReordered = fieldReordered.fillna(0.0)

    # VortexFitting expects very specific names of the data arrays. And there must be
    # the third component of velocity vector.
    fieldReordered['velocity_z'] = fieldReordered['u'].copy(
        data=np.zeros(fieldReordered['u'].values.shape))
    fieldRenamed = fieldReordered.rename_vars(
        {'u':'velocity_n', 'v':'velocity_s', 'y':'grid_z', 'x':'grid_n', })
    
    fieldRenamed.to_netcdf(path=ncFilePath, mode='w')
    
    vfield = vf.VelocityField(str(ncFilePath), file_type = 'piv_netcdf', time_step=0)
    
    return vfield