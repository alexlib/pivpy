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
    """
    "vf" in the name of the function stands for VortexFitting package.
    The function takes PIVPY data base object and converts it to an object of class
    VelocityField from VortexFitting package. We are interested only in the velocity 
    field portion of the PIVPY database because this is what VortexFitting's 
    VelocityField class is concerned about.
    There is no way to to convert PIVPY database to vortexfitting.VelocityField
    without creating a file because the later wants a file to read from.
    I wanted to create a hidden file, but that's a pain if we try doing it
    cross-platform. For now, I'm just going to create a normal file and leave
    it to the user to decide what to do with it when the analysis is done. A user
    must supply the name of the file.
    Parameters: field (xarray.DataSet) - PIVPY ofject - an Xarray data set - that contains
                                         velocity field; it may contain other para-
                                         meters (e.g. vorticity), but it doesn't 
                                         matter; it must contain only one time frame.
                ncFilePath (pathlib.Path) - a path to the .nc file that will store
                                            the velocity field that will be fed to 
                                            VortexFitting; an example of the file name
                                            is PIVpair1VelocityField.nc; note the file
                                            doesn't have to exist - the function just
                                            needs a name for the file
    Returns: vfield (vortexfitting.VelocityField) - object of the VortexFitting
                                                    package class VelocityField
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