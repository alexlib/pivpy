# -*- coding: utf-8 -*-

"""
Contains functions for reading flow fields in various formats
"""

import numpy as np
import xarray as xr
from glob import glob
import os.path


def loadvec_dir(directory):
    """ loadvec_dir (directory)
    """
    files  = glob(os.path.join(directory,'*.vec'))
    variables, units, rows, cols = parse_header(files[0])
    
    data = []
    for f in files:
        data.append(loadvec(f,rows,cols))
    
    return (data, variables, units) 

def parse_header(filename):
    """ parse_header (filename) """
    with open(filename) as fid:
        header = fid.readline()

    header_list = header.replace(',',' ').replace('=',' ').replace('"',' ').split()
    
    variables = header_list[3:12][::2]
    units = header_list[4:12][::2]
    rows = int(header_list[-5])
    cols = int(header_list[-3])
    
    
    return (variables, units, rows, cols)
    
        

def loadvec(filename,rows=None, cols=None):
    """
        loadvec(filename,rows=rows,cols=cols)
        Loads the VEC file (TECPLOT format by TSI Inc.) and OpenPIV format
        Arguments:
            filename : file name, expected to have a header and 5 columns
            rows, cols : number of rows and columns of a vector field,
            if None, None, then parse_header is called to infer the number
            written in the header
        Output:
            data is a xAarray Dataset, see xarray for help 
    """
    if rows is None or cols is None:
        _,_,rows,cols = parse_header(filename)
    d = np.loadtxt(filename,skiprows=1,delimiter=',',usecols=(0,1,2,3,4)).reshape(rows,cols,5)
    
    u = xr.DataArray(d[:,:,2],dims=('x','y'),coords={'x':d[:,:,0][0,:],'y':d[:,:,1][:,0]})
    v = xr.DataArray(d[:,:,3],dims=('x','y'),coords={'x':d[:,:,0][0,:],'y':d[:,:,1][:,0]})
    cnc = xr.DataArray(d[:,:,4],dims=('x','y'),coords={'x':d[:,:,0][0,:],'y':d[:,:,1][:,0]})
    data = xr.Dataset({'u': u, 'v': v,'cnc':cnc})  
    
    return data
    
    
    


