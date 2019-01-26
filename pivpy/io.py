# -*- coding: utf-8 -*-

"""
Contains functions for reading flow fields in various formats
"""

import numpy as np
import xarray as xr
from glob import glob
import os


def load_directory(directory):
    """ 
    load_directory (path)

    Loads all the .VEC files in the directory into a single
    xarray dataset with variables and units added as attributes

    Input: 
        directory : path to the directory with .vec files

    Output:
        data : xarray DataSet with dimensions: x,y,t and 
               data arrays of u,v,
               attributes of variables and units


    See more: loadvec
    """
    files  = glob(os.path.join(directory,'*.vec'))
    variables, units, rows, cols, dt = parse_header(files[0])
    
    data = []
    for f in files:
        data.append(loadvec(f,rows,cols,variables,units))
           
    
    combined = xr.concat(data, dim='t')
    combined.attrs['variables'] = variables
    combined.attrs['units'] = units
    combined.attrs['dt'] = dt
    return combined


        

def loadvec(filename, rows=None, cols=None, variables=None, units=None, dt=None):
    """
        loadvec(filename,rows=rows,cols=cols)
        Loads the VEC file (TECPLOT format by TSI Inc.), OpenPIV VEC or TXT formats
        Arguments:
            filename : file name, expected to have a header and 5 columns
            rows, cols : number of rows and columns of a vector field,
            if None, None, then parse_header is called to infer the number
            written in the header
        Output:
            data is a xAarray Dataset, see xarray for help 
    """
    if rows is None or cols is None:
        variables,units,rows,cols, dt = parse_header(filename)

    if rows is None: # means no headers
        d = np.loadtxt(filename,usecols=(0,1,2,3,4))
        rows = np.unique(d[:,0])
        cols = np.unique(d[:,1])
        d = d.reshape(rows,cols,5)
    else:
        d = np.loadtxt(filename,skiprows=1,delimiter=',',usecols=(0,1,2,3,4)).reshape(rows,cols,5)

    u = xr.DataArray(d[:,:,2],dims=('x','y'),coords={'x':d[:,:,0][0,:],'y':d[:,:,1][:,0]})
    v = xr.DataArray(d[:,:,3],dims=('x','y'),coords={'x':d[:,:,0][0,:],'y':d[:,:,1][:,0]})
    cnc = xr.DataArray(d[:,:,4],dims=('x','y'),coords={'x':d[:,:,0][0,:],'y':d[:,:,1][:,0]})
    data = xr.Dataset({'u': u, 'v': v,'cnc':cnc})


    data.attrs['variables'] = variables
    data.attrs['units'] = units  
    data.attrs['dt'] = dt
    
    return data
    
    
def parse_header(filename):
    """ 
    parse_header ( filename)
    Parses header of the file (.vec) to get the variables (typically X,Y,U,V)
    and units (can be m,mm, pix/dt or mm/sec, etc.), and the size of the dataset
    by the number of rows and columns.
    Input:
        filename : complete path of the file to read
    Returns:
        variables : list of strings
        units : list of strings
        rows : number of rows of the dataset
        cols : number of columns of the dataset 
        dt   : time interval between the two PIV frames in microseconds
    """
    with open(filename) as fid:
        header = fid.readline()

    # if the file does not have a header, can be from OpenPIV or elsewhere
    # return None 
    if header[:5] != 'TITLE':
        return (None,None,None,None,None)

    header_list = header.replace(',',' ').replace('=',' ').replace('"',' ').split()
    
    # get variable names, typically X,Y,U,V
    variables = header_list[3:12][::2]
    
    # get units - this is important if it's mm or m/s 
    units = header_list[4:12][::2]

    # get the size of the PIV grid in rows x cols 
    rows = int(header_list[-5])
    cols = int(header_list[-3])

    # this is also important to know the time interval, dt
    ind1 = header.find('MicrosecondsPerDeltaT')
    dt = float(header[ind1:].split('"')[1])
    
    
    return (variables, units, rows, cols, dt)
        


def get_units(fname):
    """ 
    get_units(filename)

    given a full path name to the .vec file will return the names 
    of length and velocity units fallback option is all None. Uses
    parse_header function, see below.

    """

    _, units, _, _, _ = parse_header(fname)

    lUnits = units[0]
    velUnits = units[2]

    if velUnits == 'pixel':
        velUnits = velUnits+'/dt'
        
    tUnits = velUnits.split('/')[1]
    
    return lUnits, velUnits, tUnits


#  part extended by Ron Shnapp 24.5.15


    
# def get_data_openpiv_txt(fname,path):
#     """this function gathers and retuens the data found in
#     a single .txt file created by OpenPIV"""


    
#     fname = os.path.join(os.path.abspath(path),fname)
#     if os.path.isfile(fname) and fname.endswith('.txt'): 
#         data = genfromtxt(fname,usecols=(0,1,2,3,4))
#     else:
#         raise ValueError('Wrong file or file extension')

#     return data

	   
     
# def readTimeStamp(fname,path):
#     """reads an insight tstmp file and returns
#     an array of the times at which photos were
#     taken at relative to the begining of
#     aquasition"""
#     fname = os.path.join(os.path.abspath(path),fname)
#     num_lines = sum(1 for line in open(fname))
#     f = open(fname)
#     for i in range(3):
#         f.readline()
#     strt = [f.readline().split()[1] for i in range(num_lines-4)]
#     t = [float(i)/1000000 for i in strt]
#     return t