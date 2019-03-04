# -*- coding: utf-8 -*-

"""
Contains functions for reading flow fields in various formats
"""

import numpy as np
import xarray as xr
from glob import glob
import os, re




def create_sample_field(frame = 0):
    """ creates a sample dataset for the tests """

    x  = np.arange(32.,128.,32.)
    y = np.arange(16.,128.,16.)

    xm,ym = np.meshgrid(x,y)

    u = np.ones_like(xm.T) + np.arange(0.0,7.0)
    v = np.zeros_like(ym.T)+np.random.rand(3,1)-.5

    u = u[:,:,np.newaxis]
    v = v[:,:,np.newaxis]
    chc = np.ones_like(u)

    # plt.quiver(xm.T,ym.T,u,v)



    u = xr.DataArray(u,dims=('x','y','t'),coords={'x':x,'y':y,'t':[frame]})
    v = xr.DataArray(v,dims=('x','y','t'),coords={'x':x,'y':y,'t':[frame]})
    chc = xr.DataArray(chc,dims=('x','y','t'),coords={'x':x,'y':y,'t':[frame]})
    
    data = xr.Dataset({'u': u, 'v': v,'chc':chc})

    data.attrs['variables'] = ['x','y','u','v']
    data.attrs['units'] = ['pix','pix','pix/dt','pix/dt']  
    data.attrs['dt'] = 1.0

    return data


def create_sample_dataset(n = 5):
    """ using create_sample_field that has random part in it, create
    a sample dataset of length 'n' """


    data = []
    for i in range(n):
        data.append(create_sample_field(frame=i))
           
    
    combined = xr.concat(data, dim='t')
    combined.attrs['variables'] = ['x','y','u','v']
    combined.attrs['units'] = ['pix','pix','pix/dt','pix/dt']  
    combined.attrs['dt'] = 1.0

    return combined
        

def loadvec(filename, rows=None, cols=None, variables=None, units=None, dt=None, frame=0):
    """
        loadvec(filename,rows=rows,cols=cols)
        Loads the VEC file (TECPLOT format by TSI Inc.), OpenPIV VEC or TXT formats
        Arguments:
            filename : file name, expected to have a header and 5 columns
            rows, cols : number of rows and columns of a vector field,
            if None, None, then parse_header is called to infer the number
            written in the header
            dt : time interval (default is None)
            frame : frame or time marker (default is None)
        Output:
            data is a xAarray Dataset, see xarray for help 
    """
    if rows is None or cols is None:
        variables,units,rows,cols, dt, frame = parse_header(filename)

    if rows is None: # means no headers
        d = np.loadtxt(filename,usecols=(0,1,2,3,4))
        x = np.unique(d[:,0])
        y = np.unique(d[:,1])
        d = d.reshape(len(y),len(x),5).transpose(1,0,2)
    else:
        d = np.loadtxt(filename,skiprows=1,delimiter=',',usecols=(0,1,2,3,4)).reshape(rows,cols,5)
        x = d[:,:,0][0,:]
        y = d[:,:,1][:,0]

    u = xr.DataArray(d[:,:,2],dims=('x','y'),coords={'x':x,'y':y})
    v = xr.DataArray(d[:,:,3],dims=('x','y'),coords={'x':x,'y':y})
    cnc = xr.DataArray(d[:,:,4],dims=('x','y'),coords={'x':x,'y':y})
    
    data = xr.Dataset({'u': u, 'v': v,'cnc':cnc}).expand_dims(dim='t')
    # data = data.assign_coords(t = frame)

    data.attrs['variables'] = variables
    data.attrs['units'] = units  
    data.attrs['dt'] = dt
    
    return data

def load_directory(path,basename=''):
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
    files  = glob(os.path.join(path,basename+'*.vec'))
    variables, units, rows, cols, dt, frame = parse_header(files[0])
    
    data = []
    for i,f in enumerate(files):
        data.append(loadvec(f,rows,cols,variables,units,frame+i))
           
    combined = xr.concat(data, dim='t')
    combined.attrs['variables'] = variables
    combined.attrs['units'] = units
    combined.attrs['dt'] = dt
    return combined

    
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

    # split path from the filename
    fname = os.path.basename(filename)
    # get the number in a filename if it's a .vec file from Insight
    if '.' in fname[:-4]: # day2a005003.T000.D000.P003.H001.L.vec
        frame = int(re.findall('\d+',fname.split('.')[0])[-1])
    elif '_' in filename[:-4]:
        frame = int(re.findall('\d+',fname.split('_')[1])[-1]) # exp1_001_b.vec, .txt

    print(fname, frame)

    with open(filename) as fid:
        header = fid.readline()
    
    # if the file does not have a header, can be from OpenPIV or elsewhere
    # return None 
    if header[:5] != 'TITLE':
        return (None,None,None,None,None,frame)

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
    
    
    return (variables, units, rows, cols, dt, frame)
        


def get_units(filename):
    """ 
    get_units(filename)

    given a full path name to the .vec file will return the names 
    of length and velocity units fallback option is all None. Uses
    parse_header function, see below.

    """

    lUnits, velUnits, tUnits = None, None, None

    _, units, _, _, _, _ = parse_header(filename)

    if units is None:
        return lUnits, velUnits, tUnits

    lUnits = units[0]
    velUnits = units[2]

    if velUnits == 'pixel':
        velUnits = velUnits+'/dt' # make it similar to m/s

    tUnits = velUnits.split('/')[1] # make it 's' or 'dt'
    
    return lUnits, velUnits, tUnits