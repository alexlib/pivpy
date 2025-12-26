"""
Defines input/output functions of PIVPy

see read_pivpy()
"""

from pathlib import Path
import pathlib
import os
import glob
import numpy as np
import xarray as xr

from .graphics import display_vector_field
from .pivpy_class import PIVData


def create_path(path):
    """create path if it doesn't exist
    
    Input:
        path - path to be created
    
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def get_dt(path, sample_file=0):
    """gets the time interval between two frames from 
    the attributes or list of files

    Args:
        path ([type]): [description]
        sample_file (int, optional): [description]. Defaults to 0.
    """
    pass


def get_frame_rate(directory, pattern="*.tif"):
    """uses _FrameRate:_* attribute in the multi-page TIF files to 
    to estimate the time stamp of each frame (or image) and also 
    dt, the duration between two consecutive frames 
    
    Input:
        directory : path to the multi-page TIF files
        pattern : 
    """
    pass


def read_directory(directory, basename='*', ext='*.txt'):
    """ This function imports all the files in a folder,
    reads those that match a pattern and returns the list
    of arrays, one per file, in the same order as read from
    glob.glob
    
    Args:
        directory :   string, path to the folder with files, absolute
                    or relative
        basename :  string, optional
                    a part of name on which we can split the filename, 
                    default is ''
        ext :   string, optional
                    the file extension,
                    default is '*.txt'
        
    Returns:
        list of xarray.Dataset (PIV fields from all the files)
    
    Example:
    
        >>> data = pivpy.io.read_directory('../test3', basename='*', ext='*.txt')
        >>> data.piv[0]
        
    
    """
    
    # Read all files in folder (absolute path):
    files = [f for f in glob.glob(os.path.join(directory, basename + ext))]
    files.sort()  # sort alphabetically
    
    # create a path
    create_path(os.path.join(directory,'figures'))
    
    # run through all files in the folder and read data from file:
    data = []
    for i, f in enumerate(files):
        # print(f)
        d = loadvec(f)
        d.attrs['filename'] = os.path.basename(f)
        d.attrs['path'] = directory
        data.append(d)
        
    return data


def save_vec(fname=None, data=None):
    """ saves a single velocity file to the Insight (TSI Inc.) VEC format
    

    Parameters
    ----------
    fname : string
        filename to save to
    data : xarray.Dataset
        PIV data array with x,y,u,v,chc (optional)
    
    
    
    Example:
    
        >>> data = pivpy.io.loadvec('sample.vec')
        >>> pivpy.io.save_vec('test.vec', data)
        
    """
    from numpy import meshgrid, c_, savetxt, vstack
    
    # prepare header
    
    headerlines = []
    headerlines.append('TITLE = ""')
    headerlines.append('VARIABLES = "x", "y", "u", "v", "chc"')
    headerlines.append(f'ZONE T="zone1", I= {data.sizes["x"]}, J= {data.sizes["y"]}, K=1, F=POINT')
    
    # create mesh
    X, Y = meshgrid(data.x, data.y)
    
    # write data
    with open(fname, 'w') as f:
        for line in headerlines:
            f.write(line + '\n')
        for i in range(data.sizes['y']):
            for j in range(data.sizes['x']):
                f.write(f"{X[i,j]:12.6f} {Y[i,j]:12.6f} {data.u.values[i,j]:12.6f} {data.v.values[i,j]:12.6f}")
                if 'chc' in data:
                    f.write(f" {data.chc.values[i,j]:12.6f}")
                f.write('\n')


def loadvec(filename, rows=None, cols=None):
    """ Reads the VEC file (TSI Inc, Insight),
    returns xarray DataArray with dimensions: x,y 
    and variables u,v and chc (if present)


    Parameters
    ----------
    filename : string
        a file name or a path+filename

    Returns:
    --------
    data : xarray.Dataset with ['u','v','chc']
    
    
    Examples:
    ---------
    
        >>> data = pivpy.io.loadvec('../examples/Insight/EXP_001.vec')
    
    """

    with open(filename) as f:
        # strip first 3 header rows:
        l1 = f.readline()
        l2 = f.readline()
        # get dimensions from the third line:
        l3 = f.readline()
        tmp = l3.replace(',', ' ').split()
        # print(tmp)
        # by default, row and col indices are in the 3-4-th places
        if rows is None:
            rows = int(tmp[3])
        if cols is None:
            cols = int(tmp[5])
        # read the rest of the data:
        data = np.loadtxt(f)

    # if the file is shorter than rows*cols, we need to pad it
    # with nan
    if len(data) < rows * cols:
        tmp = np.empty((rows * cols, data.shape[1]))
        tmp[:] = np.nan
        tmp[: len(data), :] = data
        data = tmp

    # x coordinate
    x = data[::rows, 0]
    # y coordinate is the first column, rows values:
    y = data[:rows, 1]

    # if there are more than 4 columns, it's 5
    if data.shape[1] > 4:
        tmp = data[:, [2, 3, 4]].T.reshape(3, cols, rows)
        u, v, chc = tmp
    else:  # otherwise it's 4 columns
        tmp = data[:, [2, 3]].T.reshape(2, cols, rows)
        u, v = tmp

    # remove nans at the edges
    u = np.squeeze(u)
    v = np.squeeze(v)

    # create data arrays:
    u = xr.DataArray(u.T, dims=["y", "x"], coords={"x": x, "y": y}, name="u")
    v = xr.DataArray(v.T, dims=["y", "x"], coords={"x": x, "y": y}, name="v")

    # merge into dataset:
    data = xr.merge([u, v])

    if chc is not None:
        chc = np.squeeze(chc)
        chc = xr.DataArray(chc.T, dims=["y", "x"], coords={"x": x, "y": y}, name="chc")
        data = xr.merge([data, chc])

    # add attributes:
    data.attrs["filename"] = filename
    data.attrs["rows"] = rows
    data.attrs["cols"] = cols

    return data


def load_directory(directory, basename='*', ext='*.vec'):
    """ This function imports all the files in a folder,
    reads those that match a pattern and returns the list
    of arrays, one per file, in the same order as read from
    glob.glob
    
    Args:
        directory :   string, path to the folder with files, absolute
                    or relative
        basename :  string, optional
                    a part of name on which we can split the filename, 
                    default is ''
        ext :   string, optional
                    the file extension,
                    default is '*.vec'
        
    Returns:
        list of xarray.Dataset (PIV fields from all the files)
    
    Example:
    
        >>> data = pivpy.io.load_directory('../test3', basename='*', ext='*.vec')
        >>> data.piv[0]
        
    
    """
    
    # Read all files in folder (absolute path):
    files = [f for f in glob.glob(os.path.join(directory, basename + ext))]
    files.sort()  # sort alphabetically
    
    # create a path
    create_path(os.path.join(directory,'figures'))
    
    # run through all files in the folder and read data from file:
    data = []
    for i, f in enumerate(files):
        # print(f)
        d = loadvec(f)
        d.attrs['filename'] = os.path.basename(f)
        d.attrs['path'] = directory
        data.append(d)
        
    return data


def loadopenpiv(filename):
    """ reads a file written in the format of:
    
        np.savetxt(fname,np.vstack([x.T,y.T,u.T,v.T,mask.T]).T)
    
    
    Parameters
    ----------
    filename : string
        a filename of the PIV fields file, written by np.savetxt() with 
        x,y,u,v,mask (optional) in columns
    
    Returns
    -------
    data : xarray.Dataset with ['u','v','mask']
    
    
    See also:
    ---------
    
        >>> pivpy.io.create_sample_field()
    
    """
    col = np.loadtxt(filename).T
    # import ipdb; ipdb.set_trace()
    nrows = len(np.unique(col[1, :]))
    if nrows > 1:
        ncols = len(np.unique(col[0, :]))
        x = col[0, :].reshape(nrows, ncols)
        y = col[1, :].reshape(nrows, ncols)
        u = col[2, :].reshape(nrows, ncols)
        v = col[3, :].reshape(nrows, ncols)
        x = x[0, :]
        y = y[:, 0]
        if col.shape[0] > 4:  # if there's a mask
            chc = col[4, :].reshape(nrows, ncols)
    else:  # single row
        x = col[0, :]
        y = col[1, :]
        u = col[2, :]
        v = col[3, :]
        if col.shape[0] > 4:  # if there's a mask
            chc = col[4, :]

    u = xr.DataArray(u, dims=["y", "x"], coords={"x": x, "y": y}, name="u")
    v = xr.DataArray(v, dims=["y", "x"], coords={"x": x, "y": y}, name="v")

    data = xr.merge([u, v])

    if col.shape[0] > 4:  # if there's a mask
        chc = xr.DataArray(chc, dims=["y", "x"], coords={"x": x, "y": y}, name="chc")
        data = xr.merge([data, chc])

    data.attrs["filename"] = filename
    data.attrs["source"] = "openpiv"

    return data


def create_sample_field(rows=5, cols=5):
    """ creates a sample velocity field for testing
    purposes

    Input:
        rows : integer, default = 5 number of rows
        cols : integer, default = 5 number of columns
    
    """
    y, x = np.meshgrid(range(rows), range(cols))
    u = np.ones((rows, cols)) * 2.0
    v = np.ones((rows, cols)) * (-1.0)

    u = xr.DataArray(
        u,
        dims=["y", "x"],
        coords={"x": np.arange(cols), "y": np.arange(rows)},
        name="u",
    )
    v = xr.DataArray(
        v,
        dims=["y", "x"],
        coords={"x": np.arange(cols), "y": np.arange(rows)},
        name="v",
    )

    return xr.merge([u, v])


def create_sample_dataset(rows=5, cols=5, num_files=1):
    """ creates a sample data set with uniform velocity fields 
    """
    
    data = []
    for i in range(num_files):
        d = create_sample_field(rows=rows,cols=cols)
        d.attrs['filename'] = f'test_{i:03d}.vec'
        data.append(d)
    
    return data


def save_to_list_of_files(data, path='.'):
    """ saves data from the list of datasets to files 
    data : list
        list of xarray.Dataset 
    path : string
        path to save, deafault is the present directory
    """
    
    for d in data:
        fname = os.path.join(path,d.attrs['filename'])
        save_vec(fname, d)
        print(f'File {fname} is written')
