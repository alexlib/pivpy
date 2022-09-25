# -*- coding: utf-8 -*-

"""
Contains functions for reading flow fields in various formats
"""

from multiprocessing.dummy import Array
from socketserver import DatagramRequestHandler
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import os
import pathlib
import re
import warnings
from typing import List, Tuple, Dict, Any
from numpy.typing import ArrayLike

import warnings

try:
    from lvreader import read_buffer
except: 
    warnings.warn('lvreader is not installed')


# Defaults
POS_UNITS: str = "pix" # or mm, m, after scaling
TIME_UNITS: str = "frame" # "frame" if not scaled, can become 'sec' or 'msec', 'usec'
# after scaling can be m/s, mm/s
VEL_UNITS: str =  POS_UNITS # default is displacement in pix
DELTA_T: np.float64 = 0.0 # default is 0. i.e. uknown, can be any float value

def set_default_attrs(dataset: xr.Dataset)-> xr.Dataset:
    """ Defines default attributes:

    # xr.DataSet.x.attrs["units"] = POS_UNITS
    POS_UNITS: str = "pix" # or mm, m, after scaling


    # None if not known, can become 'sec' or 'msec', 'usec'
    # the time units are for the sequence of PIV realizations
    # useful for animations of the DataSet and averaging
    # xr.DataSet.t.attrs["units"] = TIME_UNITS
    TIME_UNITS: str = None 

    # after scaling can be m/s, mm/s, default is POS_UNITS
    # xr.DataSet.u.attrs["units"] = VEL_UNITS
    VEL_UNITS: str =  POS_UNITS

    # attribute of the xr.DataSet, defines things for the 
    # single flow realization, frame A -> DELTA_T -> frame B
    # xr.DataSet.attrs["delta_t"] = DELTA_T 
    DELTA_T: float = None # default is unknown, can be
    
    """

    dataset.x.attrs["units"] = POS_UNITS
    dataset.y.attrs["units"] = POS_UNITS
    dataset.u.attrs["units"] = VEL_UNITS
    dataset.v.attrs["units"] = VEL_UNITS
    dataset.t.attrs["units"] = TIME_UNITS
    dataset.attrs["delta_t"] = DELTA_T
    dataset.attrs["files"] = []

    return dataset

def create_sample_field(
        rows: int=5,
        cols: int=8,
        grid: List=[32,16],
        frame: int=0, 
        noise_sigma: float=1.0
        ) -> xr.Dataset:
    """ 
    creates a sample Dataset for the tests 

    rows - number of points along vertical coordinate, 
           corresponds to 'y'
    cols - number of grid points along horizontal coordinate, 'x'
    grid - spacing between vectors in two directions (x,y)
    frame - frame number
    noise_sigma - strength of Gaussian noise to add

    Returns:
        xarray.Dataset()

Usage:
    io.create_sample_field(
            rows=3,
            cols=6,
            grid=[32,16],
            frame=0,
            noise_sigma=0.1
        )

"""

    x = np.arange(grid[0], (cols + 1) * grid[0], grid[0])
    y = np.arange(grid[1], (rows + 1) * grid[1], grid[1])

    xm, ym = np.meshgrid(x, y)
    u = np.ones_like(xm) + np.linspace(0.0, 10.0, cols)
    v = (
        np.zeros_like(ym)
        + np.linspace(-1.0, 1.0, rows).reshape(rows, 1)
        + noise_sigma * np.random.randn(rows, 1)
    )

    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = np.ones_like(u)

    u = xr.DataArray(
        u, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    v = xr.DataArray(
        v, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    chc = xr.DataArray(
        chc, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})
    dataset = set_default_attrs(dataset)

    return dataset


def create_sample_Dataset(
        n: int=5, 
        noise_sigma: float=1.0
        )-> xr.Dataset:
    """ using create_sample_field that has random part in it, create
    a sample Dataset of length 'n' """

    dataset = []
    for i in range(n):
        dataset.append(create_sample_field(frame=i, noise_sigma=noise_sigma))

    combined = xr.concat(dataset, dim="t")
    combined = set_default_attrs(combined)

    return combined

def create_uniform_strain():
    return create_sample_field(noise_sigma=0.0)

    
def from_arrays( x: ArrayLike, 
                 y: ArrayLike, 
                 u: ArrayLike, 
                 v: ArrayLike, 
                 mask: np.array,
                 frame: int=0):
    """
        from_arrays(x,y,u,v,mask,frame=0)
        creates an xArray Dataset from 5 two-dimensional Numpy arrays
        of x,y,u,v and mask

        Input:
            x,y,u,v,mask = Numpy floating arrays, all the same size
        Output:
            dataset is a xAarray Dataset, see xarray for help
    """
    # create dataset structure of appropriate size
    dataset = create_sample_field(rows=x.shape[0], cols=x.shape[1],frame=frame)
    # assign arrays
    dataset["x"] = x[0, :]
    dataset["y"] = y[:, 0]
    dataset["u"] = xr.DataArray(u[:, :, np.newaxis], dims=("y", "x", "t"))
    dataset["v"] = xr.DataArray(v[:, :, np.newaxis], dims=("y", "x", "t"))
    dataset["chc"] = xr.DataArray(mask[:, :, np.newaxis], dims=("y", "x", "t"))
    dataset = set_default_attrs(dataset)

    return dataset

def from_df(
    df: pd.DataFrame, 
    frame: int=0, 
    dt: float=None, 
    filename: str=None):
    """
        from_df(x,y,u,v,mask,frame=0)
        creates an xArray Dataset from pandas datasetframe with 5 columns

        Read the .txt files faster with pandas read_csv()

        %%time
        df = pd.read_csv(files_list[-1],delimiter='\t', 
                        names = ['x','y','u','v','mask'],header=0)
        from_df(df,filename=files_list[-1])

        is 3 times faster than the load_txt


        Input:
            x,y,u,v,mask = Numpy floating arrays, all the same size
        Output:
            dataset is a xAarray Dataset, see xarray for help
    """
    d = df.to_numpy()

    x = np.sorted_unique(d[:, 0])
    y = np.sorted_unique(d[:, 1])
    d = d.reshape(len(y), len(x), 5)  # .transpose(1, 0, 2)

    u = d[:, :, 2]
    v = d[:, :, 3]
    chc = d[:, :, 4]

    # extend dimensions
    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = chc[:, :, np.newaxis]


    u = xr.DataArray(
        u, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    v = xr.DataArray(
        v, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    chc = xr.DataArray(
        chc, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})
    dataset = set_default_attrs(dataset)
    if filename is not None:
        dataset.attrs["files"].append(filename)
    if dt is not None:
        dataset.attrs["dt"] = dt


    return dataset

def load_vec(
    filename: pathlib.Path,
    rows: int=None,
    cols: int=None,
    dt: float=None,
    frame: int=0,
)-> xr.Dataset:
    """
        load_vec(filename,rows=rows,cols=cols)
        Loads the VEC file (TECPLOT format by TSI Inc.),
        OpenPIV VEC or TXT formats
        Arguments:
            filename : file name, expected to have a header and 5 columns
            rows, cols : number of rows and columns of a vector field,
            if None, None, then parse_header is called to infer the number
            written in the header
            DELTA_T : time interval (default is None)
            frame : frame or time marker (default is None)
        Output:
            dataset is a xAarray Dataset, see xarray for help
    """
    if rows is None or cols is None:
        variables, units, rows, cols, DELTA_T, frame = parse_header(filename)

    if rows is None:  # means no headers
        d = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4))
        x = sorted_unique(d[:, 0])
        y = sorted_unique(d[:, 1])
        d = d.reshape(len(y), len(x), 5)  # .transpose(1, 0, 2)
    else:
        # d = np.genfromtxt(
        #     filename, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3, 4)
        # ).reshape(rows, cols, 5)
        d = np.genfromtxt(
            filename, skip_header=1, delimiter=",", usecols=(0, 1, 2, 3, 4)
        ).reshape(cols, rows, 5)
        x = d[:, :, 0][0, :]
        y = d[:, :, 1][:, 0]

    u = d[:, :, 2]
    v = d[:, :, 3]
    chc = d[:, :, 4]

    # extend dimensions
    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = chc[:, :, np.newaxis]

    u = xr.DataArray(
        u, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    v = xr.DataArray(
        v, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    chc = xr.DataArray(
        chc, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]}
    )

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})

    dataset = set_default_attrs(dataset)
    if filename is not None:
        dataset.attrs["files"].append(filename)
    if dt is not None:
        dataset.attrs["dt"] = dt

    return dataset

def load_vc7(
    filename: pathlib.Path,
    frame: int=0,
)-> xr.Dataset:
    """
        load_vc7(filename) or load_vc7(filename, frame=0)
        Loads the vc7 file using Lavision lvreader package,
        Arguments:
            filename : file name, pathlib.Path
        Output:
            dataset : xarray.Dataset
    """
    buffer = read_buffer(str(filename))
    data = buffer[0] # first component is a vector frame
    plane = 0 # don't understand the planes issue, simple vc7 is 0

    u = data.components["U0"][plane]
    v = data.components["V0"][plane]

    mask = np.logical_not(data.masks[plane] & data.enabled[plane])
    u[mask] = 0.0
    v[mask] = 0.0

    # scale
    u = data.scales.i.offset + u*data.scales.i.slope
    v = data.scales.i.offset + v*data.scales.i.slope

    x = np.arange(u.shape[1])
    y = np.arange(u.shape[0])

    x = data.scales.x.offset + (x+.5)*data.scales.x.slope*data.grid.x
    y = data.scales.y.offset + (y+.5)*data.scales.y.slope*data.grid.y

    x,y = np.meshgrid(x,y)
    dataset = from_arrays(x,y,u,v,mask,frame=frame)

    dataset["t"].assign_coords({"t":dataset.t+frame})

    dataset.attrs["files"].append(filename)
    dataset.attrs["dt"]  = data.attributes['FrameDt']

    return dataset

def load_directory(
        path: pathlib.Path, 
        basename: str="*", 
        ext: str=".vec",
        )->xr.Dataset:
    """
    load_directory (path,basename='*', ext='*.vec')

    Loads all the files with the chosen sextension in the directory into a
    single xarray Dataset with variables and units added as attributes

    Input:
        directory : path to the directory with .vec, .txt or .VC7 files,
        period . can be dropped

    Output:
        dataset : xarray Dataset with dimensions: x,y,t and
               dataset arrays of u,v,
               attributes of variables and units


    See more: load_vec
    """

    files = sorted(path.glob(basename+ext))


    if len(files) == 0:
        raise IOError(f"No files {basename+ext} in the directory {path} ")
    else:
        print(f"found {len(files)} files")

    dataset = []
    combined = []

    if ext.lower().endswith("vec"):
        variables, units, rows, cols, dt, frame = parse_header(files[0])

        for i, f in enumerate(files):
            dataset.append(
                load_vec(f, rows=rows, cols=cols, frame=i, dt=dt)
            )
        if len(dataset) > 0:
            combined = xr.concat(dataset, dim="t")
            combined.attrs["dt"] = dataset[0].attrs["dt"]
            combined.attrs["files"] = files
            
    elif ext.lower().endswith("vc7"):
        for i, f in enumerate(files):
            # if basename == "B*":  # quite strange to have a specific name?
            #     frame = int(f[-9:-4]) - 1
            # else:
            #     frame = i
            dataset.append(load_vc7(f, frame=i))

        if len(dataset) > 0:
            combined = xr.concat(dataset, dim="t")
            combined.attrs = dataset[-1].attrs
    elif ext.lower().endswith("txt"):
        variables, units, rows, cols, DELTA_T, frame = parse_header(files[0])

        for i, f in enumerate(files):
            dataset.append(
                load_txt(f, rows, cols, variables, units, DELTA_T, frame + i - 1)
            )
        if len(dataset) > 0:
            combined = xr.concat(dataset, dim="t")
            combined.attrs["variables"] = dataset[0].attrs["variables"]
            combined.attrs["units"] = dataset[0].attrs["units"]
            combined.attrs["DELTA_T"] = dataset[0].attrs["DELTA_T"]
            combined.attrs["files"] = files

        else:
            raise IOError("Could not read the files")

    return combined


def parse_header(filename: pathlib.Path)-> Tuple:
    """
    parse_header ( filename)
    Parses header of the file (.vec) to get the variables (typically X,Y,U,V)
    and units (can be m,mm, pix/DELTA_T or mm/sec, etc.), and the size of the
    Dataset by the number of rows and columns.
    Input:
        filename : complete path of the file to read, pathlib.Path
    Returns:
        variables : list of strings
        units : list of strings
        rows : number of rows of the Dataset
        cols : number of columns of the Dataset
        DELTA_T   : time interval between the two PIV frames in microseconds
    """

    # defaults
    frame = 0

    # split path from the filename
    fname = str(filename.name)

    # get the number in a filename if it's a .vec file from Insight
    if "." in fname[:-4]:  # day2a005003.T000.D000.P003.H001.L.vec
        frame = int(re.findall(r"\d+", fname.split(".")[0])[-1])
    elif "_" in fname[:-4]:
        frame = int(
            re.findall(r"\d+", fname.split("_")[1])[-1]
        )  # exp1_001_b.vec, .txt

    with open(filename,"r") as fid:
        header = fid.readline()

    # if the file does not have a header, can be from OpenPIV or elsewhere
    # return None
    if header[:5] != "TITLE":
        return (
            ["x", "y", "u", "v"],
            4*[POS_UNITS],
            None,
            None,
            None,
            frame,
        )

    header_list = (
        header.replace(",", " ").replace("=", " ").replace('"', " ").split()
    )

    # get variable names, typically X,Y,U,V
    variables = header_list[3:12][::2]

    # get units - this is important if it's mm or m/s
    units = header_list[4:12][::2]

    # get the size of the PIV grid in rows x cols
    rows = int(header_list[-5])
    cols = int(header_list[-3])

    # this is also important to know the time interval, DELTA_T
    ind1 = header.find("MicrosecondsPerDeltaT")
    DELTA_T = float(header[ind1:].split('"')[1])

    return (variables, units, rows, cols, DELTA_T, frame)


def get_units(filename: pathlib.Path)->Tuple[str,str,float]:
    """
    get_units(filename)

    given a full path name to the .vec file will return the names
    of length and velocity units fallback option is all None. Uses
    parse_header function, see below.

    """

    _, units, _, _, _, _ = parse_header(filename)

    if units == "":
        return (POS_UNITS, VEL_UNITS, DELTA_T)

    lUnits = units[0] # either m, mm, pix
    velUnits = units[2] # either m/s, mm/s, pix

    tUnits = velUnits.split("/")[1]  # make it 's' if exists

    return (lUnits, velUnits, tUnits)


def load_txt(
    filename: str,
    rows: int=None,
    cols: int=None,
    dt: float=None,
    frame: int=0,
)-> xr.Dataset:
    """
        load_vec(filename,rows=rows,cols=cols)
        Loads the VEC file (TECPLOT format by TSI Inc.), OpenPIV VEC or TXT
        formats
        Arguments:
            filename : file name, expected to have a header and 5 columns
            rows, cols : number of rows and columns of a vector field,
            if None, None, then parse_header is called to infer the number
            written in the header
            DELTA_T : time interval (default is None)
            frame : frame or time marker (default is None)
        Output:
            dataset is a xAarray Dataset, see xarray for help
    """
    if rows is None:  # means no headers
        d = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4))
        x = sorted_unique(d[:, 0])
        y = sorted_unique(d[:, 1])
        d = d.reshape((len(y), len(x), 5)).transpose(1, 0, 2)
    else:
        d = np.genfromtxt(
            filename, skip_header=1, delimiter=",", usecols=(0, 1, 2, 3, 4)
        )
        d = d.reshape((rows, cols, 5))

        x = d[:, :, 0][0, :]
        y = d[:, :, 1][:, 0]

    u = d[:, :, 2]
    v = d[:, :, 3]
    chc = d[:, :, 4]


    dataset = xr.Dataset({
        "u": xr.DataArray(
                    u[:, :, np.newaxis], 
                    dims=("x", "y", "t"), 
                    coords={"x": x, "y": y, "t": [frame]}
                    ), 
        "v": xr.DataArray(
                    v[:, :, np.newaxis], 
                    dims=("x", "y", "t"), 
                    coords={"x": x, "y": y, "t": [frame]}
                    ), 
        "chc": xr.DataArray(
                    chc[:, :, np.newaxis], 
                    dims=("x", "y", "t"), 
                    coords={"x": x, "y": y, "t": [frame]}
                    ),
        })

    dataset = set_default_attrs(dataset)
    if dt is not None:
        dataset.attrs["dt"] = dt
    dataset.attrs["files"].append(filename)

    return dataset


def sorted_unique(array):
    """ Returns not sorted sorted_unique """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]
