# -*- coding: utf-8 -*-

"""
Contains functions for reading flow fields in various formats
"""

import pathlib
import re
import warnings
from typing import List, Tuple
import numpy as np
import xarray as xr
import pandas as pd
from numpy.typing import ArrayLike

try:
    from lvreader import read_buffer
except ImportError:
    warnings.warn("lvreader is not installed")


# Defaults
POS_UNITS: str = "pix"  # or mm, m, after scaling
TIME_UNITS: str = "frame"  # "frame" if not scaled, can become 'sec' or 'msec', 'usec'
# after scaling can be m/s, mm/s
VEL_UNITS: str = POS_UNITS  # default is displacement in pix
DELTA_T: np.float64 = 0.0  # default is 0. i.e. uknown, can be any float value


def sorted_unique(arr: ArrayLike) -> ArrayLike:
    """creates a sorted unique numpy array"""
    arr1, c = np.unique(arr, return_counts=True)
    out = arr1[np.argsort(-c)]
    return out


def set_default_attrs(dataset: xr.Dataset) -> xr.Dataset:
    """Defines default attributes:

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
    rows: int = 5,
    cols: int = 8,
    grid: List = None,
    frame: int = 0,
    noise_sigma: float = 1.0,
) -> xr.Dataset:
    """Creates a sample Dataset for the tests.

    Args:
        rows (int) : number of points along vertical coordinate,
            corresponds to 'y'
        cols (int) : number of grid points along horizontal coordinate, 'x'
        grid (int, int) : spacing between vectors in two directions (x,y)
        frame (int):  frame number
        noise_sigma (float): strength of Gaussian noise to add

    Returns:
        xarray.Dataset(): dataset
    """
    if grid is None:
        grid = [16, 8]

    x = np.arange(grid[0], (cols + 1) * grid[0], grid[0])
    y = np.arange(grid[1], (rows + 1) * grid[1], grid[1])

    xm, ym = np.meshgrid(x, y)
    u = (
        np.ones_like(xm)
        + np.linspace(0.0, 10.0, cols)
        + noise_sigma * np.random.randn(1, cols)
    )
    v = (
        np.zeros_like(ym)
        + np.linspace(-1.0, 1.0, rows).reshape(rows, 1)
        + noise_sigma * np.random.randn(rows, 1)
    )

    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = np.ones_like(u)

    u = xr.DataArray(u, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})
    v = xr.DataArray(v, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})
    chc = xr.DataArray(chc, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})
    dataset = set_default_attrs(dataset)

    return dataset


def create_sample_Dataset(
    n_frames: int = 5, rows: int = 5, cols: int = 3, noise_sigma: float = 0.0
) -> xr.Dataset:
    """Creates a syntetic dataset

    Args:
        n_frames : number of frames
        rows (int) : number of points along vertical coordinate,
            corresponds to 'y'
        cols  : number of grid points along horizontal coordinate, 'x'
        grid : spacing between vectors in two directions (x,y)
        frame :  frame number
        noise_sigma : strength of Gaussian noise to add

    Returns:
        dataset: PIVPy dataset

    Example:
            ds = create_sample_dataset(n_frames=3)
    """

    dataset = []
    for i in range(n_frames):
        dataset.append(
            create_sample_field(rows=rows, cols=cols, frame=i, noise_sigma=noise_sigma)
        )

    combined = xr.concat(dataset, dim="t")
    combined = set_default_attrs(combined)

    return combined


def create_uniform_strain():
    """creates constant strain field"""
    return create_sample_field(noise_sigma=0.0)


def from_arrays(
    x: ArrayLike,
    y: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    mask: np.array,
    frame: int = 0,
) -> xr.Dataset:
    """creates a dataset from two-dimensional Numpy arrays
    of x,y,u,v and mask

    Args:
        x (array): x
        y (array): y
        u (array): u
        v (array): v
        mask (array): mask, all numpy arrays of the same shape
        frame (int): frame number, default is 0
    Returns:
        ds (xarray.Dataset): xarray dataset with default attributes
    Example:
        ds = io.from_arrays(x,y,u,v,mask,frame=0)
    """
    # create dataset structure of appropriate size
    dataset = create_sample_field(rows=x.shape[0], cols=x.shape[1], frame=frame)
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
    frame: int = 0,
    filename: str = None,
) -> xr.Dataset:
    """creates pivpy.Dataset from pandas DataFrame

    Args:
        df (pd.DataFrame): DataFrame with columns of x,y,u,v
        frame (int, optional): frame number. Defaults to 0.
        filename (str, optional): filename to add to the attributes. Defaults to None.

    Returns:
        xr.Dataset: pivpy.Dataset
    """
    d = df.to_numpy()

    x = sorted_unique(d[:, 0])
    y = sorted_unique(d[:, 1])
    if d.shape[1] < 5:  # not always there's a mask
        tmp = np.ones((d.shape[0], 5))
        tmp[:, :-1] = d
        d = tmp

    d = d.reshape(len(y), len(x), 5)  # .transpose(1, 0, 2)

    u = d[:, :, 2]
    v = d[:, :, 3]
    chc = d[:, :, 4]

    # extend dimensions
    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = chc[:, :, np.newaxis]

    u = xr.DataArray(u, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})
    v = xr.DataArray(v, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})
    chc = xr.DataArray(chc, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})
    dataset = set_default_attrs(dataset)
    if filename is not None:
        dataset.attrs["files"].append(str(filename))

    return dataset


def load_vec(
    filename: pathlib.Path,
    rows: int = None,
    cols: int = None,
    delta_t: float = None,
    frame: int = 0,
) -> xr.Dataset:
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
        _, _, rows, cols, dt, frame, _ = parse_header(filename)

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

    u = xr.DataArray(u, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})
    v = xr.DataArray(v, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})
    chc = xr.DataArray(chc, dims=("y", "x", "t"), coords={"x": x, "y": y, "t": [frame]})

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})

    dataset = set_default_attrs(dataset)
    if filename is not None:
        dataset.attrs["files"].append(str(filename))
    if delta_t is not None:
        dataset.attrs["delta_t"] = delta_t

    return dataset


def load_vc7(
    filename: pathlib.Path,
    frame: int = 0,
) -> xr.Dataset:
    """
    load_vc7(filename) or load_vc7(filename, frame=0)
    Loads the vc7 file using Lavision lvreader package,
    Arguments:
        filename : file name, pathlib.Path
    Output:
        dataset : xarray.Dataset
    """
    buffer = read_buffer(str(filename))
    data = buffer[0]  # first component is a vector frame
    plane = 0  # don't understand the planes issue, simple vc7 is 0

    u = data.components["U0"][plane]
    v = data.components["V0"][plane]

    mask = np.logical_not(data.masks[plane] & data.enabled[plane])
    u[mask] = 0.0
    v[mask] = 0.0

    # scale
    u = data.scales.i.offset + u * data.scales.i.slope
    v = data.scales.i.offset + v * data.scales.i.slope

    x = np.arange(u.shape[1])
    y = np.arange(u.shape[0])

    x = data.scales.x.offset + (x + 0.5) * data.scales.x.slope * data.grid.x
    y = data.scales.y.offset + (y + 0.5) * data.scales.y.slope * data.grid.y

    x, y = np.meshgrid(x, y)
    dataset = from_arrays(x, y, u, v, mask, frame=frame)

    dataset["t"].assign_coords({"t": dataset.t + frame})

    dataset.attrs["files"].append(str(filename))
    dataset.attrs["delta_t"] = data.attributes["FrameDt"]

    return dataset


def load_directory(
    path: pathlib.Path,
    basename: str = "*",
    ext: str = ".vec",
) -> xr.Dataset:
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

    files = sorted(path.glob(basename + ext))

    if len(files) == 0:
        raise IOError(f"No files {basename+ext} in the directory {path} ")
    
    print(f"found {len(files)} files")

    dataset = []
    combined = []

    _, _, rows, cols, delta_t, _, method = parse_header(files[0])

    if method is load_vc7:
        for i, f in enumerate(files):
            dataset.append(load_vc7(f, frame=i))
    else:
        for i, f in enumerate(files):
            dataset.append(method(f, rows=rows, cols=cols, frame=i, delta_t=delta_t))

    if len(dataset) > 0:
        combined = xr.concat(dataset, dim="t")
        combined.attrs["delta_t"] = dataset[-1].attrs["delta_t"]
        combined.attrs["files"] = str(files)
        return combined

    else:
        raise IOError("Could not read the files")


def parse_header(filename: pathlib.Path) -> Tuple[str, ...]:
    """ parses the header line in the file to obtain attributes

    Args:
        filename (pathlib.Path): txt, vec file name

    Returns:
        Tuple[str, ...]: 
        variables:  
        units : 
        rows : 
        cols : 
        delta_t: 
        frame : 
        method :
    """
    fname = filename.stem.split(".")[0] # str(filename.name).split(".")[0]

    try:
        frame = int(re.findall(r"\d+", fname)[-1])
        # print(int(re.findall(r'\d+', tmp)[-1]))
        # print(int(''.join(filter(str.isdigit,tmp))[-1]))
        # print(int(re.findall(r'[0-9]+', tmp)[-1]))
    except ValueError:
        frame = 0

    # binary, no header
    if filename.suffix.lower() == ".vc7":
        return (
            ["x", "y", "u", "v"],
            4 * [POS_UNITS],
            None,
            None,
            None,
            frame,
            load_vc7,
        )

    with open(filename, "r", encoding="utf-8") as fid:
        header = fid.readline()
        # print(header)

    # if the file does not have a header, can be from OpenPIV or elsewhere
    # return None
    if header.startswith("#DaVis"):
        header_list = header.split(" ")
        rows = header_list[4]
        cols = header_list[5]
        pos_units = header_list[7]
        vel_units = header_list[-1]
        variables = ["x", "y", "u", "v"]
        units = [pos_units, pos_units, vel_units, vel_units]
        dt = 0.0
        method = load_davis8_txt
        return variables, units, rows, cols, dt, frame, method

    elif header.startswith("TITLE="):  # Insight
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
        dt = float(header[ind1:].split('"')[1])
        method = load_vec

        return variables, units, rows, cols, dt, frame, method

    else:  # no header, probably OpenPIV txt
        method = load_openpiv_txt
        return (
            ["x", "y", "u", "v"],
            4 * [POS_UNITS],
            None,
            None,
            None,
            frame,
            method,
        )


def get_units(filename: pathlib.Path) -> Tuple[str, str, float]:
    """
    get_units(filename)

    given a full path name to the .vec file will return the names
    of length and velocity units fallback option is all None. Uses
    parse_header function, see below.

    """

    _, units, _, _, _, _, _ = parse_header(filename)

    if units == "":
        return (POS_UNITS, VEL_UNITS, DELTA_T)

    lUnits = units[0]  # either m, mm, pix
    velUnits = units[2]  # either m/s, mm/s, pix

    tUnits = velUnits.split("/")[1]  # make it 's' if exists

    return (lUnits, velUnits, tUnits)


def load_openpiv_txt(
    filename: str,
    rows: int = None,
    cols: int = None,
    delta_t: float = None,
    frame: int = 0,
) -> xr.Dataset:
    """ loads OpenPIV txt file 

    Args:
        filename (str): _description_
        rows (int, optional): _description_. Defaults to None.
        cols (int, optional): _description_. Defaults to None.
        delta_t (float, optional): _description_. Defaults to None.
        frame (int, optional): _description_. Defaults to 0.

    Returns:
        xr.Dataset: _description_
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

    dataset = xr.Dataset(
        {
            "u": xr.DataArray(
                u[:, :, np.newaxis],
                dims=("x", "y", "t"),
                coords={"x": x, "y": y, "t": [frame]},
            ),
            "v": xr.DataArray(
                v[:, :, np.newaxis],
                dims=("x", "y", "t"),
                coords={"x": x, "y": y, "t": [frame]},
            ),
            "chc": xr.DataArray(
                chc[:, :, np.newaxis],
                dims=("x", "y", "t"),
                coords={"x": x, "y": y, "t": [frame]},
            ),
        }
    )

    dataset = set_default_attrs(dataset)
    if delta_t is not None:
        dataset.attrs["delta_t"] = delta_t
    dataset.attrs["files"].append(str(filename))

    return dataset


def load_davis8_txt(
    filename: pathlib.Path,
    rows: int = None, # pylint: disable=W0613
    cols: int = None, # pylint: disable=W0613
    delta_t: float = 0.0, # pylint: disable=W0613
    frame: int = 0,
) -> xr.Dataset:
    """loads Davis8 old ASCII tables format

    Args:
        filename (pathlib.Path): Davis8 filename.txt
        rows (int, optional): rows. Defaults to None.
        cols (int, optional): cols. Defaults to None.
        delta_t (float, optional): delta_t. Defaults to 0.0.
        frame (int, optional): frame number. Defaults to 0.

    Returns:
        xr.Dataset: pivpy.Dataset
    """
    dataframe = pd.read_csv(
        filename, delimiter="\t", skiprows=1, names=["x", "y", "u", "v"], decimal=","
    )
    dataset = from_df(dataframe, frame=frame)
    # print(f'{rows},{cols},{delta_t}')
    return dataset


# def sorted_unique(array):
#     """Returns not sorted sorted_unique"""
#     uniq, index = np.unique(array, return_index=True)
#     return uniq[index.argsort()]
