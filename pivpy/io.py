# -*- coding: utf-8 -*-

"""
PIV Data I/O Module with Plugin-Based Architecture

This module provides functions for reading and writing PIV (Particle Image 
Velocimetry) data in various formats. It features a modern plugin-based 
architecture for extensibility while maintaining backward compatibility.

High-Level API (Recommended)
-----------------------------
* `read_piv()` - Auto-detect format and read single file
* `read_directory()` - Batch read directory of files
* `save_piv()` - Save dataset to NetCDF, Zarr, or CSV
* `register_reader()` - Register custom readers

Legacy API (Backward Compatible)
---------------------------------
* `load_vec()` - Load Insight VEC files
* `load_openpiv_txt()` - Load OpenPIV text files
* `load_davis8_txt()` - Load Davis8 ASCII files
* `load_vc7()` - Load LaVision VC7 binary files
* `load_pivlab()` - Load PIVLab MAT files
* `load_directory()` - Load directory (old API)

Plugin Architecture
-------------------
The module uses a plugin-based architecture with:
* `PIVReader` - Abstract base class for readers
* `PIVReaderRegistry` - Registry for managing readers
* `PIVMetadata` - Structured metadata storage

Built-in Readers:
* `InsightVECReader` - TSI Insight VEC files (TECPLOT format)
* `OpenPIVReader` - OpenPIV text files (5-6 columns)
* `Davis8Reader` - Davis8 ASCII format
* `LaVisionVC7Reader` - LaVision VC7 binary files
* `PIVLabReader` - PIVLab MAT files (HDF5)

Examples
--------
Basic usage with auto-detection:

    >>> import pivpy.io as io
    >>> data = io.read_piv('velocity_field.vec')
    >>> print(data['u'].shape)

Read with explicit format:

    >>> data = io.read_piv('field.txt', format='openpiv')

Batch read directory:

    >>> data = io.read_directory('data/', pattern='Run*', ext='.vec')
    >>> print(f"Loaded {len(data['t'])} frames")

Extract metadata without loading data:

    >>> reader = io._REGISTRY.find_reader('field.vec')
    >>> metadata = reader.read_metadata('field.vec')
    >>> print(f"Grid size: {metadata.rows}x{metadata.cols}")

Save to different formats:

    >>> io.save_piv(data, 'output.nc', format='netcdf')
    >>> io.save_piv(data, 'output.csv', format='csv', frame=0)

Create custom reader:

    >>> class MyReader(io.PIVReader):
    ...     def can_read(self, filepath):
    ...         return filepath.suffix == '.custom'
    ...     def read_metadata(self, filepath):
    ...         return io.PIVMetadata()
    ...     def read(self, filepath, **kwargs):
    ...         # Custom reading logic
    ...         return dataset
    >>> io.register_reader(MyReader())

Utility Functions
-----------------
* `create_sample_field()` - Generate synthetic velocity field
* `create_sample_Dataset()` - Generate multi-frame synthetic dataset
* `from_arrays()` - Create dataset from NumPy arrays
* `from_df()` - Create dataset from pandas DataFrame
* `set_default_attrs()` - Apply standard attributes to dataset
"""

import pathlib
import re
import warnings
from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import xarray as xr
import pandas as pd
from numpy.typing import ArrayLike

try:
    from lvpyio import read_buffer
except ImportError:
    read_buffer = None
    warnings.warn("lvreader is not installed, use pip install lvpyio")

try:
    import h5py
except ImportError:
    h5py = None
    warnings.warn("h5py is not installed, use pip install h5py to read PIVLab MAT files")


# Defaults
POS_UNITS: str = "pix"  # or mm, m, after scaling
TIME_UNITS: str = "frame"  # "frame" if not scaled, can become 'sec' or 'msec', 'usec'
# after scaling can be m/s, mm/s
VEL_UNITS: str = POS_UNITS  # default is displacement in pix
DELTA_T: np.float64 = 0.0  # default is 0. i.e. uknown, can be any float value


def unsorted_unique(arr: ArrayLike) -> ArrayLike:
    """creates a sorted unique numpy array"""
    arr1, c = np.unique(arr, return_index=True)
    out = arr1[c.argsort()]
    return out, c


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
    """Creates a synthetic velocity field dataset for testing
    
    Args:
        rows (int, optional): Number of grid points along vertical (y) direction. Defaults to 5.
        cols (int, optional): Number of grid points along horizontal (x) direction. Defaults to 8.
        grid (List, optional): Grid spacing [dx, dy] in pixels. Defaults to [16, 8].
        frame (int, optional): Frame number for time coordinate. Defaults to 0.
        noise_sigma (float, optional): Standard deviation of Gaussian noise to add. Defaults to 1.0.
        
    Returns:
        xr.Dataset: Synthetic PIVPy dataset with u, v velocity components and chc mask
        
    Example:
        >>> data = create_sample_field(rows=10, cols=15, noise_sigma=0.5)
        >>> print(data.dims)  # {'x': 15, 'y': 10, 't': 1}
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
    """Creates a synthetic time-series dataset with multiple frames
    
    Args:
        n_frames (int, optional): Number of time frames to generate. Defaults to 5.
        rows (int, optional): Number of grid points along vertical (y) direction. Defaults to 5.
        cols (int, optional): Number of grid points along horizontal (x) direction. Defaults to 3.
        noise_sigma (float, optional): Standard deviation of Gaussian noise. Defaults to 0.0 (no noise).
        
    Returns:
        xr.Dataset: Synthetic PIVPy dataset with time dimension
        
    Example:
        >>> dataset = create_sample_Dataset(n_frames=10, rows=20, cols=30, noise_sigma=0.1)
        >>> print(dataset.dims)  # {'x': 30, 'y': 20, 't': 10}
        >>> print(dataset.t.values)  # [0, 1, 2, ..., 9]
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
    """Creates a PIVPy dataset from 2D NumPy arrays
    
    Args:
        x (ArrayLike): 2D array of x coordinates
        y (ArrayLike): 2D array of y coordinates  
        u (ArrayLike): 2D array of u velocity component
        v (ArrayLike): 2D array of v velocity component
        mask (np.array): 2D array of mask/quality values
        frame (int, optional): Frame number for time coordinate. Defaults to 0.
        
    Returns:
        xr.Dataset: PIVPy dataset with velocity fields and coordinates
        
    Example:
        >>> x, y = np.meshgrid(np.arange(10), np.arange(8))
        >>> u = np.random.rand(8, 10)
        >>> v = np.random.rand(8, 10)
        >>> mask = np.ones_like(u)
        >>> dataset = from_arrays(x, y, u, v, mask, frame=0)
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
    """Creates PIVPy dataset from pandas DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'x', 'y', 'u', 'v', and optionally 'chc' (mask)
        frame (int, optional): Frame number for time coordinate. Defaults to 0.
        filename (str, optional): Filename to store in dataset attributes. Defaults to None.
        
    Returns:
        xr.Dataset: PIVPy dataset with velocity fields and coordinates
        
    Raises:
        ValueError: If x or y coordinates are not properly sorted
        
    Example:
        >>> df = pd.DataFrame({'x': [...], 'y': [...], 'u': [...], 'v': [...]})
        >>> dataset = from_df(df, frame=0)
    """
    d = df.to_numpy()

    x, ix = unsorted_unique(d[:, 0])
    y, iy = unsorted_unique(d[:, 1])
    
    if d.shape[1] < 5: # davis8 does not have mask or chc
        d = np.column_stack((d,np.zeros_like(d[:,-1])))
        
    if ix[1] == 1:  # x grows first
        d = d.reshape(len(y), len(x), 5).transpose(1, 0, 2)
    elif iy[1] == 1:  # y grows first
        d = d.reshape(len(x), len(y), 5)
    else:
        raise ValueError(
            'Data is not properly sorted. Either x or y coordinates must be '
            'monotonically ordered. Check your input data format.'
        )

    u = d[:, :, 2]
    v = d[:, :, 3]
    chc = d[:, :, 4]

    # extend dimensions
    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = chc[:, :, np.newaxis]

    u = xr.DataArray(u, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]})
    v = xr.DataArray(v, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]})
    chc = xr.DataArray(chc, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]})

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})
    dataset = set_default_attrs(dataset)
    if filename is not None:
        dataset.attrs["files"].append(str(filename))

    return dataset


def load_vec(
    filename,
    rows: int = None,
    cols: int = None,
    delta_t: float = None,
    frame: int = 0,
) -> xr.Dataset:
    """Loads VEC file in TECPLOT format (TSI Inc.), OpenPIV VEC or TXT formats
    
    Args:
        filename (str or pathlib.Path): Path to the VEC file with header and 5 columns (x, y, u, v, mask)
        rows (int, optional): Number of rows in the vector field. If None, will be parsed from header.
        cols (int, optional): Number of columns in the vector field. If None, will be parsed from header.
        delta_t (float, optional): Time interval between frames. Defaults to None.
        frame (int, optional): Frame or time marker. Defaults to 0.
        
    Returns:
        xr.Dataset: PIVPy dataset with velocity fields and coordinates
        
    Raises:
        ValueError: If coordinates are not properly sorted
        IOError: If file cannot be read
        
    Example:
        >>> dataset = load_vec('data/velocity_field.vec')
        >>> dataset = load_vec('data/field.vec', delta_t=0.001, frame=5)
    """
    # Ensure filename is a pathlib.Path object
    filename = pathlib.Path(filename) if not isinstance(filename, pathlib.Path) else filename
    if rows is None or cols is None:
        _, _, rows, cols, dt, frame, _ = parse_header(filename)
        # print(f'rows = {rows}, cols = {cols}')

    if rows is None:  # means no headers, openpiv vec file
        # d = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4))
        d = np.genfromtxt(
            filename, 
            usecols=(0, 1, 2, 3, 4)
          )
        x, ix = unsorted_unique(d[:, 0])
        y, iy = unsorted_unique(d[:, 1])
        
        # print(f'rows = {len(y)}, cols = {len(x)}')
        
        if ix[1] == 1:  # x grows first
            d = d.reshape(len(y), len(x), 5).transpose(1, 0, 2)
        elif iy[1] == 1:  # y grows first
            d = d.reshape(len(y), len(x), 5)
        else:
            raise ValueError(
                'Data is not properly sorted. Either x or y coordinates must be '
                'monotonically ordered. Check your VEC file format.'
            )
    else:  # Insight VEC file
        d = np.genfromtxt(
            filename, skip_header=1, delimiter=",", usecols=(0, 1, 2, 3, 4)
        ).reshape(cols, rows, 5).transpose(1, 0, 2)
        
        x = d[:, :, 0][:, 0]
        y = d[:, :, 1][0, :]
 

    u = d[:, :, 2]
    v = d[:, :, 3]
    chc = d[:, :, 4]

    # extend dimensions
    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = chc[:, :, np.newaxis]

    u = xr.DataArray(u, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]})
    v = xr.DataArray(v, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]})
    chc = xr.DataArray(chc, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]})

    dataset = xr.Dataset({"u": u, "v": v, "chc": chc})

    dataset = set_default_attrs(dataset)
    if filename is not None:
        dataset.attrs["files"].append(str(filename))
    if delta_t is not None:
        dataset.attrs["delta_t"] = delta_t

    return dataset

def load_insight_vec_as_csv(
    filename: pathlib.Path,
    rows: int = None,
    cols: int = None,
    delta_t: float = None,
    frame: int = 0,
) -> xr.Dataset:
    """
    load_insight_vec_as_csv(filename,rows=rows,cols=cols)
    Loads the VEC file (TECPLOT format by TSI Inc.),
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
    df = pd.read_csv(
        filename,
        header=None,
        skiprows=1,
        usecols=[0,1,2,3,4],
        names=["x","y","u","v","chc"],
        )
    dataset = from_df(df,frame=frame,filename=filename)
    
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
    if read_buffer is None:
        raise ImportError(
            "lvpyio is required to read VC7 files. "
            "Install it with: pip install lvpyio"
        )
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
    """Loads all velocity field files from a directory into a single xarray Dataset
    
    Args:
        path (pathlib.Path): Path to directory containing velocity field files
        basename (str, optional): Filename pattern to match (e.g., 'Run*'). Defaults to "*".
        ext (str, optional): File extension (.vec, .txt, or .vc7). Defaults to ".vec".
        
    Returns:
        xr.Dataset: Combined dataset with time dimension containing all loaded files
        
    Raises:
        IOError: If no matching files are found in the directory
        
    Example:
        >>> import pathlib
        >>> path = pathlib.Path('data/velocity_fields')
        >>> dataset = load_directory(path, basename='Run*', ext='.vec')
        >>> print(dataset.dims)  # Should show x, y, t dimensions
        
    Note:
        Files are sorted alphabetically and assigned sequential frame numbers (0, 1, 2, ...).
        All files must have the same grid dimensions (rows, cols).
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
    """Loads OpenPIV text output file
    
    Args:
        filename (str): Path to OpenPIV text file with 5 or 6 columns (x, y, u, v, flags, mask).
            The function automatically detects the number of columns.
            - 5 columns: x, y, u, v, flags (older format)
            - 6 columns: x, y, u, v, flags, mask (newer format with mask)
        rows (int, optional): Number of rows in vector field. Defaults to None (auto-detect).
        cols (int, optional): Number of columns in vector field. Defaults to None (auto-detect).
        delta_t (float, optional): Time interval between frames. Defaults to None.
        frame (int, optional): Frame number. Defaults to 0.
        
    Returns:
        xr.Dataset: PIVPy dataset with velocity fields. Includes 'mask' DataArray if available
            in the input file.
        
    Raises:
        ValueError: If coordinates are not properly sorted
        
    Example:
        >>> dataset = load_openpiv_txt('openpiv_output.txt')
        >>> # For files with mask column:
        >>> dataset = load_openpiv_txt('openpiv_with_mask.txt')
        >>> print('mask' in dataset)  # True if mask column was present
    """
    # Detect number of columns by reading first non-comment line
    with open(filename, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Count number of columns in first data line
                n_cols = len(stripped.split())
                break
    
    if rows is None:  # means no headers
        # Load data based on detected number of columns
        if n_cols >= 6:
            d = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4, 5))
        else:
            d = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4))
            
        x, ix = unsorted_unique(d[:, 0])
        y, iy = unsorted_unique(d[:, 1])

        if ix[1] == 1:  # x grows first
            d = d.reshape(len(y), len(x), n_cols if n_cols >= 6 else 5).transpose(1, 0, 2)
        elif iy[1] == 1:  # y grows first
            d = d.reshape(len(y), len(x), n_cols if n_cols >= 6 else 5)
        else:
            raise ValueError(
                'Data is not properly sorted. Either x or y coordinates must be '
                'monotonically ordered. Check your OpenPIV file format.'
            )
    else:
        if n_cols >= 6:
            d = np.genfromtxt(
                filename, skip_header=1, delimiter=",", usecols=(0, 1, 2, 3, 4, 5)
            )
            d = d.reshape((rows, cols, 6))
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

    dataset_dict = {
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
    
    # Add mask if available (6th column)
    if n_cols >= 6:
        mask = d[:, :, 5]
        dataset_dict["mask"] = xr.DataArray(
            mask[:, :, np.newaxis],
            dims=("x", "y", "t"),
            coords={"x": x, "y": y, "t": [frame]},
        )

    dataset = xr.Dataset(dataset_dict)

    dataset = set_default_attrs(dataset)
    if delta_t is not None:
        dataset.attrs["delta_t"] = delta_t
    dataset.attrs["files"].append(str(filename))

    return dataset


def load_openpiv_txt_as_csv(
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
    df = pd.read_csv(
        filename, 
        header=None,
        names=['x','y','u','v','chc'],
        delim_whitespace=True,
        usecols = (0,1,2,3,4),
    )
    
    dataset = from_df(
        df,
        frame=frame,
        filename=filename
    )
    
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
    dataset = from_df(dataframe, frame=frame,filename=filename)
    # print(f'{rows},{cols},{delta_t}')
    return dataset


def load_pivlab(
    filename: pathlib.Path,
    frame: int = None,
) -> xr.Dataset:
    """Loads PIVLab MAT file (MATLAB HDF5 format)
    
    PIVLab is a MATLAB toolbox for PIV analysis that saves results in HDF5-based MAT files.
    The data structure uses a 'resultslist' array containing references to datasets:
    - resultslist[:, 0]: x coordinates (meshgrid format)
    - resultslist[:, 1]: y coordinates (meshgrid format)
    - resultslist[:, 2]: u velocity component
    - resultslist[:, 3]: v velocity component
    - resultslist[:, 4]: typevector (mask: 1=valid, 0=invalid)
    - resultslist[:, 5-10]: additional derived quantities (optional)
    
    Args:
        filename (pathlib.Path): Path to PIVLab .mat file
        frame (int, optional): Specific frame to load. If None, loads all frames. Defaults to None.
        
    Returns:
        xr.Dataset: PIVPy dataset with velocity fields and coordinates.
            If multiple frames exist and frame=None, returns dataset with time dimension.
            If single frame or specific frame requested, returns single time slice.
            
    Raises:
        ImportError: If h5py is not installed
        IOError: If file cannot be read or doesn't contain expected PIVLab structure
        ValueError: If resultslist structure is invalid
        
    Example:
        >>> # Load all frames from a PIVLab file
        >>> dataset = load_pivlab('pivlab_results.mat')
        >>> print(dataset.dims)  # {'x': 169, 'y': 340, 't': 11}
        >>> 
        >>> # Load a specific frame
        >>> dataset = load_pivlab('pivlab_results.mat', frame=5)
        >>> print(dataset.dims)  # {'x': 169, 'y': 340, 't': 1}
        
    Note:
        - Requires h5py package: pip install h5py
        - PIVLab files are MATLAB v7.3 format (HDF5-based)
        - Coordinates are extracted from the first frame and assumed constant across frames
        - Invalid vectors (typevector==0) are preserved in the mask but not modified in u/v
    """
    # Ensure filename is a pathlib.Path object
    filename = pathlib.Path(filename) if not isinstance(filename, pathlib.Path) else filename
    
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to read PIVLab MAT files. "
            "Install it with: pip install h5py"
        )
    
    try:
        with h5py.File(filename, 'r') as f:
            # Check if resultslist exists
            if 'resultslist' not in f:
                raise IOError(
                    f"File {filename} does not contain 'resultslist'. "
                    "This may not be a valid PIVLab MAT file."
                )
            
            resultslist = f['resultslist']
            n_frames = resultslist.shape[0]
            
            # Validate structure
            if resultslist.shape[1] < 5:
                raise ValueError(
                    f"resultslist has {resultslist.shape[1]} columns, expected at least 5 "
                    "(x, y, u, v, typevector)"
                )
            
            # Determine which frames to load
            if frame is not None:
                if frame < 0 or frame >= n_frames:
                    raise ValueError(
                        f"Frame {frame} out of range. File contains {n_frames} frames (0-{n_frames-1})"
                    )
                frames_to_load = [frame]
            else:
                frames_to_load = list(range(n_frames))
            
            # Load data for each frame
            datasets = []
            for frame_idx in frames_to_load:
                # Extract references and load data
                x_ref = resultslist[frame_idx, 0]
                y_ref = resultslist[frame_idx, 1]
                u_ref = resultslist[frame_idx, 2]
                v_ref = resultslist[frame_idx, 3]
                mask_ref = resultslist[frame_idx, 4]
                
                # Dereference and load arrays
                x = np.array(f[x_ref])
                y = np.array(f[y_ref])
                u = np.array(f[u_ref])
                v = np.array(f[v_ref])
                typevector = np.array(f[mask_ref])
                
                # Extract coordinate vectors from meshgrid
                # PIVLab uses a non-standard meshgrid convention where:
                # - The 'x' array has x-values varying along rows (axis 0)
                # - The 'y' array has y-values varying along columns (axis 1)
                # For PIVPy's (y, x, t) convention, we swap them:
                x_coords = y[0, :]  # y array's first row gives x coordinates
                y_coords = x[:, 0]  # x array's first column gives y coordinates
                
                # Create dataset for this frame
                u_xr = xr.DataArray(
                    u[:, :, np.newaxis],
                    dims=("y", "x", "t"),
                    coords={"x": x_coords, "y": y_coords, "t": [frame_idx]},
                )
                v_xr = xr.DataArray(
                    v[:, :, np.newaxis],
                    dims=("y", "x", "t"),
                    coords={"x": x_coords, "y": y_coords, "t": [frame_idx]},
                )
                # Use typevector as chc (choice/mask): 1=valid, 0=invalid
                chc_xr = xr.DataArray(
                    typevector[:, :, np.newaxis],
                    dims=("y", "x", "t"),
                    coords={"x": x_coords, "y": y_coords, "t": [frame_idx]},
                )
                
                frame_dataset = xr.Dataset({"u": u_xr, "v": v_xr, "chc": chc_xr})
                datasets.append(frame_dataset)
            
            # Read calibration if available
            calxy = 1.0
            caluv = 1.0
            if 'calxy' in f:
                calxy = float(np.array(f['calxy']).flat[0])
            if 'caluv' in f:
                caluv = float(np.array(f['caluv']).flat[0])
    
    except (IOError, ValueError):
        # Re-raise our own exceptions
        raise
    except (KeyError, h5py.h5r.ReferenceError) as e:
        raise IOError(
            f"Error reading PIVLab file {filename}: Invalid file structure - {str(e)}"
        )
    except Exception as e:
        raise IOError(f"Error reading PIVLab file {filename}: {str(e)}")
    
    # Combine frames if multiple
    if len(datasets) > 1:
        combined = xr.concat(datasets, dim="t")
    else:
        combined = datasets[0]
    
    # Set default attributes
    combined = set_default_attrs(combined)
    combined.attrs["files"].append(str(filename))
    
    # Apply calibration factors if not default
    if calxy != 1.0:
        combined.coords["x"] = combined.coords["x"] * calxy
        combined.coords["y"] = combined.coords["y"] * calxy
    if caluv != 1.0:
        combined["u"] = combined["u"] * caluv
        combined["v"] = combined["v"] * caluv
    
    return combined


# def sorted_unique(array):
#     """Returns not sorted sorted_unique"""
#     uniq, index = np.unique(array, return_index=True)
#     return uniq[index.argsort()]


# ============================================================================
# New Plugin-Based Architecture
# ============================================================================


@dataclass
class PIVMetadata:
    """Structured metadata for PIV datasets
    
    Attributes:
        pos_units (str): Position/length units (e.g., 'pix', 'mm', 'm')
        vel_units (str): Velocity units (e.g., 'pix', 'mm/s', 'm/s')
        time_units (str): Time units (e.g., 'frame', 's', 'ms', 'us')
        delta_t (float): Time interval between frames
        variables (List[str]): Variable names (e.g., ['x', 'y', 'u', 'v'])
        rows (Optional[int]): Number of rows in the grid
        cols (Optional[int]): Number of columns in the grid
        frame (int): Frame number
        extra (Dict[str, Any]): Additional format-specific metadata
    """
    pos_units: str = POS_UNITS
    vel_units: str = VEL_UNITS
    time_units: str = TIME_UNITS
    delta_t: float = DELTA_T
    variables: List[str] = field(default_factory=lambda: ['x', 'y', 'u', 'v'])
    rows: Optional[int] = None
    cols: Optional[int] = None
    frame: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class PIVReader(ABC):
    """Abstract base class for PIV file readers
    
    All reader implementations must inherit from this class and implement
    the abstract methods. This provides a consistent interface for reading
    different PIV file formats.
    """
    
    @abstractmethod
    def can_read(self, filepath: pathlib.Path) -> bool:
        """Check if this reader can handle the given file
        
        Args:
            filepath (pathlib.Path): Path to the file to check
            
        Returns:
            bool: True if this reader can handle the file, False otherwise
        """
        pass
    
    @abstractmethod
    def read_metadata(self, filepath: pathlib.Path) -> PIVMetadata:
        """Extract metadata from file without loading full data
        
        Args:
            filepath (pathlib.Path): Path to the file
            
        Returns:
            PIVMetadata: Structured metadata object
            
        Raises:
            IOError: If file cannot be read or parsed
        """
        pass
    
    @abstractmethod
    def read(self, filepath: pathlib.Path, **kwargs) -> xr.Dataset:
        """Load full dataset from file
        
        Args:
            filepath (pathlib.Path): Path to the file
            **kwargs: Additional reader-specific options
            
        Returns:
            xr.Dataset: PIVPy dataset with velocity fields and coordinates
            
        Raises:
            IOError: If file cannot be read or parsed
            ValueError: If file format is invalid
        """
        pass
    
    def _set_default_attrs(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply standard PIVPy attributes to dataset
        
        Args:
            dataset (xr.Dataset): Dataset to modify
            
        Returns:
            xr.Dataset: Dataset with standard attributes applied
        """
        return set_default_attrs(dataset)


class InsightVECReader(PIVReader):
    """Reader for TSI Insight VEC files (TECPLOT format)"""
    
    def can_read(self, filepath: pathlib.Path) -> bool:
        """Check if file is an Insight VEC file"""
        filepath = pathlib.Path(filepath)
        if filepath.suffix.lower() not in ['.vec', '.txt']:
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline()
            return header.startswith('TITLE=')
        except (IOError, UnicodeDecodeError):
            return False
    
    def read_metadata(self, filepath: pathlib.Path) -> PIVMetadata:
        """Extract metadata from Insight VEC file"""
        filepath = pathlib.Path(filepath)
        variables, units, rows, cols, dt, frame, _ = parse_header(filepath)
        
        return PIVMetadata(
            pos_units=units[0] if units and len(units) > 0 else POS_UNITS,
            vel_units=units[2] if units and len(units) > 2 else VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=float(dt) if dt is not None else DELTA_T,
            variables=variables,
            rows=int(rows) if rows is not None else None,
            cols=int(cols) if cols is not None else None,
            frame=frame,
        )
    
    def read(self, filepath: pathlib.Path, **kwargs) -> xr.Dataset:
        """Load Insight VEC file"""
        filepath = pathlib.Path(filepath)
        metadata = self.read_metadata(filepath)
        
        # Use kwargs or metadata values
        rows = kwargs.get('rows', metadata.rows)
        cols = kwargs.get('cols', metadata.cols)
        delta_t = kwargs.get('delta_t', metadata.delta_t)
        frame = kwargs.get('frame', metadata.frame)
        
        return load_vec(filepath, rows=rows, cols=cols, delta_t=delta_t, frame=frame)


class OpenPIVReader(PIVReader):
    """Reader for OpenPIV text files (5 or 6 columns)"""
    
    def can_read(self, filepath: pathlib.Path) -> bool:
        """Check if file is an OpenPIV text file"""
        filepath = pathlib.Path(filepath)
        if filepath.suffix.lower() not in ['.txt', '.vec']:
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline()
            # OpenPIV files either have no header or don't start with known headers
            if header.startswith('TITLE=') or header.startswith('#DaVis'):
                return False
            
            # Try to parse first data line
            with open(filepath, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):
                        parts = stripped.split()
                        # OpenPIV has 5 or 6 numeric columns
                        if len(parts) in [5, 6]:
                            try:
                                [float(x) for x in parts]
                                return True
                            except ValueError:
                                return False
                        return False
            return False
        except (IOError, UnicodeDecodeError):
            return False
    
    def read_metadata(self, filepath: pathlib.Path) -> PIVMetadata:
        """Extract metadata from OpenPIV text file"""
        filepath = pathlib.Path(filepath)
        
        # Extract frame number from filename
        fname = filepath.stem.split(".")[0]
        try:
            frame = int(re.findall(r"\d+", fname)[-1])
        except (ValueError, IndexError):
            frame = 0
        
        return PIVMetadata(
            pos_units=POS_UNITS,
            vel_units=VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=DELTA_T,
            variables=['x', 'y', 'u', 'v'],
            rows=None,
            cols=None,
            frame=frame,
        )
    
    def read(self, filepath: pathlib.Path, **kwargs) -> xr.Dataset:
        """Load OpenPIV text file"""
        filepath = pathlib.Path(filepath)
        metadata = self.read_metadata(filepath)
        
        rows = kwargs.get('rows', metadata.rows)
        cols = kwargs.get('cols', metadata.cols)
        delta_t = kwargs.get('delta_t', metadata.delta_t)
        frame = kwargs.get('frame', metadata.frame)
        
        return load_openpiv_txt(filepath, rows=rows, cols=cols, delta_t=delta_t, frame=frame)


class Davis8Reader(PIVReader):
    """Reader for Davis8 ASCII format"""
    
    def can_read(self, filepath: pathlib.Path) -> bool:
        """Check if file is a Davis8 file"""
        filepath = pathlib.Path(filepath)
        if filepath.suffix.lower() not in ['.txt']:
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline()
            return header.startswith('#DaVis')
        except (IOError, UnicodeDecodeError):
            return False
    
    def read_metadata(self, filepath: pathlib.Path) -> PIVMetadata:
        """Extract metadata from Davis8 file"""
        filepath = pathlib.Path(filepath)
        variables, units, rows, cols, dt, frame, _ = parse_header(filepath)
        
        return PIVMetadata(
            pos_units=units[0] if units and len(units) > 0 else POS_UNITS,
            vel_units=units[2] if units and len(units) > 2 else VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=float(dt) if dt is not None else DELTA_T,
            variables=variables,
            rows=int(rows) if rows is not None else None,
            cols=int(cols) if cols is not None else None,
            frame=frame,
        )
    
    def read(self, filepath: pathlib.Path, **kwargs) -> xr.Dataset:
        """Load Davis8 file"""
        filepath = pathlib.Path(filepath)
        metadata = self.read_metadata(filepath)
        
        rows = kwargs.get('rows', metadata.rows)
        cols = kwargs.get('cols', metadata.cols)
        delta_t = kwargs.get('delta_t', metadata.delta_t)
        frame = kwargs.get('frame', metadata.frame)
        
        return load_davis8_txt(filepath, rows=rows, cols=cols, delta_t=delta_t, frame=frame)


class LaVisionVC7Reader(PIVReader):
    """Reader for LaVision VC7 binary files"""
    
    def can_read(self, filepath: pathlib.Path) -> bool:
        """Check if file is a VC7 file"""
        filepath = pathlib.Path(filepath)
        return filepath.suffix.lower() == '.vc7'
    
    def read_metadata(self, filepath: pathlib.Path) -> PIVMetadata:
        """Extract metadata from VC7 file"""
        filepath = pathlib.Path(filepath)
        
        # Extract frame number from filename
        fname = filepath.stem.split(".")[0]
        try:
            frame = int(re.findall(r"\d+", fname)[-1])
        except (ValueError, IndexError):
            frame = 0
        
        return PIVMetadata(
            pos_units=POS_UNITS,
            vel_units=VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=DELTA_T,
            variables=['x', 'y', 'u', 'v'],
            rows=None,
            cols=None,
            frame=frame,
        )
    
    def read(self, filepath: pathlib.Path, **kwargs) -> xr.Dataset:
        """Load VC7 file"""
        filepath = pathlib.Path(filepath)
        metadata = self.read_metadata(filepath)
        frame = kwargs.get('frame', metadata.frame)
        
        return load_vc7(filepath, frame=frame)


class PIVLabReader(PIVReader):
    """Reader for PIVLab MAT files (MATLAB HDF5 format)"""
    
    def can_read(self, filepath: pathlib.Path) -> bool:
        """Check if file is a PIVLab MAT file"""
        filepath = pathlib.Path(filepath)
        if filepath.suffix.lower() != '.mat':
            return False
        
        # Check if h5py is available
        try:
            import h5py
        except ImportError:
            return False
        
        # Try to open as HDF5 and check for resultslist
        try:
            with h5py.File(filepath, 'r') as f:
                return 'resultslist' in f
        except:
            return False
    
    def read_metadata(self, filepath: pathlib.Path) -> PIVMetadata:
        """Extract metadata from PIVLab file"""
        filepath = pathlib.Path(filepath)
        
        # Extract frame number from filename if present
        fname = filepath.stem.split(".")[0]
        try:
            frame = int(re.findall(r"\d+", fname)[-1])
        except (ValueError, IndexError):
            frame = 0
        
        return PIVMetadata(
            pos_units=POS_UNITS,
            vel_units=VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=DELTA_T,
            variables=['x', 'y', 'u', 'v'],
            rows=None,
            cols=None,
            frame=frame,
        )
    
    def read(self, filepath: pathlib.Path, **kwargs) -> xr.Dataset:
        """Load PIVLab file"""
        filepath = pathlib.Path(filepath)
        frame = kwargs.get('frame', None)
        
        return load_pivlab(filepath, frame=frame)


class PIVReaderRegistry:
    """Registry for managing PIV file readers
    
    This class maintains a list of available readers and provides
    auto-detection capabilities to select the appropriate reader
    for a given file.
    """
    
    def __init__(self):
        """Initialize the registry with built-in readers"""
        self._readers: List[PIVReader] = []
        self._register_builtin_readers()
    
    def _register_builtin_readers(self):
        """Register all built-in readers"""
        # Order matters: more specific readers first
        self.register(InsightVECReader())
        self.register(Davis8Reader())
        self.register(LaVisionVC7Reader())
        self.register(PIVLabReader())
        self.register(OpenPIVReader())  # Most generic, check last
    
    def register(self, reader: PIVReader):
        """Register a custom reader
        
        Args:
            reader (PIVReader): Reader instance to register
            
        Note:
            Custom readers are added at the beginning of the list,
            so they take precedence over built-in readers.
        """
        self._readers.insert(0, reader)
    
    def find_reader(self, filepath: pathlib.Path) -> Optional[PIVReader]:
        """Find appropriate reader for a file
        
        Args:
            filepath (pathlib.Path): Path to the file
            
        Returns:
            Optional[PIVReader]: Reader that can handle the file, or None
        """
        filepath = pathlib.Path(filepath)
        for reader in self._readers:
            if reader.can_read(filepath):
                return reader
        return None
    
    def get_readers(self) -> List[PIVReader]:
        """Get list of all registered readers
        
        Returns:
            List[PIVReader]: List of registered readers
        """
        return self._readers.copy()


# Global registry instance
_REGISTRY = PIVReaderRegistry()


def register_reader(reader: PIVReader):
    """Register a custom PIV reader
    
    Args:
        reader (PIVReader): Custom reader instance to register
        
    Example:
        >>> class MyCustomReader(PIVReader):
        ...     def can_read(self, filepath):
        ...         return filepath.suffix == '.custom'
        ...     def read_metadata(self, filepath):
        ...         return PIVMetadata()
        ...     def read(self, filepath, **kwargs):
        ...         # Custom read logic
        ...         return dataset
        >>> register_reader(MyCustomReader())
    """
    _REGISTRY.register(reader)


def read_piv(
    filepath: Union[str, pathlib.Path],
    format: Optional[str] = None,
    **kwargs
) -> xr.Dataset:
    """Read a PIV file with automatic format detection
    
    This is the main entry point for reading PIV files. It automatically
    detects the file format and uses the appropriate reader.
    
    Args:
        filepath (Union[str, pathlib.Path]): Path to the PIV file
        format (Optional[str]): Force specific format ('insight', 'openpiv', 
            'davis8', 'vc7', 'pivlab'). If None, auto-detect.
        **kwargs: Additional reader-specific options (rows, cols, delta_t, frame, etc.)
        
    Returns:
        xr.Dataset: PIVPy dataset with velocity fields and coordinates
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format cannot be detected or is unsupported
        IOError: If file cannot be read
        
    Examples:
        >>> # Auto-detect format
        >>> data = read_piv('velocity_field.vec')
        >>> 
        >>> # Force specific format
        >>> data = read_piv('field.txt', format='openpiv')
        >>> 
        >>> # With additional options
        >>> data = read_piv('field.vec', delta_t=0.001, frame=5)
    """
    filepath = pathlib.Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # If format is specified, use specific reader
    if format is not None:
        format = format.lower()
        reader_map = {
            'insight': InsightVECReader(),
            'openpiv': OpenPIVReader(),
            'davis8': Davis8Reader(),
            'vc7': LaVisionVC7Reader(),
            'pivlab': PIVLabReader(),
        }
        if format not in reader_map:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {list(reader_map.keys())}"
            )
        reader = reader_map[format]
        if not reader.can_read(filepath):
            warnings.warn(
                f"File {filepath} may not be in {format} format. "
                f"Attempting to read anyway..."
            )
        return reader.read(filepath, **kwargs)
    
    # Auto-detect format
    reader = _REGISTRY.find_reader(filepath)
    if reader is None:
        raise ValueError(
            f"Cannot determine format for file: {filepath}. "
            f"Try specifying the format explicitly with format='...' parameter."
        )
    
    return reader.read(filepath, **kwargs)


def read_directory(
    path: Union[str, pathlib.Path],
    pattern: str = "*",
    ext: str = ".vec",
    parallel: bool = False,
    **kwargs
) -> xr.Dataset:
    """Read all PIV files from a directory
    
    Args:
        path (Union[str, pathlib.Path]): Directory path
        pattern (str): Filename pattern (default: "*")
        ext (str): File extension (default: ".vec")
        parallel (bool): Use parallel processing (not yet implemented)
        **kwargs: Additional options passed to reader
        
    Returns:
        xr.Dataset: Combined dataset with time dimension
        
    Raises:
        IOError: If no files found or cannot read files
        
    Examples:
        >>> # Read all VEC files
        >>> data = read_directory('data/velocity_fields')
        >>> 
        >>> # Read with pattern
        >>> data = read_directory('data', pattern='Run*', ext='.vec')
        >>> 
        >>> # Preserve delta_t from files
        >>> data = read_directory('data', ext='.vec')
        >>> print(data.attrs['delta_t'])
    """
    path = pathlib.Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Find matching files
    files = sorted(path.glob(pattern + ext))
    
    if len(files) == 0:
        raise IOError(f"No files matching {pattern}{ext} in directory {path}")
    
    if parallel:
        warnings.warn("Parallel processing not yet implemented. Using sequential processing.")
    
    # Read first file to get metadata
    first_reader = _REGISTRY.find_reader(files[0])
    if first_reader is None:
        raise ValueError(f"Cannot determine format for file: {files[0]}")
    
    first_metadata = first_reader.read_metadata(files[0])
    
    # Read all files
    datasets = []
    for i, filepath in enumerate(files):
        dataset = read_piv(filepath, frame=i, **kwargs)
        datasets.append(dataset)
    
    # Combine datasets
    if len(datasets) > 0:
        combined = xr.concat(datasets, dim="t")
        # Preserve delta_t from last dataset (should be consistent)
        if hasattr(datasets[-1], 'attrs') and 'delta_t' in datasets[-1].attrs:
            combined.attrs["delta_t"] = datasets[-1].attrs["delta_t"]
        combined.attrs["files"] = [str(f) for f in files]
        return combined
    else:
        raise IOError("Could not read any files")


def save_piv(
    dataset: xr.Dataset,
    filepath: Union[str, pathlib.Path],
    format: str = "netcdf",
    **kwargs
) -> None:
    """Save PIV dataset to file
    
    Args:
        dataset (xr.Dataset): Dataset to save
        filepath (Union[str, pathlib.Path]): Output file path
        format (str): Output format ('netcdf', 'zarr', 'csv')
        **kwargs: Format-specific options
            For NetCDF: compression options (compression='zlib', complevel=4)
            For Zarr: chunking options
            For CSV: frame selection (frame=0)
            
    Raises:
        ValueError: If format is unsupported
        IOError: If cannot write to file
        
    Examples:
        >>> # Save as NetCDF with compression
        >>> save_piv(data, 'output.nc', format='netcdf')
        >>> 
        >>> # Save as Zarr
        >>> save_piv(data, 'output.zarr', format='zarr')
        >>> 
        >>> # Save single frame as CSV
        >>> save_piv(data, 'output.csv', format='csv', frame=0)
    """
    filepath = pathlib.Path(filepath)
    format = format.lower()
    
    if format == 'netcdf':
        # Default compression settings
        encoding = {}
        if kwargs.get('compression', True):
            comp_type = kwargs.get('comp_type', 'zlib')
            comp_level = kwargs.get('complevel', 4)
            for var in dataset.data_vars:
                encoding[var] = {'zlib': comp_type == 'zlib', 'complevel': comp_level}
        
        dataset.to_netcdf(filepath, encoding=encoding if encoding else None)
        
    elif format == 'zarr':
        import zarr
        dataset.to_zarr(filepath, **kwargs)
        
    elif format == 'csv':
        # Export single frame to CSV
        frame = kwargs.get('frame', 0)
        if 't' in dataset.dims:
            ds_frame = dataset.isel(t=frame)
        else:
            ds_frame = dataset
        
        # Convert to DataFrame
        df_list = []
        x_vals, y_vals = np.meshgrid(ds_frame.x.values, ds_frame.y.values, indexing='ij')
        df = pd.DataFrame({
            'x': x_vals.ravel(),
            'y': y_vals.ravel(),
            'u': ds_frame['u'].values.ravel(),
            'v': ds_frame['v'].values.ravel(),
        })
        if 'chc' in ds_frame:
            df['chc'] = ds_frame['chc'].values.ravel()
        
        df.to_csv(filepath, index=False)
        
    else:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Supported formats: 'netcdf', 'zarr', 'csv'"
        )
