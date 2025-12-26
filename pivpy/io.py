"""
This module contains functions to load data from various PIV software packages.
"""
import numpy as np
import xarray as xr
from pathlib import Path
import h5py
from typing import Union, Tuple
import re


def load_txt(filename: str) -> xr.Dataset:
    """
    Load PIV data from a text file.
    
    Parameters
    ----------
    filename : str
        Path to the text file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    
    # Reshape data
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    
    x = x.reshape((ny, nx))
    y = y.reshape((ny, nx))
    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :],
            'y': y[:, 0],
        }
    )
    
    return ds


def load_vec(filename: str) -> xr.Dataset:
    """
    Load PIV data from a VEC file (DaVis format).
    
    Parameters
    ----------
    filename : str
        Path to the VEC file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find header end
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('ZONE'):
            header_end = i + 1
            break
    
    # Read data
    data = []
    for line in lines[header_end:]:
        if line.strip():
            data.append([float(x) for x in line.split()])
    
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    
    # Reshape data
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    
    x = x.reshape((ny, nx))
    y = y.reshape((ny, nx))
    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :],
            'y': y[:, 0],
        }
    )
    
    return ds


def load_mat(filename: str) -> xr.Dataset:
    """
    Load PIV data from a MAT file (MATLAB format).
    
    Parameters
    ----------
    filename : str
        Path to the MAT file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    import scipy.io as sio
    
    mat = sio.loadmat(filename)
    
    # Try common variable names
    x = None
    y = None
    u = None
    v = None
    
    for key in mat.keys():
        if key.lower() in ['x', 'x_coord', 'xcoord']:
            x = mat[key]
        elif key.lower() in ['y', 'y_coord', 'ycoord']:
            y = mat[key]
        elif key.lower() in ['u', 'u_filt', 'ufilt']:
            u = mat[key]
        elif key.lower() in ['v', 'v_filt', 'vfilt']:
            v = mat[key]
    
    if x is None or y is None or u is None or v is None:
        raise ValueError("Could not find required variables in MAT file")
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :] if x.ndim > 1 else x,
            'y': y[:, 0] if y.ndim > 1 else y,
        }
    )
    
    return ds


def load_hdf5(filename: str) -> xr.Dataset:
    """
    Load PIV data from an HDF5 file.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    with h5py.File(filename, 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
        u = f['u'][:]
        v = f['v'][:]
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x,
            'y': y,
        }
    )
    
    return ds


def load_openpiv(filename: str) -> xr.Dataset:
    """
    Load PIV data from OpenPIV format.
    
    Parameters
    ----------
    filename : str
        Path to the file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    data = np.loadtxt(filename, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    
    # Reshape data
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    
    x = x.reshape((ny, nx))
    y = y.reshape((ny, nx))
    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :],
            'y': y[:, 0],
        }
    )
    
    return ds


def load_insight(filename: str) -> xr.Dataset:
    """
    Load PIV data from Insight format (TSI).
    
    Parameters
    ----------
    filename : str
        Path to the file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if 'X(mm)' in line or 'x(mm)' in line:
            data_start = i + 1
            break
    
    # Read data
    data = []
    for line in lines[data_start:]:
        if line.strip():
            data.append([float(x) for x in line.split()])
    
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    
    # Reshape data
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    
    x = x.reshape((ny, nx))
    y = y.reshape((ny, nx))
    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :],
            'y': y[:, 0],
        }
    )
    
    return ds


def load_pivware(filename: str) -> xr.Dataset:
    """
    Load PIV data from PIVware format.
    
    Parameters
    ----------
    filename : str
        Path to the file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    
    # Reshape data
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    
    x = x.reshape((ny, nx))
    y = y.reshape((ny, nx))
    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :],
            'y': y[:, 0],
        }
    )
    
    return ds


def load_pivlab(filename: str) -> xr.Dataset:
    """
    Load PIV data from PIVlab MATLAB format.
    
    This function loads PIV data exported from PIVlab (a MATLAB-based PIV software).
    It expects a .mat file containing specific variables produced by PIVlab.
    
    Parameters
    ----------
    filename : str
        Path to the PIVlab .mat file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data with coordinates and velocity components
        
    Raises
    ------
    IOError
        If file cannot be read or doesn't contain expected PIVlab structure
        
    Examples
    --------
    >>> ds = load_pivlab('pivlab_results.mat')
    >>> print(ds)
    """
    import scipy.io as sio
    
    try:
        mat = sio.loadmat(filename)
    except Exception as e:
        raise IOError(f"Could not read file {filename}: {e}")
    
    # PIVlab exports data with specific variable names
    # Try to find the main variables
    x = None
    y = None
    u = None
    v = None
    
    # Common PIVlab variable names
    for key in mat.keys():
        if key.lower() == 'x':
            x = mat[key]
        elif key.lower() == 'y':
            y = mat[key]
        elif key.lower() in ['u', 'u_filtered']:
            u = mat[key]
        elif key.lower() in ['v', 'v_filtered']:
            v = mat[key]
    
    if x is None or y is None or u is None or v is None:
        raise IOError("File doesn't contain expected PIVlab structure (x, y, u, v variables)")
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :] if x.ndim > 1 else x,
            'y': y[:, 0] if y.ndim > 1 else y,
        }
    )
    
    return ds


def load_pivview(filename: str) -> xr.Dataset:
    """
    Load PIV data from PIVview format.
    
    Parameters
    ----------
    filename : str
        Path to the file
        
    Returns
    -------
    xr.Dataset
        Dataset containing the PIV data
    """
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    
    # Reshape data
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    
    x = x.reshape((ny, nx))
    y = y.reshape((ny, nx))
    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    
    # Create dataset
    ds = xr.Dataset(
        {
            'u': (['y', 'x'], u),
            'v': (['y', 'x'], v),
        },
        coords={
            'x': x[0, :],
            'y': y[:, 0],
        }
    )
    
    return ds


def load_directory(directory: str, pattern: str = '*.txt', loader_func=None) -> xr.Dataset:
    """
    Load multiple PIV files from a directory and combine them into a single dataset.
    
    Parameters
    ----------
    directory : str
        Path to the directory containing PIV files
    pattern : str, optional
        File pattern to match (default: '*.txt')
    loader_func : callable, optional
        Function to use for loading individual files (default: load_txt)
        
    Returns
    -------
    xr.Dataset
        Dataset containing all PIV data with time dimension
    """
    from pathlib import Path
    
    if loader_func is None:
        loader_func = load_txt
    
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    
    if not files:
        raise ValueError(f"No files found matching pattern {pattern} in {directory}")
    
    # Load first file to get structure
    datasets = []
    for i, file in enumerate(files):
        ds = loader_func(str(file))
        ds = ds.expand_dims({'t': [i]})
        datasets.append(ds)
    
    # Combine all datasets
    combined = xr.concat(datasets, dim='t')
    
    return combined
