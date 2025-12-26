"""
Defines input/output functions of PIVPy

see read_pivpy()
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

from pathlib import Path
import pathlib
import os
import glob
import re
import warnings
from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
