# -*- coding: utf-8 -*-
"""Created on Tue Apr 18 10:37:50 2017.

@author: alexlib
"""

from pathlib import Path
import re
import warnings
import numpy as np
from numpy import loadtxt

try:
    from lvpyio import read_buffer
except ImportError:
    read_buffer = None
    warnings.warn("lvpyio is not installed, use pip install lvpyio")
    warnings.warn("lvreader is not installed, use pip install lvpyio")

try:
    import h5py
except ImportError:
    warnings.warn("h5py is not installed, use pip install h5py to read PIVLab MAT files")


# Defaults
POS_UNITS: str = "pix"  # or mm, m, after scaling
TIME_UNITS: str = "frame"  # "frame" if not scaled, can become 'sec' or 'msec', 'usec'
# after scaling can be m/s, mm/s
VEL_UNITS: str = POS_UNITS  # default is displacement in pix
DELTA_T: np.float64 = 0.0  # default is 0. i.e. uknown, can be any float value


def parse_header(filename):
    """Parse INSIGHT header of the .vec file, return dictionary.

    Parameters
    ----------
    filename : str or Path
        path to the file with .vec extension, of VEC format INSIGHT 3G

    Returns
    -------
    dict : dictionary with the key,value from the header

    Examples
    --------
    >>> dict = parse_header('Camera1-01.vec')
    """
    # read all the header into a dict key,value
    # #TITLE:
    # PIVREC - PIV Recordings, Processing and Analysis
    # #HEADER_LINES:
    # 26
    # #FIRST_FRAME:
    # 1
    # #LAST_FRAME:
    # 1
    # #VELOCITY_COMPONENT:
    # 2
    # #DATASET:
    # 3D_Dataset_tomo
    # #PROCESSED_IMAGE_A:
    # /gpfs0/home/rjd/avi/test4/image_1_a.bmp
    # #PROCESSED_IMAGE_B:
    # /gpfs0/home/rjd/avi/test4/image_1_b.bmp
    # #DATE_PROCESSED:
    # 11:19:14  8 Jan 13
    # #INTERROGATION_METHOD:
    # Multigrid, Iteration
    # #VALIDATION_METHOD:
    # Normalized-Median
    # #PIV_SOFTWARE:
    # PIVREC_3G_2013-01-08
    # #COORDINATE_SYSTEM:
    # Fixed
    # #GRID_TYPE:
    # Uniform
    # #X_RESOLUTION:
    # 136
    # #Y_RESOLUTION:
    # 102
    # #XINF:
    # -4.5000e+01
    # #YINF:
    # -3.4000e+01
    # #XSUP:
    # 8.9500e+01
    # #YSUP:
    # 6.6500e+01
    # #PASS:
    # 1
    # #COLUMN_ORDER:
    # X Y U V Choice Correlation Mask
    # #COLUMN_UNITS:
    # mm mm m/s m/s None None None
    # I	J	U	V	Choice	Correlation	Mask

    header = {}
    with open(filename, encoding="latin-1") as f:
        for _ in range(30):  # maximum 30 header lines
            line = f.readline()
            if not line.startswith("#"):
                break
            tmp = line.split(":")
            if len(tmp) > 1:
                header[tmp[0].replace("#", "")] = ":".join(tmp[1:]).strip()

    return header


def load_vec(
    filename,
    rows=None,
    cols=None,
    skiprows=0,
    variables=None,
):
    """Load VEC file (INSIGHT format).

    Parameters
    ----------
    filename : str or Path
        path to the file with .vec extension, of VEC format INSIGHT 3G

    rows : int, optional
        is the number of rows in the vector field, default is None

    cols : int, optional
        number of columns in the vector field, default is None

    skiprows : int, optional
        is the number of non-header rows to skip, default is 0

    variables : list of str, optional
        variables to load, default is None

    Returns
    -------
    tuple: (x, y, u, v, ...)
        np.arrays of x, y meshgrid and velocity fields u, v or whatever
        else columns are

    Examples
    --------
    >>> x, y, u, v = load_vec('Camera1-01.vec')
    """
    filename = Path(filename)

    header = parse_header(filename)
    # print(header)

    if "X_RESOLUTION" in header:
        cols = int(header["X_RESOLUTION"])
    if "Y_RESOLUTION" in header:
        rows = int(header["Y_RESOLUTION"])
    if "HEADER_LINES" in header:
        skiprows = int(header["HEADER_LINES"])

    if rows is None or cols is None:
        raise ValueError(
            "Either provide rows and cols or use file with INSIGHT header"
        )

    # if header['GRID_TYPE'] == 'Half-Staggered':
    #     raise ValueError('Half-Staggered grid is not yet supported')

    if variables is None:
        d = loadtxt(filename, skiprows=skiprows, unpack=True)
    else:
        # get variable names from the header
        names = [
            i.strip()
            for i in re.split(r"[\t,\s]+", header["COLUMN_ORDER"].strip())
        ]

        # get indices of variables
        indices = tuple(names.index(v) for v in variables)

        d = loadtxt(
            filename, skiprows=skiprows, usecols=indices, unpack=True
        )

    # reshape using order F for Fortran or Matlab
    # column major order
    return tuple(np.reshape(a, (rows, cols), order="F") for a in d)


def load_openpiv_txt(filename, rows=None, cols=None):
    """Load OpenPIV TXT output file.

    Parameters
    ----------
    filename : str or Path
        path to the file with .txt extension,
    rows : int, optional
        number of rows, default is None
    cols : int, optional
        number of columns, default is None

    Returns
    -------
    x,y,u,v,mask : np.ndarray
        Meshgrid and velocity fields

    Examples
    --------
    >>> x,y,u,v,mask = load_openpiv_txt('exp1_001_a.txt')

    """
    # header is from openpiv.tools.save
    # x,y,u,v,mask
    # check if cols, rows are in the first two lines:
    # #0
    # #23,23
    # or
    # #cols, rows
    # #23, 23

    with open(filename) as f:
        line = f.readline()
        if "," in line:
            cols, rows = [int(i) for i in line.split(",")]
        else:
            line = f.readline()
            if "," in line:
                cols, rows = [int(i) for i in line.split(",")]

    d = loadtxt(
        filename, skiprows=0, usecols=(0, 1, 2, 3, 4), unpack=True
    )
    return tuple(np.reshape(a, (rows, cols)) for a in d)


def load_davis8_txt(filename, frame: int = 0):
    """Load LaVision DaVis 8 TXT file (as-exported).

    Parameters
    ----------
    filename : str or Path
        path to the file with .txt extension, as-exported from DaVis 8.x
    frame : int, optional
        frame number to load, default is 0

    Returns
    -------
    x,y,u,v : np.ndarray
        Meshgrid and velocity fields

    Examples
    --------
    >>> x,y,u,v = load_davis8_txt('B0001.txt')

    """
    filename = Path(filename)

    # DaVis 8.x.x TXT export format has variable number of rows in header
    # read all the header into a dict key,value

    header = {}
    with open(filename, encoding="latin-1") as f:
        for i in range(100):  # maximum 100 lines
            line = f.readline()
            if not line.startswith("#"):
                break
            tmp = line.split(":")
            if len(tmp) > 1:
                header[tmp[0].replace("#", "").strip()] = ":".join(
                    tmp[1:]
                ).strip()

    # print(header)

    # get number of frames:
    # #Datasets: 23
    datasets = int(header["Datasets"])

    # get column names
    # #Frame_0001	X	Y	U	V
    # #[Frame],[mm],[mm],[m/s],[m/s]

    # first line is: "Frame_0001	X	Y	U	V"
    # second line is units "#[Frame],[mm],[mm],[m/s],[m/s]"
    # remove all [] and split by comma:
    # variables = [i.strip() for i in \
    # header[f'Frame_{frame+1:04d}\tX\tY\tU\tV\tChoice\tMask'].split(',')]
    # variables = [i.strip() for i in header[header.keys()[6]].split(',')]
    # get 7th key: list(header.keys())[6]
    # variables = [i.strip() for i in \
    #   header[list(header.keys())[6]].replace('[','').replace(']','').split(',')]
    # usecols = tuple(variables.index(v) for v in variables)
    # print(variables)

    cols, rows = [int(i) for i in header["Size"].split("x")]

    # read the data
    # Davis 8 txt file has the frame number in the first column
    # the first frame is 0
    # X and Y positions are in columns, velocity in rows
    # Frame	X	Y	U	V
    # 1	-44.500	-33.500	-1.168	1.013
    # X [mm], Y [mm], Z [mm] and U [m/s], V [m/s]

    # read the file, skip header lines
    d = loadtxt(
        filename,
        skiprows=i,
        usecols=(0, 1, 2, 3, 4),
        unpack=True,
        max_rows=datasets * rows * cols,
    )

    # get the frame
    # d[0] is the frame number
    # d[1] is x
    # d[2] is y
    # d[3] is u
    # d[4] is v

    if frame == 0:  # convert all frames
        # reshape the data
        d = tuple(
            np.reshape(a, (datasets, rows, cols), order="C") for a in d[1:]
        )
    else:  # get the frame
        ind = d[0] == frame
        d = tuple(np.reshape(a[ind], (rows, cols), order="C") for a in d[1:])

    return d


def create_path(filename):
    """Create all the path to filename if not exists."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)


def save_vec(
    filename,
    x,
    y,
    u,
    v,
    s2n=None,
):
    """Save vector field to INSIGHT format .vec file.

    Parameters
    ----------
    filename : str or Path
        path to the file with .vec extension

    x,y : np.ndarray
        meshgrid

    u,v : np.ndarray
        velocity fields

    Examples
    --------
    >>> save_vec('field_001.vec', x,y,u,v)

    """
    create_path(filename)

    rows, cols = u.shape

    # write header:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("TITLE: pivpy\n")
        f.write("#HEADER_LINES: 23\n")
        f.write("#FIRST_FRAME: 1\n")
        f.write("#LAST_FRAME: 1\n")
        f.write("#VELOCITY_COMPONENT: 2\n")
        f.write("#DATASET: pivpy\n")
        f.write("#PROCESSED_IMAGE_A: image_a.bmp\n")
        f.write("#PROCESSED_IMAGE_B: image_b.bmp\n")
        f.write("#DATE_PROCESSED: unknown\n")
        f.write("#INTERROGATION_METHOD: unknown\n")
        f.write("#VALIDATION_METHOD: unknown\n")
        f.write("#PIV_SOFTWARE: pivpy\n")
        f.write("#COORDINATE_SYSTEM: Fixed\n")
        f.write("#GRID_TYPE: Uniform\n")
        f.write(f"#X_RESOLUTION: {cols}\n")
        f.write(f"#Y_RESOLUTION: {rows}\n")
        f.write(f"#XINF: {x.min()}\n")
        f.write(f"#YINF: {y.min()}\n")
        f.write(f"#XSUP: {x.max()}\n")
        f.write(f"#YSUP: {y.max()}\n")
        f.write("#PASS: 1\n")
        f.write("#COLUMN_ORDER: X Y U V\n")
        f.write("#COLUMN_UNITS: mm mm m/s m/s\n")
        f.write("#I\tJ\tU\tV\n")

        # write data
        for i in range(cols):
            for j in range(rows):
                f.write(
                    f"{x[j, i]}\t{y[j, i]}\t{u[j, i]}\t{v[j, i]}\n"
                )


def load_vc7(filename, frame=0):
    """Load LaVision VC7 file.

    Parameters
    ----------
    filename : str or Path
        path to the file with .vc7 extension

    frame : int, optional
        frame number to load, default is 0

    Returns
    -------
    x,y,u,v : np.ndarray
        Meshgrid and velocity fields

    Examples
    --------
    >>> x,y,u,v = load_vc7('B0001.vc7')


    References
    ----------
    Uses the package lvpyio, see https://github.com/OpenPIV/lvpyio
    created by Theo Käufer (@thkaf on Github) based on the ReadIM package
    by Julien Heymes (@rgranted-hub on Github)

    Note
    ----
    The VC7 file format is a binary format from LaVision DaVis 7.x
    If your installation does not have lvpyio, install it using pip:
        pip install lvpyio

    """
    if read_buffer is None:
        raise ImportError(
            "lvpyio is required to read VC7 files. Install it with: pip install lvpyio"
        )

    buffer = read_buffer(str(filename))
    u, v = buffer.frames[frame]
    (rows, cols) = u.shape
    x, y = np.meshgrid(buffer.scales[0], buffer.scales[1])

    # cut the u, v to the size of x, y
    u = u[: x.shape[0], : x.shape[1]]
    v = v[: y.shape[0], : y.shape[1]]

    return x, y, u, v


def from_arrays(
    x,
    y,
    u,
    v,
    mask=None,
):
    """Create a dictionary from arrays.

    Parameters
    ----------
    x,y : np.ndarray
        meshgrid
    u,v : np.ndarray
        velocity fields
    mask : np.ndarray, optional
        mask, default is None

    Returns
    -------
    d : dict
        dictionary with keys 'x', 'y', 'u', 'v', 'mask'

    Examples
    --------
    >>> d = from_arrays(x,y,u,v,mask)

    """
    d = {"x": x, "y": y, "u": u, "v": v}
    if mask is not None:
        d["mask"] = mask
    return d


def load_directory(
    directory,
    basename="",
    ext="txt",
    rows=None,
    cols=None,
    variables=None,
):
    """Load directory of files with the same extension.

    Parameters
    ----------
    directory : str or Path
        path to the directory
    basename : str, optional
        basename of the files, default is empty string
    ext : str, optional
        extension of the files, default is 'txt'
    rows : int, optional
        number of rows, default is None
    cols : int, optional
        number of columns, default is None
    variables : list of str, optional
        variables to load, default is None

    Returns
    -------
    list : list
        list of tuples (x,y,u,v,...)

    Examples
    --------
    >>> data = load_directory('../test1', basename='B', ext='vec')
    """
    directory = Path(directory)
    files = sorted(directory.glob(f"{basename}*.{ext}"))

    if ext == "vec":
        return [
            load_vec(f, rows=rows, cols=cols, variables=variables)
            for f in files
        ]
    elif ext == "txt":
        return [load_openpiv_txt(f, rows=rows, cols=cols) for f in files]

    raise ValueError(f"Unknown extension {ext}")


def create_sample_field(rows=21, cols=31):
    """Create a sample field - an artificial PIV vector field.

    Parameters
    ----------
    rows : int, optional
        number of rows, default is 21
    cols : int, optional
        number of columns, default is 31

    Returns
    -------
    x,y,u,v,s2n : np.ndarray
        Meshgrid, velocity fields and signal-to-noise ratio

    Examples
    --------
    >>> x,y,u,v = create_sample_field()

    """
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    u = np.ones_like(x)
    v = np.zeros_like(x)

    return x.astype(float), y.astype(float), u.astype(float), v.astype(float)


def create_sample_field_extended(
    rows=21, cols=31, u_mean=1.0, v_mean=0.0, u_std=0.1, v_std=0.1
):
    """Create a sample field - an artificial PIV vector field with noise.

    Parameters
    ----------
    rows : int, optional
        number of rows, default is 21
    cols : int, optional
        number of columns, default is 31
    u_mean : float, optional
        mean velocity in u, default is 1.0
    v_mean : float, optional
        mean velocity in v, default is 0.0
    u_std : float, optional
        standard deviation of u, default is 0.1
    v_std : float, optional
        standard deviation of v, default is 0.1

    Returns
    -------
    x,y,u,v : np.ndarray
        Meshgrid and velocity fields

    Examples
    --------
    >>> x,y,u,v = create_sample_field_extended()

    """
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u_mean * np.ones_like(x) + u_std * np.random.randn(*x.shape)
    v = v_mean * np.ones_like(x) + v_std * np.random.randn(*x.shape)

    return x.astype(float), y.astype(float), u.astype(float), v.astype(float)


def add_nans(data, num_nans=10):
    """Add NaNs to a data array.

    Parameters
    ----------
    data : np.ndarray
        data array
    num_nans : int, optional
        number of NaNs to add, default is 10

    Returns
    -------
    np.ndarray
        data array with NaNs

    Examples
    --------
    >>> x,y,u,v = create_sample_field()
    >>> u = add_nans(u)

    """
    shape = data.shape
    ind = np.random.randint(0, shape[0] * shape[1], size=num_nans)
    data.flat[ind] = np.nan
    return data


def create_sample_dataset(num_frames=10, rows=21, cols=31):
    """Create a sample dataset - an artificial PIV dataset.

    Parameters
    ----------
    num_frames : int, optional
        number of frames, default is 10
    rows : int, optional
        number of rows, default is 21
    cols : int, optional
        number of columns, default is 31

    Returns
    -------
    tuple: (x, y, u, v)
        np.arrays of x, y meshgrid and velocity fields u, v
        with additional time dimension

    Examples
    --------
    >>> x,y,u,v = create_sample_dataset()

    """
    x, y, u, v = create_sample_field(rows=rows, cols=cols)
    x = np.repeat(x[np.newaxis, :, :], num_frames, axis=0)
    y = np.repeat(y[np.newaxis, :, :], num_frames, axis=0)
    u = np.repeat(u[np.newaxis, :, :], num_frames, axis=0)
    v = np.repeat(v[np.newaxis, :, :], num_frames, axis=0)

    return x, y, u, v


def create_sample_dataset_extended(
    num_frames=10,
    rows=21,
    cols=31,
    u_mean=1.0,
    v_mean=0.0,
    u_std=0.1,
    v_std=0.1,
):
    """Create a sample dataset - an artificial PIV dataset with noise.

    Parameters
    ----------
    num_frames : int, optional
        number of frames, default is 10
    rows : int, optional
        number of rows, default is 21
    cols : int, optional
        number of columns, default is 31
    u_mean : float, optional
        mean velocity in u, default is 1.0
    v_mean : float, optional
        mean velocity in v, default is 0.0
    u_std : float, optional
        standard deviation of u, default is 0.1
    v_std : float, optional
        standard deviation of v, default is 0.1

    Returns
    -------
    tuple: (x, y, u, v)
        np.arrays of x, y meshgrid and velocity fields u, v
        with additional time dimension

    Examples
    --------
    >>> x,y,u,v = create_sample_dataset_extended()

    """
    x, y, u, v = create_sample_field_extended(
        rows=rows, cols=cols, u_mean=u_mean, v_mean=v_mean, u_std=u_std, v_std=v_std
    )
    x = np.repeat(x[np.newaxis, :, :], num_frames, axis=0)
    y = np.repeat(y[np.newaxis, :, :], num_frames, axis=0)
    u = np.repeat(u[np.newaxis, :, :], num_frames, axis=0)
    v = np.repeat(v[np.newaxis, :, :], num_frames, axis=0)

    return x, y, u, v


def loadvec(
    filename,
    rows=None,
    cols=None,
    skiprows=0,
    variables=None,
):
    """Load VEC file (INSIGHT format), compatibility with openpiv-python.

    Parameters
    ----------
    filename : str or Path
        path to the file with .vec extension, of VEC format INSIGHT 3G

    rows : int, optional
        is the number of rows in the vector field, default is None

    cols : int, optional
        number of columns in the vector field, default is None

    skiprows : int, optional
        is the number of non-header rows to skip, default is 0

    variables : list of str, optional
        variables to load, default is None

    Returns
    -------
    tuple: (x, y, u, v, ...)
        np.arrays of x, y meshgrid and velocity fields u, v or whatever
        else columns are

    Examples
    --------
    >>> x, y, u, v = loadvec('Camera1-01.vec')
    """
    return load_vec(filename, rows, cols, skiprows, variables)


def savevec(
    filename,
    x,
    y,
    u,
    v,
    s2n=None,
):
    """Save VEC file (INSIGHT format), compatibility with openpiv-python.

    Parameters
    ----------
    filename : str or Path
        path to the file with .vec extension

    x,y : np.ndarray
        meshgrid

    u,v : np.ndarray
        velocity fields

    Examples
    --------
    >>> savevec('field_001.vec', x,y,u,v)

    """
    save_vec(filename, x, y, u, v, s2n)
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
