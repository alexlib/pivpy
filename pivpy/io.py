"""
This module contains functions for reading and writing PIV data files.
"""

import pathlib
import warnings
from typing import Union, Optional, List, Tuple
import numpy as np
import xarray as xr

try:
    from lvpyio import read_buffer
except ImportError:
    read_buffer = None


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
        raise ImportError("lvpyio is required to read VC7 files. Install it with: pip install lvpyio")
    
    buffer = read_buffer(str(filename))
    
    # Rest of the function implementation would follow here
    # (I'm only showing the modified beginning as requested)
    
    return dataset
