# -*- coding: utf-8 -*-

"""
Contains functions for reading flow fields in various formats
"""

import numpy as np
import xarray as xr
from glob import glob
import os
import re
try:
    import ReadIM
except:
    warnings.warn('ReadIM is not available, do not read IM7 these files')

default_units = ["pix", "pix", "pix/dt", "pix/dt"]
default_variables = ["x", "y", "u", "v", "s2n"]


def create_sample_field(rows=5, cols=8, frame=0, noise_sigma=1.0):
    """ creates a sample dataset for the tests """

    x = np.arange(32.0, (cols + 1) * 32.0, 32.0)
    y = np.arange(16.0, (rows + 1) * 16.0, 16.0)

    xm, ym = np.meshgrid(x, y)
    u = np.ones_like(xm) + np.linspace(0.0, 10.0, cols)
    v = (
        np.zeros_like(ym)
        + np.linspace(0.0, 1.0, rows).reshape(rows, 1)
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

    data = xr.Dataset({"u": u, "v": v, "chc": chc})

    data.attrs["variables"] = ["x", "y", "u", "v"]
    data.attrs["units"] = ["pix", "pix", "pix/dt", "pix/dt"]
    data.attrs["dt"] = 1.0
    data.attrs["files"] = ""

    return data


def create_sample_dataset(n=5, noise_sigma=1.0):
    """ using create_sample_field that has random part in it, create
    a sample dataset of length 'n' """

    data = []
    for i in range(n):
        data.append(create_sample_field(frame=i, noise_sigma=noise_sigma))

    combined = xr.concat(data, dim="t")
    combined.attrs["variables"] = ["x", "y", "u", "v"]
    combined.attrs["units"] = ["pix", "pix", "pix/dt", "pix/dt"]
    combined.attrs["dt"] = 1.0
    combined.attrs["files"] = ""

    return combined

def create_uniform_strain():
    return create_sample_field(noise_sigma=0.0)

    
def from_arrays(x, y, u, v, mask):
    """
        from_arrays(x,y,u,v,mask,frame=0)
        creates an xArray Dataset from 5 two-dimensional Numpy arrays
        of x,y,u,v and mask

        Input:
            x,y,u,v,mask = Numpy floating arrays, all the same size
        Output:
            data is a xAarray Dataset, see xarray for help
    """
    # create data structure of appropriate size
    data = create_sample_field(rows=x.shape[0], cols=x.shape[1])
    # assign arrays
    data["x"] = x[0, :]
    data["y"] = y[:, 0]
    data["u"] = xr.DataArray(u.T[:, :, np.newaxis], dims=("x", "y", "t"))
    data["v"] = xr.DataArray(v.T[:, :, np.newaxis], dims=("x", "y", "t"))
    data["chc"] = xr.DataArray(mask.T[:, :, np.newaxis], dims=("x", "y", "t"))

    return data

def from_df(df, frame=0, dt=1.0, filename=''):
    """
        from_df(x,y,u,v,mask,frame=0)
        creates an xArray Dataset from pandas dataframe with 5 columns

        Read the .txt files faster with pandas read_csv()

        %%time
        df = pd.read_csv(files_list[-1],delimiter='\t', 
                        names = ['x','y','u','v','mask'],header=0)
        from_df(df,filename=files_list[-1])

        is 3 times faster than the load_txt


        Input:
            x,y,u,v,mask = Numpy floating arrays, all the same size
        Output:
            data is a xAarray Dataset, see xarray for help
    """
    d = df.to_numpy()

    x = np.unique(d[:, 0])
    y = np.unique(d[:, 1])
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

    data = xr.Dataset({"u": u, "v": v, "chc": chc})

    data.attrs["variables"] = df.columns.to_list()
    data.attrs["units"] = ['pix','pix','pix/dt','pix/dt']
    data.attrs["dt"] = dt
    data.attrs["files"] = filename

    return data

def load_vec(
    filename,
    rows=None,
    cols=None,
    variables=default_variables,
    units=default_units,
    dt=1.0,
    frame=0,
):
    """
        load_vec(filename,rows=rows,cols=cols)
        Loads the VEC file (TECPLOT format by TSI Inc.),
        OpenPIV VEC or TXT formats
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
        variables, units, rows, cols, dt, frame = parse_header(filename)

    if rows is None:  # means no headers
        d = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4))
        x = unique(d[:, 0])
        y = unique(d[:, 1])
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

    data = xr.Dataset({"u": u, "v": v, "chc": chc})

    data.attrs["variables"] = variables
    data.attrs["units"] = units
    data.attrs["dt"] = dt
    data.attrs["files"] = filename

    return data


def load_directory(path, basename="*", ext=".vec"):
    """
    load_directory (path,basename='*', ext='*.vec')

    Loads all the files with the chosen sextension in the directory into a
    single xarray Dataset with variables and units added as attributes

    Input:
        directory : path to the directory with .vec, .txt or .VC7 files,
        period . can be dropped

    Output:
        data : xarray DataSet with dimensions: x,y,t and
               data arrays of u,v,
               attributes of variables and units


    See more: load_vec
    """
    files = sorted(glob(os.path.join(path, basename + ext)))
    if len(files) == 0:
        raise IOError(f"No files {basename+ext} in the directory {path} ")
    else:
        print(f"found {len(files)} files")

    data = []
    combined = []

    if ext.lower().endswith("vec"):
        variables, units, rows, cols, dt, frame = parse_header(files[0])

        for i, f in enumerate(files):
            data.append(
                load_vec(f, rows, cols, variables, units, dt, frame + i - 1)
            )

        if len(data) > 0:
            combined = xr.concat(data, dim="t")
            combined.attrs["variables"] = data[0].attrs["variables"]
            combined.attrs["units"] = data[0].attrs["units"]
            combined.attrs["dt"] = data[0].attrs["dt"]
            combined.attrs["files"] = files
    elif ext.lower().endswith("vc7"):
        frame = 1
        for i, f in enumerate(files):
            if basename == "B*":  # quite strange to have a specific name?
                time = int(f[-9:-4]) - 1
            else:
                time = i
            data.append(load_vc7(f, time))
        if len(data) > 0:
            combined = xr.concat(data, dim="t")
            combined.attrs = data[-1].attrs
    elif ext.lower().endswith("txt"):
        variables, units, rows, cols, dt, frame = parse_header(files[0])

        for i, f in enumerate(files):
            data.append(
                load_txt(f, rows, cols, variables, units, dt, frame + i - 1)
            )
        if len(data) > 0:
            combined = xr.concat(data, dim="t")
            combined.attrs["variables"] = data[0].attrs["variables"]
            combined.attrs["units"] = data[0].attrs["units"]
            combined.attrs["dt"] = data[0].attrs["dt"]
            combined.attrs["files"] = files
        else:
            raise IOError("Could not read the files")

    return combined


def parse_header(filename):
    """
    parse_header ( filename)
    Parses header of the file (.vec) to get the variables (typically X,Y,U,V)
    and units (can be m,mm, pix/dt or mm/sec, etc.), and the size of the
    dataset by the number of rows and columns.
    Input:
        filename : complete path of the file to read
    Returns:
        variables : list of strings
        units : list of strings
        rows : number of rows of the dataset
        cols : number of columns of the dataset
        dt   : time interval between the two PIV frames in microseconds
    """

    # defaults
    frame = 0

    # split path from the filename
    fname = os.path.basename(filename)
    # get the number in a filename if it's a .vec file from Insight
    if "." in fname[:-4]:  # day2a005003.T000.D000.P003.H001.L.vec
        frame = int(re.findall(r"\d+", fname.split(".")[0])[-1])
    elif "_" in filename[:-4]:
        frame = int(
            re.findall(r"\d+", fname.split("_")[1])[-1]
        )  # exp1_001_b.vec, .txt

    with open(filename) as fid:
        header = fid.readline()

    # if the file does not have a header, can be from OpenPIV or elsewhere
    # return None
    if header[:5] != "TITLE":
        return (
            ["x", "y", "u", "v"],
            ["pix", "pix", "pix/dt", "pix/dt"],
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

    # this is also important to know the time interval, dt
    ind1 = header.find("MicrosecondsPerDeltaT")
    dt = float(header[ind1:].split('"')[1])

    return (variables, units, rows, cols, dt, frame)


def get_units(filename):
    """
    get_units(filename)

    given a full path name to the .vec file will return the names
    of length and velocity units fallback option is all None. Uses
    parse_header function, see below.

    """

    # lUnits, velUnits, tUnits = 'pixel', 'pixel', 'dt'

    _, units, _, _, _, _ = parse_header(filename)

    if units == "":
        return "pix", "pix", "dt"

    lUnits = units[0]
    velUnits = units[2]

    if velUnits == "pixel":
        velUnits = velUnits + "/dt"  # make it similar to m/s

    tUnits = velUnits.split("/")[1]  # make it 's' or 'dt'

    return lUnits, velUnits, tUnits


def load_vc7(path, time=0):
    """
    input path for files format from davis tested for im7&vc7
    out put [X Y U V mask]
    valid only for 2d piv cases
    RETURN:
     in case of images (image type=0):
              X = scaled x-coordinates
              Y = scaled y-coordinates
              U = scaled image intensities
              v=0
              MASK=0
            in case of 2D vector fields (A.IType = 1,2 or 3):
              X = scaled x-coordinates
               Y = scaled y-coordinates
               U = scaled vx-components of vectors
               V = scaled vy-components of vectors

    """
    # you need to add clear to prevent data leaks
    buff, vatts = ReadIM.extra.get_Buffer_andAttributeList(path)
    v_array, buff1 = ReadIM.extra.buffer_as_array(buff)
    nx = buff.nx
    # nz = buff.nz # flake8 claims it's not used
    ny = buff.ny
    # set data range:
    baseRangeX = np.arange(nx)
    baseRangeY = np.arange(ny)
    # baseRangeZ = np.arange(nz)
    lhs1 = (
        baseRangeX + 0.5
    ) * buff.vectorGrid * buff.scaleX.factor + buff.scaleX.offset  # x-range
    lhs2 = (
        baseRangeY + 0.5
    ) * buff.vectorGrid * buff.scaleY.factor + buff.scaleY.offset  # y-range
    lhs3 = 0
    lhs4 = 0
    mask = 0
    if buff.image_sub_type <= 0:  # grayvalue image format
        [lhs1, lhs2] = np.meshgrid(lhs1, lhs2)
        lhs3 = v_array[0, :, :]
        lhs4 = v_array[1, :, :]
        Im = xr.DataArray(
            v_array,
            dims=("frame", "z", "x"),
            coords={"x": lhs1[0, :], "z": lhs2[:, 0], "frame": [0, 1]},
        )
        data = xr.Dataset({"Im": Im})

    elif buff.image_sub_type == 2:  # simple 2D vector format: (vx,vy)
        # Calculate vector position and components
        [lhs1, lhs2] = np.meshgrid(lhs1, lhs2)
        #    lhs1=np.transpose(lhs1)
        #    lhs2=np.transpose(lhs2)
        lhs3 = v_array[0, :, :] * buff.scaleI.factor + buff.scaleI.offset
        lhs4 = v_array[1, :, :] * buff.scaleI.factor + buff.scaleI.offset
        if buff.scaleY.factor < 0.0:
            lhs4 = -lhs4
        lhs3 = lhs3[:, :, np.newaxis]
        lhs4 = lhs4[:, :, np.newaxis]
        u = xr.DataArray(
            lhs3,
            dims=("z", "x", "t"),
            coords={"x": lhs1[0, :], "z": lhs2[:, 0], "t": [time]},
        )
        v = xr.DataArray(
            lhs4,
            dims=("z", "x", "t"),
            coords={"x": lhs1[0, :], "z": lhs2[:, 0], "t": [time]},
        )
        data = xr.Dataset({"u": u, "v": v})
    #    	plt.quiver(lhs1,lhs2,lhs3,lhs4);
    elif buff.image_sub_type == 3 or buff.image_sub_type == 1:
        # normal 2D vector format + peak: sel+4*(vx,vy) (+peak)
        # Calculate vector position and components
        [lhs1, lhs2] = np.meshgrid(lhs1, lhs2)
        #    lhs1=np.transpose(lhs1)
        #    lhs2=np.transpose(lhs2)
        lhs3 = lhs1 * 0
        lhs4 = lhs2 * 0
        # Get choice
        maskData = v_array[0, :, :].astype(np.int8)
        # Build best vectors from choice field
        for i in range(5):
            mask = maskData == (i + 1)
            if i < 4:  # get best vectors
                dat = v_array[2 * i + 1, :, :]
                lhs3[mask] = dat[mask]
                dat = v_array[2 * i + 2, :, :]
                lhs4[mask] = dat[mask]
            else:  # get interpolated vectors
                dat = v_array[7, :, :]
                lhs3[mask] = dat[mask]
                dat = v_array[8, :, :]
                lhs4[mask] = dat[mask]
        lhs3 = lhs3 * buff.scaleI.factor + buff.scaleI.offset
        lhs4 = lhs4 * buff.scaleI.factor + buff.scaleI.offset
        # Display vector field
        if buff.scaleY.factor < 0.0:
            lhs4 = -1 * lhs4
        lhs3 = lhs3.T[:, :, np.newaxis]
        lhs4 = lhs4.T[:, :, np.newaxis]
        chc = maskData.T[:, :, np.newaxis]
        u = xr.DataArray(
            lhs3,
            dims=("x", "y", "t"),
            coords={"x": lhs1[0, :], "y": lhs2[:, 0], "t": [time]},
        )
        v = xr.DataArray(
            lhs4,
            dims=("x", "y", "t"),
            coords={"x": lhs1[0, :], "y": lhs2[:, 0], "t": [time]},
        )
        chc = xr.DataArray(
            chc,
            dims=("x", "y", "t"),
            coords={"x": lhs1[0, :], "y": lhs2[:, 0], "t": [time]},
        )
        data = xr.Dataset({"u": u, "v": v, "chc": chc})
    if buff.image_sub_type > 0:
        data.attrs = ReadIM.extra.att2dict(vatts)
        data.attrs["variables"] = ["x", "y", "u", "v"]
        data.attrs["units"] = ["mm", "mm", "m/s", "m/s"]
        data.attrs["dt"] = int(data.attrs["FrameDt0"][:-3])
        data.attrs["files"] = path
    # clean memory
    ReadIM.DestroyBuffer(buff1)
    del buff1
    ReadIM.DestroyBuffer(buff)
    del buff
    ReadIM.DestroyAttributeListSafe(vatts)
    del vatts
    return data


def load_txt(
    filename,
    rows=None,
    cols=None,
    variables=default_variables,
    units=default_units,
    dt=1.0,
    frame=0,
):
    """
        load_vec(filename,rows=rows,cols=cols)
        Loads the VEC file (TECPLOT format by TSI Inc.), OpenPIV VEC or TXT
        formats
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
    if rows is None:  # means no headers
        d = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4))
        x = unique(d[:, 0])
        y = unique(d[:, 1])
        d = d.reshape(len(y), len(x), 5).transpose(1, 0, 2)
    else:
        d = np.genfromtxt(
            filename, skip_header=1, delimiter=",", usecols=(0, 1, 2, 3, 4)
        ).reshape(rows, cols, 5)
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
        u, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    v = xr.DataArray(
        v, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]}
    )
    chc = xr.DataArray(
        chc, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [frame]}
    )

    data = xr.Dataset({"u": u, "v": v, "chc": chc})

    data.attrs["variables"] = variables
    data.attrs["units"] = units
    data.attrs["dt"] = dt
    data.attrs["files"] = filename

    return data


def unique(array):
    """ Returns not sorted unique """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]
