import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example using the 4 column files
    """)
    return


@app.cell
def _():
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    from importlib.resources import files

    return files, np, plt, xr


@app.cell
def _(files):
    filename = files("pivpy").joinpath("data/PIV_Challenge/B00001.txt")
    return (filename,)


@app.cell
def _(filename):
    with open(filename) as f:
        print(f.readline()) #header
        print(f.readline()) #with commas
        print(f.readline().replace(',','.')) #replace commas by dots
    return


@app.cell
def _(filename, np):
    import builtins

    def _to_str(x):
        return x.decode() if isinstance(x, (bytes, bytearray)) else builtins.str(x)

    c = lambda x: float((_to_str(x)).replace(',', '.') or -999)
    tmp = np.genfromtxt(filename, skip_header=1, converters={0: c, 1: c, 2: c, 3: c})
    x, y, u, v = tmp[:, 0], tmp[:, 1], tmp[:, 2], tmp[:, 3]
    return builtins, u, v, x, y


@app.cell
def _(plt, u, v, x, y):
    plt.quiver(x,y,u,v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the following cell only explains what happened here:
    """)
    return


@app.cell
def _():
    from locale import setlocale, atof, LC_NUMERIC

    setlocale(LC_NUMERIC, '') # set to your default locale; for me this is
    # 'English_Canada.1252'. Or you could explicitly specify a locale in which floats
    # are formatted the way that you describe, if that's not how your locale works :)
    atof('123,456') # 123456.0
    # To demonstrate, let's explicitly try a locale in which the comma is a
    # decimal point:
    # setlocale(LC_NUMERIC, 'French_Canada.1252')
    result = atof('123,456') # 123.456
    return (result,)


@app.cell
def _():
    from pivpy import io, pivpy, graphics

    return graphics, io


@app.cell
def _(np, x, y):
    rows = np.unique(y).shape[0]
    cols = np.unique(x).shape[0]
    return cols, rows


@app.cell
def _(cols, rows, u, v, x, y):
    x1 = x.reshape(rows,cols)
    y1 = y.reshape(rows,cols)
    u1 = u.reshape(rows,cols)
    v1 = v.reshape(rows,cols)
    return u1, v1, x1, y1


@app.cell
def _(plt, u1, v1, x1, y1):
    plt.quiver(x1,y1,u1,v1)
    return


@app.cell
def _(io, np, u1, v1, x1, y1):
    d = io.from_arrays(x1,y1,u1,v1,np.ones_like(u1))
    return (d,)


@app.cell
def _(d, graphics):
    graphics.quiver(d.isel(t=0), scalingFactor=50);
    return


@app.function
def parse_header_davis816(filename):
    
    with open(filename) as f:
        header = f.readline() #header
        
    indp = header.find('"position"')+10
    indv = header.find('"velocity"')+10

    ind1 = header[indp:].find('"')
    ind2 = header[indp+ind1+1:].find('"')
    lUnits = header[indp+ind1+1:indp+ind1+ind2+1]

    ind1 = header[indv:].find('"')
    ind2 = header[indv+ind1+1:].find('"')
    velUnits = header[indv+ind1+1:indv+ind1+ind2+1]
    
    return (lUnits, velUnits)


@app.cell
def _(filename):
    len_units,vel_units = parse_header_davis816(filename)
    return len_units, vel_units


@app.cell
def _(len_units, vel_units):
    print(len_units,vel_units)
    return


@app.cell
def _(d, len_units, vel_units):
    d.attrs['units'] = [len_units, len_units, vel_units, vel_units]
    return


@app.cell
def _(d, graphics):
    graphics.quiver(d, scalingFactor=50);
    return


@app.cell
def _(builtins, io, np):
    def _to_str(x):
        return x.decode() if isinstance(x, (bytes, bytearray)) else builtins.str(x)
    convert = lambda x: float(_to_str(x).replace(',', '.') or -999)

    def load_txt_davis816(filename, frame=0):
        tmp = np.genfromtxt(filename, skip_header=1, converters={0: convert, 1: convert, 2: convert, 3: convert})
        x, y, u, v = (tmp[:, 0], tmp[:, 1], tmp[:, 2], tmp[:, 3])
        rows = np.unique(y).shape[0]
        cols = np.unique(x).shape[0]
        x1 = x.reshape(rows, cols)
        y1 = y.reshape(rows, cols)
        u1 = u.reshape(rows, cols)
        v1 = v.reshape(rows, cols)
        d = io.from_arrays(x1, y1, u1, v1, np.ones_like(u1))
        len_units, vel_units = parse_header_davis816(filename)
        d.attrs['units'] = [len_units, len_units, vel_units, vel_units]
        d['t'] = d['t'] + frame
        return d  # set frame

    return (load_txt_davis816,)


@app.cell
def _(load_txt_davis816, load_vc7, load_vec, parse_header, xr):
    from glob import glob
    import os

    def load_directory_davis816(path, basename='*',ext='.txt', soft='davis816'):
        """ 
        load_directory (path,basename='*', ext='*.txt')

        Loads all the files with the chosen sextension in the directory into a single
        xarray Dataset with variables and units added as attributes

        Input: 
            directory : path to the directory with .vec, .txt or .VC7 files
            basename  : for directories with different sets of runs, add some string, 'B00*'
            ext : string, with a dot: '.txt'
            soft : default is None, optional ['openpiv','davis','davis816']
        

        Output:
            data : xarray DataSet with dimensions: x,y,t and 
                   data arrays of u,v,
                   attributes of variables and units


        See more: load_vec
        """
        files  = sorted(glob(os.path.join(path,basename+ext)))
        data = []

        if ext == '.vec':
            variables, units, rows, cols, dt, frame = parse_header(files[0])

            for i,f in enumerate(files):
                data.append(load_vec(f,rows,cols,variables,units,dt,frame+i-1))

            if len(data) > 0:
                combined = xr.concat(data, dim='t')
                combined.attrs['variables'] = data[0].attrs['variables']
                combined.attrs['units'] = data[0].attrs['units']
                combined.attrs['dt'] = data[0].attrs['dt']
                combined.attrs['files'] = files
        elif ext.lower() == '.vc7':
            frame = 1
            for i,f in enumerate(files):
                if basename=='B*':
                    time=int(f[-9:-4])-1
                else:
                    time=i
                data.append(load_vc7(f,time))
            if len(data) > 0:
                combined = xr.concat(data, dim='t')
                combined.attrs = data[-1].attrs
            
        elif ext.lower() == '.txt' and soft.lower() == 'davis816':
            frame = 1
            for i,f in enumerate(files):
                data.append(load_txt_davis816(f,i))
            if len(data) > 0:
                combined = xr.concat(data, dim='t')
                combined.attrs = data[-1].attrs

        return combined

    return (load_directory_davis816,)


@app.cell
def _(filename, load_txt_davis816):
    d_1 = load_txt_davis816(filename, 25)
    d_1
    return


@app.cell
def _(files, load_directory_davis816):
    ds = load_directory_davis816(path = files("pivpy").joinpath("data/PIV_Challenge"), basename='B*',ext='.txt', soft='davis816')
    return (ds,)


@app.cell
def _(ds, graphics):
    graphics.quiver(ds.isel(t=1),5, scalingFactor=50)
    return


if __name__ == "__main__":
    app.run()
