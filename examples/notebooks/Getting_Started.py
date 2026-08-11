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
    # PIVpy getting started notebook:

    This notebook shows an example of how one can use vecpy in order to load manipulate and display analyzed PIV data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### step 1 - import pivpy and dependencies

    here we import the package code so that we can use it next
    """)
    return


@app.cell
def _():
    import os, sys 
    # sys.path.append(os.path.abspath('../'))

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo

    from pivpy import io, pivpy, graphics

    # for the sake of this tutorial, ignore warnings
    # import warnings
    # warnings.filterwarnings('ignore')
    return graphics, io, plt, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### step 2 - load the tests data

    In order to load the data, first we need to set up the path to the data directory. Following that we need to get a list of files names that we would like to view/analyze. Finally we very quickly load the data in to a list of vec instances.
    """)
    return


@app.cell
def _():
    # pointer to the directory with the data
    import importlib.resources as importlib_resources
    path_to_data  = importlib_resources.files('pivpy') / 'data'
    # list the directory
    # os.listdir(path_to_data)
    return (path_to_data,)


@app.cell
def _(io, path_to_data):
    # let's read only the files from the Run* 
    data = io.load_directory(path_to_data / 'Insight') # you can add also: basename='day2a*',ext='.vec')
    return (data,)


@app.cell
def _(data):
    # let's check if it's read:
    data.attrs['files']
    return


@app.cell
def _(data, plt):
    plt.quiver(data.x, data.y, data.u.isel(t=0), data.v.isel(t=0),scale=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### step 3 - plot some arrows

    first things first - show a quiver plot
    """)
    return


@app.cell
def _(data, graphics):
    _fig, _ax = graphics.quiver(data.isel(t=0), nthArr=2, arrScale=20, add_guide=False)
    return


@app.cell
def _(data, graphics):
    _fig, _ax = graphics.quiver(data.isel(t=0), nthArr=3, arrScale=5)
    return


@app.cell
def _(data, plt):
    _tmp = data.isel(t=0)
    plt.quiver(_tmp.x, _tmp.y, _tmp.u.T, _tmp.v.T, scale=1)
    return


@app.cell
def _(io, path_to_data):
    # we can read also a single file only into a 1 frame dataset
    d = io.load_vec(path_to_data / 'Insight'/ 'Run000001.T000.D000.P000.H001.L.vec' )
    return (d,)


@app.cell
def _(d, graphics):
    graphics.quiver(d.isel(t=0),arrScale=10, add_guide = False)
    return


@app.cell
def _(d):
    d.isel(t=0).differentiate(coord='x').differentiate(coord='y')['u'].plot.pcolormesh()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    and a vorticity map
    """)
    return


@app.cell
def _(d, graphics):
    # prepare vorticity
    d.piv.vec2scal('curl')  # it will appear as d['w'] variable, 'w' for all scalar properties
    # plot
    _fig, _ax = graphics.contour_plot(d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also, velocity histograms in x and y directions
    """)
    return


@app.cell
def _(data, graphics):
    _fig, _ax = graphics.histogram(data, normed=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also plot a whole list of vec's as subplots:
    """)
    return


@app.cell
def _(data, graphics):
    _fig, _ax = graphics.quiver(data.isel(t=0), nthArr=4, arrScale=10)
    _fig.set_size_inches(10, 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Last but not least - manipulation

    lets create a linear combinatino of our data and then see how to manipulate the coordinate system
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __Addition and Scalar multiplication__
    """)
    return


@app.cell
def _(data, graphics):
    v = (data + 3*data - 2 * data.isel(t=0)) / 3.
    graphics.quiver(v.isel(t=-1), arrScale=10)
    return (v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __Crop__
    """)
    return


@app.cell
def _(graphics, v):
    v_1 = v.piv.crop([5, 15, -5, -15])  #(xmin, xmax, ymin, ymax)
    graphics.quiver(v_1.isel(t=-1), arrScale=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __Rotate__
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __Translation of Coordinate System__
    """)
    return


@app.cell
def _(data):
    # we can also use some default plot from xarray
    data.piv.vorticity()
    data.isel(t=0)['w'].plot(robust=True)
    return


@app.cell
def _(data, plt):
    # low level quiver
    plt.figure(figsize=(8,6))
    plt.quiver(data.x,data.y,data.u[:,:,0], -data.v[:,:,0] ,data.u[:,:,0]**2 + data.v[:,:,0]**2,scale=.75)
    plt.gca().invert_yaxis()
    return


@app.cell
def _(io):
    test = io.create_sample_field(rows=25,cols=5)
    return (test,)


@app.cell
def _(graphics, test):
    graphics.quiver(test,arrScale=5,aspectratio='auto')
    return


@app.cell
def _(io, path_to_data):
    data_1 = io.load_vec(path_to_data / 'openpiv_vec' / 'exp1_001_b.vec')
    return (data_1,)


@app.cell
def _(io, path_to_data):
    variables,units,rows,cols, dt, frame, method = io.parse_header(path_to_data / 'openpiv_vec' / 'exp1_001_b.vec')
    variables,units,rows,cols, dt, frame, method
    return


@app.cell
def _(data_1):
    data_1.piv.quiver()
    return


@app.cell
def _(io, path_to_data):
    data_2 = io.load_directory(path_to_data / 'urban_canopy', ext='.vc7')
    # vc7 files are for some reason need transpose
    data_2['u'] = data_2['u'].transpose()
    data_2['v'] = data_2['v'].transpose()
    return (data_2,)


@app.cell
def _(data_2, plt):
    data_2.isel(t=0).piv.quiver(arrScale=15, colorbar=True)
    #plt.gca().invert_yaxis()
    plt.gcf().set_size_inches(12, 10)
    return


@app.cell
def _(data_2):
    # magic command not supported in marimo; please file an issue to add support
    # %time 
    df = data_2.to_dataframe()
    try:
        df.to_parquet('tmp.pq')
        print('Wrote tmp.pq')
    except ImportError as e:
        print('Parquet engine not installed; skipping to_parquet:', e)
    return (df,)


@app.cell
def _(data_2):
    # magic command not supported in marimo; please file an issue to add support
    # %time 
    data_2.to_netcdf('tmp.nc')
    return


@app.cell
def _(data_2, xr):
    _tmp = xr.load_dataset('tmp.nc')
    assert _tmp == data_2
    return


@app.cell
def _(df, xr):
    ds = xr.Dataset.from_dataframe(df)
    df.head(), ds.head()
    return


if __name__ == "__main__":
    app.run()
