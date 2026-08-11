import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import pathlib
    from importlib.resources import files
    from pivpy import io, pivpy, graphics

    return files, graphics, io, pathlib


@app.cell
def _():
    import xarray as xr
    from typing import List 
    import numpy as np
    import matplotlib.pyplot as plt

    from pivpy.graphics import quiver

    return plt, quiver


@app.cell
def _(io):
    data = io.create_sample_Dataset(n_frames=2)
    data
    return (data,)


@app.cell
def _(data):
    data["t"].shape
    return


@app.cell
def _(data):
    data.plot.quiver(x='x',y='y',u='u',v='v',row='t',scale=200)
    return


@app.cell
def _(data):
    data.isel(t=0).plot.quiver(x='x',y='y',u='u',v='v',scale=200)
    return


@app.cell
def _(data, quiver):
    quiver(data, colorbar=True, colorbar_orient='horizontal', scalingFactor=200)
    return


@app.cell
def _(data, quiver):
    quiver(data,colorbar=True, colorbar_orient='horizontal', streamlines=True, scalingFactor=200)
    return


@app.cell
def _(files, io, pathlib):
    ds = io.load_directory(
        pathlib.Path(files('pivpy').joinpath('data/PIV_Challenge')),
        ext='txt',
    )
    return (ds,)


@app.cell
def _(ds):
    ds.isel(t=0).piv.quiver(scalingFactor=30, colorbar=True, colorbar_orient='vertical', streamlines=True)
    return


@app.cell
def _(ds, graphics):
    graphics.contour_plot(ds.isel(t=0),colorbar=True)
    return


@app.cell
def _(ds, plt):
    _fig, _ax = plt.subplots()
    ds['u'].isel(t=0).plot.contourf(ax=_ax)
    return


@app.cell
def _(ds, plt):
    _fig, _ax = plt.subplots()
    c = ds['u'].isel(t=0).plot.contourf(x='x', y='y', cmap=plt.get_cmap('RdYlBu'), ax=_ax)
    return


@app.cell
def _(files, io, pathlib):
    filename = pathlib.Path(files("pivpy").joinpath("data/Insight/Run000001.T000.D000.P000.H001.L.vec"))
    # load data
    d0 = io.load_vec(filename)
    d0
    return (d0,)


@app.cell
def _(d0, graphics):
    graphics.quiver(d0)
    return


@app.cell
def _(d0):
    d1 = d0.isel(t=0) if "t" in d0.coords else d0
    return (d1,)


@app.cell
def _(d1, graphics):
    graphics.quiver(d1)
    return


@app.cell
def _(d1):
    d1
    return


@app.cell
def _(d1, graphics):
    d = d1.piv.vec2scal('vorticity')
    _fig, _ax = graphics.contour_plot(d)
    graphics.quiver(d, scalingFactor=1, ax=_ax)
    return (d,)


@app.cell
def _(d, graphics):
    graphics.showf(d,colorbar=True)
    return


@app.cell
def _(io):
    data_1 = io.create_sample_Dataset()
    tmp = data_1.piv.average
    return (tmp,)


@app.cell
def _(tmp):
    tmp
    return


@app.cell
def _(graphics, tmp):
    graphics.quiver(tmp, scalingFactor=50, colorbar=True)
    return


if __name__ == "__main__":
    app.run()
