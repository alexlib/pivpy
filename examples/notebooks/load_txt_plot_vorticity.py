import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    from pivpy import io, graphics, pivpy
    from importlib.resources import files
    import matplotlib.pyplot as plt
    import numpy as np
    import pathlib
    import os

    return files, graphics, io, np, os, pathlib, plt, pivpy


@app.cell
def _(files, io, pathlib):
    filename = pathlib.Path(files("pivpy").joinpath("data/openpiv_txt/exp1_001_b.txt"))
    data = io.load_vec(filename)
    return (data,)


@app.cell
def _(np, pivpy):
    try:
        import watermark
        print(watermark.watermark(python=True, packages="numpy,pivpy"))
    except ModuleNotFoundError:
        print('watermark not installed; skipping version stamp')
        print('numpy', np.__version__)
        print('pivpy', pivpy.__version__ if hasattr(pivpy, '__version__') else '')
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data):
    data.piv.vec2scal('vorticity')

    data
    return


@app.cell
def _(data):
    data.piv.vec2scal('vorticity')
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, graphics):
    graphics.contour_plot(data.isel(t=0))
    return


@app.cell
def _(data):
    data.piv.vec2scal('ke')
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, graphics):
    graphics.contour_plot(data.isel(t=0))
    return


@app.cell
def _(data):
    fig,ax = data.piv.quiver(colorbar=True)
    # fig.set_size_inches(11,10)
    return


@app.cell
def _(data, graphics):
    fig_1, ax_1 = graphics.quiver(data.isel(t=-1), colorbar=True, colorbar_orient='horizontal', arrScale=15)
    fig_1.set_size_inches(12, 8)
    return


if __name__ == "__main__":
    app.run()
