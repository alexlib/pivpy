import marimo

# /// script
# dependencies = [
#     "marimo",
#     "pivpy",
#     "numpy",
#     "xarray",
# ]
# ///

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data structure is based on xarray.Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PIV data requires:
    - data in 2D or 3D matrices
    - coordinates for x,y or x,y,z
    - metadata that will contain the information from the header, information about the origin of the data file (image, experimental settings), units for each variables, coordinates, etc.

    Among various possibilities the most suitable one is `xarray`, or so-called N-D labeled arrays, Read more about this format in this [paper](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.148/) or in their [docs](https://xarray.pydata.org/en/stable/)
    """)
    return


@app.cell
def _():
    import xarray as xr
    import numpy as np
    from pivpy import pivpy, io, graphics

    return io, np, xr


@app.cell
def _(np, xr):
    x = np.linspace(32.0, 128.0, 3) # 3 columns
    y = np.linspace(16.0, 128.0, 4) # 4 rows

    xm, ym = np.meshgrid(x, y)
    u = np.ones_like(xm.T) + np.linspace(0.0, 7.0, 4)
    v = (
        np.zeros_like(ym.T)
        + np.linspace(0.0, 1.0, 4)
        + np.random.rand(3, 1)
        - 0.5
    )

    u = u[:, :, np.newaxis]
    v = v[:, :, np.newaxis]
    chc = np.ones_like(u)

    # plt.quiver(xm.T,ym.T,u,v)

    u = xr.DataArray(
        u, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [0]}
    )
    v = xr.DataArray(
        v, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [0]}
    )
    chc = xr.DataArray(
        chc, dims=("x", "y", "t"), coords={"x": x, "y": y, "t": [0]}
    )

    data = xr.Dataset({"u": u, "v": v, "chc": chc})

    data.attrs["variables"] = ["x", "y", "u", "v"]
    data.attrs["units"] = ["pix", "pix", "pix/dt", "pix/dt"]
    data.attrs["dt"] = 1.0
    data.attrs["files"] = ""

    data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using xarray plotting machinery
    """)
    return


@app.cell
def _(np, xr):
    ds = xr.Dataset()
    ds.coords['x'] = ('x', np.arange(10))
    ds.coords['y'] = ('y', np.arange(20))
    ds.coords['t'] = ('t', np.arange(4))
    sx = xr.apply_ufunc(np.sin, (ds.x - 5) / 5)
    sy = xr.apply_ufunc(np.sin, (ds.y - 10) / 10)
    cy = xr.apply_ufunc(np.cos, (ds.y - 10) / 10)
    ds['u'] = sx * sy
    ds['v'] = sx * cy
    mod = 2 * xr.apply_ufunc(np.cos, ds.t * 2 * np.pi / 0.75)
    ds = ds * mod
    ds['u'].attrs['units'] = 'm/s'
    ds['mag'] = (ds['u'] ** 2 + ds['v'] ** 2) ** 0.5
    ds.mag.plot(col='t', x='x')
    _fg = ds.plot.quiver(x='x', y='y', u='u', v='v', col='t', hue='mag', scale=1)  # type: ignore[call-arg]
    return


@app.cell
def _(io):
    ds_1 = io.create_sample_Dataset(n_frames=3, rows=5, cols=9, noise_sigma=0.2)
    ds_1['mag'] = (ds_1['u'] ** 2 + ds_1['v'] ** 2) ** 0.5
    _fg = ds_1.plot.quiver(x='x', y='y', u='u', v='v', col='t', hue='mag', scale=100)
    return (ds_1,)


@app.cell
def _(ds_1):
    # using overloaded pivpy graphics.quiver
    ds_1.piv.quiver(scalingFactor=100)
    return


if __name__ == "__main__":
    app.run()
