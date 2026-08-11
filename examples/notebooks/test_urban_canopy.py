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
    # Example on the urban canopy data
    """)
    return


@app.cell
def _():
    from pivpy import io, graphics, pivpy
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    from importlib.resources import files
    try:
        from lvreader import read_buffer  # optional
    except ModuleNotFoundError:
        read_buffer = None
    import pathlib

    return files, graphics, io, np, pathlib, pivpy, plt, read_buffer, xr


@app.cell
def _(files, pathlib, read_buffer):
    filename = pathlib.Path(files('pivpy').joinpath('data/urban_canopy/B00001.vc7'))
    if read_buffer is not None:
        buffer = read_buffer(str(filename))
        buffer.plot()
    else:
        print('lvreader not installed; skipping raw buffer preview')
    return (filename,)


@app.cell
def _(filename, io):
    ds = io.load_vc7(filename)
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds, np):
    ds["s"] = np.sqrt(ds["u"]**2 + ds["v"]**2)
    ds.isel(t=0).plot.quiver(x='x',y='y',u='u',v='v',hue='s')
    return


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds, np, plt):
    plt.figure()
    # graphics.quiver(ds.isel(t=-1),arrScale=50) # show last
    ds.isel(t=0).plot.quiver(
                x='x',
                y='y',
                u='u',
                v='v',
                hue='s',
                units='width',
                scale=np.max(ds['s'].values * 25),
                headwidth=2,
                )
    return


@app.cell
def _(ds):
    ds.isel(t=0).piv.quiver(colorbar=False);
    return


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(filename, io):
    ds_1 = io.load_directory(filename.parent, ext='vc7')
    return (ds_1,)


@app.cell
def _(ds_1):
    ds_1
    return


@app.cell
def _(ds_1):
    tmp = ds_1.piv.average
    tmp
    return


@app.cell
def _(ds_1, graphics):
    graphics.quiver(ds_1.piv.average, arrScale=50)
    return


@app.cell
def _(ds_1, plt):
    # Let's try vertical profile of streamwise mean velocity U(z)
    # we could define it a shortcut of the type:
    # graphics.profile(velocity_component = 'u', coordinate = 'z') 
    plt.plot(ds_1.mean(dim='x').mean(dim='t').u, ds_1.y)
    plt.xlabel('U (m/s)')
    plt.ylabel('z (mm)')
    return


@app.cell
def _(ds_1, graphics):
    # something strange with the mask
    graphics.quiver(ds_1[dict(y=slice(100, 175))].isel(t=0))
    return


@app.cell
def _(ds_1):
    # let's test homogeneity above the canopy: 
    # take y above some value and .mean(dim='y')
    # take mean with time .mean(dim='t')
    # plot:
    ds_1.where(ds_1.y > 120, drop=True).mean(dim='t').mean(dim='y').u.plot()
    ds_1.where(ds_1.y > 150, drop=True).mean(dim='t').mean(dim='y').u.plot()
    return


@app.cell
def _(ds_1):
    ds_1.where((ds_1.x > -40) & (ds_1.x < 0), drop=True).mean(dim='t').u.mean(dim='x').plot()
    ds_1.where((ds_1.x > 20) & (ds_1.x < 80), drop=True).mean(dim='t').u.mean(dim='x').plot()
    return


@app.cell
def _(ds_1, graphics):
    graphics.quiver(ds_1.where(ds_1.y > 100, drop=True).mean(dim='t'), units=['mm', 'mm', 'm/s', 'm/s'], arrScale=50)
    return


@app.cell
def _(ds_1, np):
    ds_1['s'] = np.sqrt(ds_1['u'] ** 2 + ds_1['v'] ** 2)
    ds_1.isel(t=0)['s'].plot.contourf(x='x', y='y')
    return


if __name__ == "__main__":
    app.run()
