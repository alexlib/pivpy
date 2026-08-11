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
    # Updated notebook with the new Lavision library

    Lavision released a new Python library to read their files, called lvreader, read below:

    "
    LaVision's Python library lvreader got a new version 1.2.0 and extended its feature list by dedicated read/write functionalities for particle data sets. This makes it possible to further analyze your Shake-the-Box or PTV data with your own Python post-processing scripts. Whether you are interested in whole particle distributions or single particle tracks from your experimental data, lvreader lets you directly access DaVis set files without the need of a further export step.

    Using the particle write functions, it is possible to import particle data from other sources and use DaVis' advanced processing operations, such as binning or fine-scale reconstruction, to gain further insights into your data.

    If you are already working with the most recent Python version 3.10, the new release of lvreader supports it now as well. Click   https://www.lavision.de/en/downloads/software/index.php   to download lvreader 1.2.0."

    Or use the direct link https://www.lavision.de/en/download.php?id=4817
    """)
    return


@app.cell
def _():
    # !pip install /home/user/Downloads/lvreader-1.2.0/lvreader-1.2.0-cp38-cp38-linux_x86_64.whl
    from pivpy import io, pivpy
    from pivpy import graphics
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import pathlib

    return graphics, io, np, pathlib, xr


@app.cell
def _(io, pathlib):
    from importlib import resources
    filename = pathlib.Path(resources.files('pivpy') / "data" / "PIVMAT_jet" / "B00001.VC7")
    ds = io.load_vc7(filename)
    return ds, filename


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds, graphics):
    graphics.quiver(ds.isel(t=0),scalingFactor=200)
    return


@app.cell
def _(ds, graphics):
    new = ds.piv.filterf()
    graphics.quiver(new.isel(t=0),scalingFactor=200)
    return


@app.cell
def _(ds, graphics):
    new_1 = ds.piv.filterf([3, 3, 0])
    graphics.quiver(new_1.isel(t=0), arrScale=200)
    return (new_1,)


@app.cell
def _(new_1):
    new_1.isel(t=0).plot.quiver(x='x', y='y', u='u', v='v')
    return


@app.cell
def _(io, np):
    ds_1 = io.create_sample_Dataset(n_frames=3, rows=5, cols=10)
    ds_1 = ds_1.piv.filterf([0.5, 0.5, 0.0])
    ds_1['mag'] = np.hypot(ds_1['u'], ds_1['v'])
    ds_1.plot.quiver(x='x', y='y', u='u', v='v', hue='mag', col='t', scale=150, cmap='RdBu')
    return (ds_1,)


@app.cell
def _(mask_vars, xr):
    # from https://github.com/kaipak/xrsigproc

    from scipy.ndimage import gaussian_filter

    def _get_dims(data):
        """Get primary x-y dimensions of dataset
        """
        return ('y', 'x')

    def gaussian_smooth(data, sigma = [1., 1.], mask=False, mode='reflect'):
        """Apply gaussian kernel to convolution. Uses Scipy
           gaussian_filter method.
           Parameters:
           mode (str): {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
                       What to do at edges of matrix input. See Scipy docs
                       for details on what these do.
        """
        dims = _get_dims(data)

        sc_gaussian_nd = lambda data: gaussian_filter(data, sigma, mode=mode)

        if mask:
            data_masked = data.where(data[mask_vars[dims]])
        else:
            data_masked = data.fillna(0.)

        return xr.apply_ufunc(sc_gaussian_nd, data_masked,
                              vectorize=True,
                              dask='parallelized',
                              input_core_dims = [dims],
                              output_core_dims = [dims],
                              # output_dtypes=[data.dtype]
                            )

    return (gaussian_smooth,)


@app.cell
def _(ds_1, gaussian_smooth):
    tmp = gaussian_smooth(ds_1)
    tmp.plot.quiver(x='x', y='y', u='u', v='v', hue='mag', col='t', scale=150, cmap='RdBu')
    return


@app.cell
def _(filename, gaussian_smooth, graphics, io):
    ds_2 = io.load_vc7(filename)
    new_2 = gaussian_smooth(ds_2, sigma=[2.0, 2.0])
    # new.isel(t=0).plot.quiver(x='x',y='y',u='u',v='v')
    graphics.quiver(new_2.isel(t=0), scalingFactor=200)
    return


if __name__ == "__main__":
    app.run()
