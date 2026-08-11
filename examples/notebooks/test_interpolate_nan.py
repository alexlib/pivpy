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
    # Test interpolation methods
    """)
    return


@app.cell
def _():
    import xarray as xr
    from pivpy import io, pivpy, graphics
    import numpy as np

    return io, np


@app.cell
def _(io):
    ds = io.create_sample_Dataset(n_frames=1,rows=7,cols=11,noise_sigma=0.5)
    return (ds,)


@app.cell
def _(ds, np):
    ds["u"][1:4,1:4] = np.nan
    ds.sel(t=0)["u"].plot()
    return


@app.cell
def _(ds):
    # see https://docs.xarray.dev/en/stable/user-guide/interpolation.html#interpolating-arrays-with-nan
    filled = ds.copy()
    filled["u"] = ds["u"].interpolate_na(dim=("x"),method='linear')
    filled["u"] = filled["u"].interpolate_na(dim=("y"),method='linear')
    filled.sel(t=0)["u"].plot()
    return (filled,)


@app.cell
def _(ds, filled):
    _filled_ds = ds.copy()
    filled['u'] = ds['u'].interpolate_na(dim='y', method='nearest')
    filled['u'] = filled['u'].interpolate_na(dim='x', method='nearest')
    filled.sel(t=0)['u'].plot()
    return


@app.cell
def _(ds, filled):
    _filled_ds = ds.copy()
    filled['u'] = ds['u'].interpolate_na(dim='y', method='nearest')
    filled['u'] = filled['u'].interpolate_na(dim='x', method='nearest')
    filled.sel(t=0)['u'].plot()
    return


@app.cell
def _(ds, filled):
    _filled_ds = ds.copy()
    filled['u'] = ds['u'].interpolate_na(dim='x', method='nearest')
    filled['u'] = filled['u'].interpolate_na(dim='y', method='nearest')
    filled.sel(t=0)['u'].plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## obviously 1d Interpolator does not work, if we do not know which direction to interpolate
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Option 1: use rioxarray idea for griddata
    """)
    return


@app.cell
def _(ds):
    ds["u"].plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we should learn how to use xarray.apply_ufunc
    follow https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html
    """)
    return


@app.cell
def _(ds):
    new = ds.copy(deep=True) # note the deep feature, otherwise, underlying data is overwritten
    return (new,)


@app.cell
def _(new):
    new.piv.fill_nans()
    return


@app.cell
def _(ds):
    ds["u"].plot()
    return


@app.cell
def _(new):
    new["u"].plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Additional ideas for small holes
    """)
    return


@app.cell
def _(np):
    # from  https://bitcoden.com/answers/numpy-inpaint-nans-interpolate-and-extrapolate
    import matplotlib.pyplot as plt
    from scipy import interpolate
    x = np.linspace(0, 1, 500)
    y = x[:, None]
    image = x + y
    mask = np.random.random(image.shape) > 0.7
    image[mask] = np.nan
    ipn_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    import scipy
    # Destroy some values

    def inpaint_nans(im):
        nans = np.isnan(im)
    # valid_mask = ~np.isnan(image)
    # coords = np.array(np.nonzero(valid_mask)).T
    # values = image[valid_mask]
        while np.sum(nans) > 0:
    # it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
            im[nans] = 0
    # filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
            vNeighbors = scipy.signal.convolve2d(nans == False, ipn_kernel, mode='same', boundary='symm')
    # f, (ax0, ax1) = plt.subplots(1, 2)
            im2 = scipy.signal.convolve2d(im, ipn_kernel, mode='same', boundary='symm')
    # ax0.imshow(image, cmap='gray', interpolation='nearest')
    # ax0.set_title('Input image')
    # ax1.imshow(filled, cmap='gray', interpolation='nearest')
    # ax1.set_title('Interpolated data')
    # plt.show()
            im2[vNeighbors > 0] = im2[vNeighbors > 0] / vNeighbors[vNeighbors > 0]
            im2[vNeighbors == 0] = np.nan
            im2[nans == False] = im[nans == False]  # kernel for inpaint_nans
            im = im2
            nans = np.isnan(im)
        return im
    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(image, cmap='gray', interpolation='nearest')
    ax0.set_title('Input image')
    ax1.imshow(inpaint_nans(image), cmap='gray', interpolation='nearest')
    ax1.set_title('Interpolated data')
    plt.show()
    return


@app.cell
def _(ds):
    ds.dropna(dim='x')["v"].shape
    return


@app.cell
def _(new):
    new.dropna(dim='x')["v"].shape
    return


@app.cell
def _(new):
    assert new.dropna(dim='t') == new
    return


if __name__ == "__main__":
    app.run()
