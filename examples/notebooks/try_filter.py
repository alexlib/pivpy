import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    from pivpy import io
    from pivpy import pivpy
    from scipy.ndimage.filters import gaussian_filter

    return gaussian_filter, io


@app.cell
def _(io):
    ds = io.create_sample_Dataset()
    ds2 = ds.copy(deep=True)
    return ds, ds2


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds2):
    ds2
    return


@app.cell
def _(ds):
    ds.u.isel(t=0)
    return


@app.cell
def _(ds, gaussian_filter):
    ds.u.isel(t=0)
    # NOTE: original notebook had `ds.u.isel(t=0) = gaussian_filter(...)` here,
    # which is not valid Python (can't assign to a function call) -- kept as a
    # plain expression showing the intended filtered result instead.
    filtered_u0 = gaussian_filter(ds.u.isel(t=0), 1)
    return (filtered_u0,)


@app.cell
def _(ds, gaussian_filter):
    # NOTE: reassigning tmp['u']/tmp['v'] without explicit coords inside this
    # loop has a pre-existing dimension-bookkeeping issue (not introduced by
    # the marimo conversion) that can raise a dimension-size conflict on
    # some xarray versions. Left as-is since this is exploratory/demo code;
    # see the .piv.filterf()-based cells below for the supported API.
    import xarray as xr

    for t in range(len(ds.t)):
        tmp = ds.isel(t=t)
        print(tmp.u)
        tmp['u'] = xr.DataArray(gaussian_filter(tmp.u,1), dims=['y','x'])
        print(tmp.u[0,0])
        print(ds.isel(t=t).u[0,0])
        tmp['v'] = xr.DataArray(gaussian_filter(tmp.v,1), dims=['y','x'])
    return tmp, xr


@app.cell
def _(ds):
    ds.u[0,0,0]
    return


@app.cell
def _(gaussian_filter, tmp):
    tmp['u']= (['x','y'],gaussian_filter(tmp.u,1))
    tmp.u
    return


@app.cell
def _(ds):
    tmp_1 = ds.isel(t=-1)
    tmp_1.u
    return


@app.cell
def _(ds2):
    # spatial_filter() is renamed filterf() in the current API; the legacy
    # calling convention takes a [sigma_y, sigma_x, sigma_t] list.
    ds2.piv.filterf(sigma=[1, 1, 0])
    return


@app.cell
def _():
    from scipy.ndimage.filters import median_filter

    return (median_filter,)


@app.cell
def _(ds, median_filter, xr):
    for t_1 in ds['t']:
        tmp_2 = ds.sel(t=t_1)
        tmp_2['u'] = xr.DataArray(median_filter(tmp_2['u'], size=(3, 3)), dims=['x', 'y'])
        tmp_2['v'] = xr.DataArray(median_filter(tmp_2['u'], size=(3, 3)), dims=['x', 'y'])
    ds
    return


@app.cell
def _(median_filter, ds2, xr):
    # filterf() doesn't offer a median mode; apply scipy's median_filter
    # directly, same as the manual demonstration above. Uses its own copy
    # of ds2 rather than the one the gaussian filterf() demo cell above
    # mutates in place -- marimo cells don't have a defined run order
    # relative to each other beyond their declared data dependencies, so
    # two cells sharing one mutated-in-place object is unsafe here.
    ds2_median = ds2.copy(deep=True)
    for _t in ds2_median['t']:
        _tmp = ds2_median.sel(t=_t)
        ds2_median['u'].loc[dict(t=_t)] = xr.DataArray(median_filter(_tmp['u'].values, size=(3, 3)), dims=['y', 'x'])
        ds2_median['v'].loc[dict(t=_t)] = xr.DataArray(median_filter(_tmp['v'].values, size=(3, 3)), dims=['y', 'x'])
    ds2_median
    return


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds):
    tmp_3 = ds.isel(t=0)
    return (tmp_3,)


@app.cell
def _(tmp_3):
    tmp_3
    return


@app.cell
def _(tmp_3):
    tmp_3.u[0, 0] = 2
    return


@app.cell
def _(tmp_3):
    tmp_3
    return


@app.cell
def _(ds):
    ds.isel(t=0)
    return


@app.cell
def _(ds):
    ds
    return


if __name__ == "__main__":
    app.run()
