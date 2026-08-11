import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import sys
    import pathlib
    import importlib.util

    print(f'Python executable: {sys.executable}')
    print(f'Working directory: {pathlib.Path.cwd().resolve()}')

    spec = importlib.util.find_spec('pivpy')
    if spec is None:
        cwd = pathlib.Path.cwd().resolve()
        for p in [cwd] + list(cwd.parents):
            if (p / 'pyproject.toml').exists() and (p / 'pivpy').is_dir():
                sys.path.insert(0, str(p))
                print(f'Added repo root to sys.path: {p}')
                spec = importlib.util.find_spec('pivpy')
                break
    if spec is None:
        raise RuntimeError('pivpy is not importable. Run from inside the repository, with the pivpy .venv active.')

    import pivpy
    import xarray as xr
    print('pivpy version:', getattr(pivpy, '__version__', 'unknown'))
    print('xarray version:', xr.__version__)
    return (pathlib,)


@app.cell
def _():
    from pivpy import io, graphics
    import pivpy.pivpy as _register_piv_accessor  # noqa: F401 -- registers the .piv accessor
    import matplotlib.pyplot as plt

    return graphics, io, plt


@app.cell
def _(io, pathlib):
    f1 = 'Run000001.T000.D000.P000.H001.L.vec'
    f2 = 'Run000002.T000.D000.P000.H001.L.vec'

    # Ensure compatibility with different Python versions (3.9+ has 'files', 3.7 and 3.8 need 'path')
    try:
        from importlib.resources import files
    except ImportError:
        from importlib.resources import path as resource_path

    # For Python 3.9+
    try:
        path = files('pivpy') / 'data'
    except NameError:
        # For Python 3.7 and 3.8
        with resource_path('pivpy', 'data') as data_path:
            path = pathlib.Path(data_path)



    a = io.load_vec(path / "Insight" / f1 )
    b = io.load_vec(path / "Insight" / f2 )
    return a, b


@app.cell
def _(a, graphics):
    # select where t = 1 (explicit time)
    _fig, _ax = graphics.quiver(a.sel(t=1), scalingFactor=1)
    #increase figure size
    _fig.set_size_inches(11, 8)
    return


@app.cell
def _(b, graphics, plt):
    # select just the first frame whatever t is .
    b['t'] = b['t'] + 10
    # define size before the plot
    plt.figure(figsize=(11, 8))
    # show less vectors using nthArr 
    _fig, _ax = graphics.quiver(b.isel(t=0), scalingFactor=2)
    return


@app.cell
def _(a):
    c = a.piv.crop([5, 15,-5,-15])
    a.u.shape, c.u.shape
    return (c,)


@app.cell
def _(c, graphics, plt):
    # define size before the plot
    plt.figure(figsize=(11, 8))
    # show less vectors using nthArr 
    _fig, _ax = graphics.quiver(c.isel(t=0), scalingFactor=1)
    return


@app.cell
def _(io):
    # let's play with some synthetic data 
    c_1 = io.create_sample_Dataset()
    return (c_1,)


@app.cell
def _(c_1):
    # want to slice it and not crop? 
    d = c_1.sel(x=slice(35, 70), y=slice(30, 90))
    print(d)
    return


@app.cell
def _(graphics, io):
    # want to show an ensemble average of 10 frames?
    data = io.create_sample_Dataset(10)
    _fig, _ = graphics.quiver(data.piv.average, scalingFactor=80)
    # want to change the size of arrows and figure aspectratio?
    _fig.set_size_inches(11, 8)
    return


@app.cell
def _(io):
    data_1 = io.create_sample_field()
    data_1.piv.strain()
    return (data_1,)


@app.cell
def _(data_1):
    data_1.piv.vorticity()
    data_1
    return


if __name__ == "__main__":
    app.run()
