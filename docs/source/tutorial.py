import marimo

# /// script
# dependencies = [
#     "marimo",
#     "pivpy",
#     "matplotlib",
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
    # PIVPy graphics tutorial
    """)
    return


@app.cell
def _():
    # import xarray as xr
    # import numpy as np
    from pivpy import io, pivpy, graphics
    import matplotlib.pyplot as plt

    return graphics, io, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's create a sample dataset
    """)
    return


@app.cell
def _(io):
    df = io.create_sample_Dataset()
    df
    return (df,)


@app.cell
def _(df, graphics, plt):
    plt.figure(figsize=(10,10))
    graphics.quiver(df.piv.average,arrScale=3,streamlines=True);
    return


@app.cell
def _(df, plt):
    plt.figure(figsize=(10,10)) 
    # plot quiver of the first frame (t[0]), selected by .isel and apply quiver()
    df.isel(t=0).piv.quiver(arrScale=7,streamlines=True)
    return


@app.cell
def _(df, graphics):
    fig,ax = graphics.contour_plot(df.isel(t=-1),colorbar='vertical')
    fig.set_size_inches(6,6)
    ax.set_xlabel('$x$ (pix)',fontsize=16);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Publication-quality figure in one call

    `streamscal` draws a smooth vorticity field under dense black streamlines with small
    direction arrows. No axes, no margins, nothing to tune.
    """)
    return


@app.cell
def _(df):
    _fig, _ax = df.isel(t=0).piv.streamscal()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Change the colour quantity with `scalar=` (`"speed"`, `"u"`, `"divergence"`, any variable)
    and the line density with `density=`. `streamscal_panels` places several frames side by
    side on one shared colour scale.
    """)
    return


@app.cell
def _(df, graphics):
    _fig, _axs = graphics.streamscal_panels(
        [df.isel(t=0), df.isel(t=-1)], scalar="speed", density=1.8, labels=["(a)", "(b)"]
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
