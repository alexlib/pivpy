import marimo

# /// script
# dependencies = [
#     "marimo",
#     "pivpy",
#     "matplotlib",
# ]
# ///

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explore a PIV field, live

    Move the controls below and watch the plot redraw immediately -- this is
    the same interactive-tuning workflow used to pick visualization
    parameters for a real experiment, running here on a synthetic sample
    dataset, entirely client-side in your browser.
    """)
    return


@app.cell
def _():
    from pivpy import io, pivpy  # noqa: F401  (registers the .piv accessor)

    return (io,)


@app.cell
def _(io):
    ds = io.create_sample_Dataset(n_frames=1, rows=24, cols=32, noise_sigma=0.15)
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    background_dd = mo.ui.dropdown(
        options=["vorticity", "mag", "ke", "divergence", "off"],
        value="vorticity",
        label="background",
    )
    color_by_dd = mo.ui.dropdown(
        options=["none", "mag", "u", "v"], value="none", label="color arrows by"
    )
    cmap_dd = mo.ui.dropdown(
        options=["viridis", "RdBu_r", "coolwarm", "plasma", "jet"],
        value="viridis",
        label="colormap",
    )
    streamlines_cb = mo.ui.checkbox(value=True, label="streamlines")
    row1 = mo.hstack([background_dd, color_by_dd, cmap_dd, streamlines_cb])

    arrow_scale_auto_cb = mo.ui.checkbox(value=True, label="auto arrow scale")
    arrow_scale_slider = mo.ui.slider(
        0.1, 10.0, step=0.1, value=1.0, label="arrow scale (if not auto)"
    )
    arrow_alpha_slider = mo.ui.slider(0.1, 1.0, step=0.05, value=0.75, label="arrow alpha")
    row2 = mo.hstack([arrow_scale_auto_cb, arrow_scale_slider, arrow_alpha_slider])

    skip_rows_slider = mo.ui.slider(1, 5, step=1, value=1, label="skip rows")
    skip_cols_slider = mo.ui.slider(1, 5, step=1, value=1, label="skip cols")
    row3 = mo.hstack([skip_rows_slider, skip_cols_slider])

    mo.vstack([row1, row2, row3])
    return (
        arrow_alpha_slider,
        arrow_scale_auto_cb,
        arrow_scale_slider,
        background_dd,
        cmap_dd,
        color_by_dd,
        skip_cols_slider,
        skip_rows_slider,
        streamlines_cb,
    )


@app.cell
def _(
    arrow_alpha_slider,
    arrow_scale_auto_cb,
    arrow_scale_slider,
    background_dd,
    cmap_dd,
    color_by_dd,
    ds,
    skip_cols_slider,
    skip_rows_slider,
    streamlines_cb,
):
    fig, ax = ds.piv.plot(
        background=None if background_dd.value == "off" else background_dd.value,
        color_by=None if color_by_dd.value == "none" else color_by_dd.value,
        cmap=cmap_dd.value,
        streamlines=streamlines_cb.value,
        arrow_scale=None if arrow_scale_auto_cb.value else arrow_scale_slider.value,
        arrow_alpha=arrow_alpha_slider.value,
        skip=(skip_rows_slider.value, skip_cols_slider.value),
        title="Sample PIV field",
    )
    fig.gca()
    return


if __name__ == "__main__":
    app.run()
