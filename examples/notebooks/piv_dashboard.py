import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PIVPy — interactive flow dashboard

    Load bundled Insight PIV data, explore vector fields interactively,
    and reuse the same cells in a dashboard, a printable team report,
    and a seminar slide deck (marimo-studio views).
    """)
    return


@app.cell
def _():
    import importlib.resources as importlib_resources

    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from pivpy import graphics, io

    return graphics, importlib_resources, io


@app.cell
def _(importlib_resources, io, mo):
    path_to_data = importlib_resources.files("pivpy") / "data"
    data = io.load_directory(path_to_data / "Insight")
    n_frames = int(data.sizes["t"])
    mo.md(
        f"Loaded **{n_frames}** frames from the bundled `Insight` directory "
        f"(`{int(data.sizes['x'])}×{int(data.sizes['y'])}` grid)."
    )
    return data, n_frames


@app.cell
def _(mo, n_frames):
    frame = mo.ui.slider(
        start=0, stop=int(n_frames) - 1, step=1, value=0, label="Frame (t index)"
    )
    vec_scale = mo.ui.slider(
        start=0.2, stop=5.0, step=0.1, value=1.0, label="Vector scaling"
    )
    show_vectors = mo.ui.checkbox(value=True, label="Overlay vectors on scalar maps")
    cmap = mo.ui.dropdown(
        options=["RdYlBu_r", "viridis", "coolwarm", "magma"],
        value="RdYlBu_r",
        label="Colormap",
    )
    mo.vstack([frame, vec_scale, show_vectors, cmap], gap=0.5)
    return cmap, frame, show_vectors, vec_scale


@app.cell
def _(data, frame):
    frame_idx = int(frame.value)
    # vorticity() returns a new dataset with a 'w' variable; keep original intact
    frame_data = data.isel(t=frame_idx).piv.vorticity()
    speed = float(
        ((frame_data["u"] ** 2 + frame_data["v"] ** 2) ** 0.5).mean().values
    )
    vort_mean = float(frame_data["w"].mean().values)
    vort_max = float(abs(frame_data["w"]).max().values)
    # frame_data
    return frame_data, frame_idx, speed, vort_max, vort_mean


@app.cell(hide_code=True)
def _(frame_idx, mo, speed, vort_max, vort_mean):
    mo.md(f"""
    **Frame {frame_idx}** · mean speed `{speed:.3f}` · mean vorticity
    `{vort_mean:.4f}` · max |vorticity| `{vort_max:.4f}`
    """)
    return


@app.cell
def quiver_view_(frame_data, graphics, vec_scale):
    fig_q, _ax_q = graphics.quiver(
        frame_data, scalingFactor=float(vec_scale.value)
    )
    fig_q.set_size_inches(7, 5.5)
    fig_q
    return


@app.cell
def vorticity_view_(cmap, frame_data, graphics, show_vectors):
    fig_w, ax_w = graphics.contour_plot(
        frame_data, property="w", cmap=cmap.value
    )
    if bool(show_vectors.value):
        graphics.quiver(frame_data, ax=ax_w, scalingFactor=1.0, arrowColor="k")
    fig_w.set_size_inches(7, 5.5)
    fig_w
    return


@app.cell
def speed_overlay_view_(cmap, frame_data, graphics, show_vectors):
    import numpy as _np

    _fr = frame_data.copy()
    _fr["mag"] = _np.sqrt(_fr["u"] ** 2 + _fr["v"] ** 2)
    fig_s, ax_s = graphics.contour_plot(_fr, property="mag", cmap=cmap.value)
    ax_s.set_title("Speed magnitude with vector overlay")
    if bool(show_vectors.value):
        graphics.quiver(frame_data, ax=ax_s, scalingFactor=1.0, arrowColor="white")
    fig_s.set_size_inches(7, 5.5)
    fig_s
    return


@app.cell
def histogram_view_(frame_data, graphics):
    fig_h, _ax_h = graphics.histogram(frame_data, bins=50)
    fig_h.set_size_inches(7, 4.2)
    fig_h
    return


@app.cell
def streamlines_view_(frame_data, graphics):
    fig_st, _ax_st = graphics.streamplot(frame_data, density=1.2)
    fig_st.set_size_inches(7, 5.5)
    fig_st
    return


@app.cell
def stats_view_(frame_data, mo):
    _u = float(frame_data["u"].mean().values)
    _v = float(frame_data["v"].mean().values)
    _sp = float(
        ((frame_data["u"] ** 2 + frame_data["v"] ** 2) ** 0.5).mean().values
    )
    _w = float(frame_data["w"].mean().values)
    _n = int((frame_data["chc"] > 0).sum().values)
    # Plain markdown (not mo.ui.table / dataframe widget) so the cell stays
    # portable for Prepared static export; live interactivity stays in run mode.
    mo.md(
        f"""
    | quantity | value |
    |---|---|
    | u mean | {_u:.4f} |
    | v mean | {_v:.4f} |
    | speed mean | {_sp:.4f} |
    | vorticity mean | {_w:.4f} |
    | valid vectors | {_n} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Team discussion notes

    - Check vector outliers near the edges (`chc` flag) before trusting vorticity peaks.
    - Compare frames 0–4: does the high-vorticity patch advect or diffuse?
    - Decide crop window + filter settings for the next processing pass.
    """)
    return


if __name__ == "__main__":
    app.run()
