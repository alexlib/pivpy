# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "pivpy",
#     "lvpyio",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    from pathlib import Path
    from lvpyio import read_buffer
    from pivpy.schema import build_dataset

    return Path, build_dataset, mo, np, read_buffer


@app.cell
def _(mo):
    mo.md("""
    # Channel Flow - Vmax After Pump Shutdown
    """)
    return


@app.cell
def _(Path, mo):
    data_dir = Path(r"D:\channel_flow_research\baseline_channel\Vmax_after_pump_shutdown")
    vc7_dir = data_dir / "PIV_MPd(4x16x16_25%ov_ImgCorr)"
    vc7_files = sorted(vc7_dir.glob("B*.vc7"))
    n_files = len(vc7_files)
    mo.md(f"Found **{n_files}** vector field files")
    return n_files, vc7_files


@app.cell
def _(mo, n_files):
    frame_slider = mo.ui.slider(1, max(1, n_files), 1, label="Frame")
    frame_slider
    return (frame_slider,)


@app.cell
def _(mo):
    bg_inst = mo.ui.radio(
        options=["vorticity", "speed", "divergence", "none"],
        value="vorticity",
        label="Instantaneous background",
    )
    skip_inst = mo.ui.slider(0, 16, 2, label="Arrow skip")
    scale_inst = mo.ui.slider(0.1, 200.0, step=0.5, value=1.0, label="Arrow scale")
    stream_inst = mo.ui.checkbox(label="Streamlines", value=True)
    mo.hstack([bg_inst, skip_inst, scale_inst, stream_inst], justify="start")
    return bg_inst, scale_inst, skip_inst, stream_inst


@app.cell
def _(build_dataset, frame_slider, np, read_buffer, vc7_files):
    _idx = frame_slider.value - 1
    _path = vc7_files[_idx]
    _buffer = read_buffer(str(_path))
    _data = _buffer[0]
    _plane = 0

    _u = _data.components["U0"][_plane].astype(float)
    _v = _data.components["V0"][_plane].astype(float)
    _mask_bad = np.logical_not(_data.masks[_plane] & _data.enabled[_plane])
    _u[_mask_bad] = np.nan
    _v[_mask_bad] = np.nan

    _x = np.arange(_u.shape[1])
    _y = np.arange(_u.shape[0])
    _x = _data.scales.x.offset + (_x + 0.5) * _data.scales.x.slope * _data.grid.x
    _y = _data.scales.y.offset + (_y + 0.5) * _data.scales.y.slope * _data.grid.y

    _u_phys = _data.scales.i.offset + _u * _data.scales.i.slope
    _v_phys = _data.scales.i.offset + _v * _data.scales.i.slope
    if _data.scales.y.slope < 0:
        _v_phys = -_v_phys

    _mask = (~_mask_bad).astype(float)
    ds_inst = build_dataset(_x, _y, _u_phys, _v_phys, mask=_mask, frame=0)
    path = _path
    return ds_inst, path


@app.cell
def _(bg_inst, ds_inst, mo, path, scale_inst, skip_inst, stream_inst):
    _bg = bg_inst.value if bg_inst.value != "none" else None
    _fig, _ax = ds_inst.piv.plot(
        background=_bg,
        streamlines=stream_inst.value,
        quiver=True,
        skip=skip_inst.value,
        arrow_scale=scale_inst.value,
        title=f"Instantaneous: {path.name}",
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell
def _(mo):
    bg_avg = mo.ui.radio(
        options=["vorticity", "speed", "divergence", "none"],
        value="speed",
        label="Ensemble average background",
    )
    skip_avg = mo.ui.slider(1, 16, 1, label="Arrow skip")
    scale_avg = mo.ui.slider(0.1, 200.0, step=0.5, value=5.0, label="Arrow scale")
    stream_avg = mo.ui.checkbox(label="Streamlines", value=True)
    mo.hstack([bg_avg, skip_avg, scale_avg, stream_avg], justify="start")
    return bg_avg, scale_avg, skip_avg, stream_avg


@app.cell
def _(build_dataset, mo, np, read_buffer, vc7_files):
    _u_sum = None
    _v_sum = None
    _mask_sum = None
    _x = None
    _y = None
    _n = len(vc7_files)

    for _f in vc7_files:
        _buffer = read_buffer(str(_f))
        _data = _buffer[0]
        _plane = 0

        _u = _data.components["U0"][_plane].astype(float)
        _v = _data.components["V0"][_plane].astype(float)
        _mask_bad = np.logical_not(_data.masks[_plane] & _data.enabled[_plane])
        _u[_mask_bad] = np.nan
        _v[_mask_bad] = np.nan

        if _x is None:
            _x = np.arange(_u.shape[1])
            _y = np.arange(_u.shape[0])
            _x = _data.scales.x.offset + (_x + 0.5) * _data.scales.x.slope * _data.grid.x
            _y = _data.scales.y.offset + (_y + 0.5) * _data.scales.y.slope * _data.grid.y

        _u_phys = _data.scales.i.offset + _u * _data.scales.i.slope
        _v_phys = _data.scales.i.offset + _v * _data.scales.i.slope
        if _data.scales.y.slope < 0:
            _v_phys = -_v_phys

        if _u_sum is None:
            _u_sum = np.zeros_like(_u_phys)
            _v_sum = np.zeros_like(_v_phys)
            _mask_sum = np.zeros_like(_u_phys)

        _valid = ~np.isnan(_u_phys) & ~np.isnan(_v_phys)
        _u_sum[_valid] += _u_phys[_valid]
        _v_sum[_valid] += _v_phys[_valid]
        _mask_sum[_valid] += 1.0

    _u_avg = np.where(_mask_sum > 0, _u_sum / _mask_sum, np.nan)
    _v_avg = np.where(_mask_sum > 0, _v_sum / _mask_sum, np.nan)
    _mask_avg = (_mask_sum > 0).astype(float)

    ds_avg = build_dataset(_x, _y, _u_avg, _v_avg, mask=_mask_avg, frame=0)
    mo.md(f"Ensemble average over **{_n}** frames")
    return (ds_avg,)


@app.cell
def _(bg_avg, ds_avg, mo, scale_avg, skip_avg, stream_avg):
    _bg = bg_avg.value if bg_avg.value != "none" else None
    _fig, _ax = ds_avg.piv.plot(
        background=_bg,
        streamlines=stream_avg.value,
        quiver=True,
        skip=skip_avg.value,
        arrow_scale=scale_avg.value,
        title="Ensemble Average",
    )
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
