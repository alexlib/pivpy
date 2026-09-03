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

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from lvpyio import read_buffer
    return mo, np, plt, Path, read_buffer


@app.cell
def _(mo):
    mo.md("# Channel Flow - Vmax After Pump Shutdown")
    return


@app.cell
def _(Path):
    data_dir = Path(r"D:\channel_flow_research\baseline_channel\Vmax_after_pump_shutdown")
    vc7_dir = data_dir / "PIV_MPd(4x16x16_25%ov_ImgCorr)"
    vc7_files = sorted(vc7_dir.glob("B*.vc7"))
    mo.md(f"Found **{len(vc7_files)}** vector field files")
    return data_dir, vc7_dir, vc7_files


@app.cell
def _(mo):
    frame_slider = mo.ui.slider(1, 100, 1, label="Frame number")
    frame_slider
    return (frame_slider,)


@app.cell
def _(vc7_files, frame_slider, read_buffer, np):
    idx = frame_slider.value - 1
    path = vc7_files[idx]
    buffer = read_buffer(str(path))
    data = buffer[0]
    plane = 0

    u = data.components["U0"][plane].astype(float)
    v = data.components["V0"][plane].astype(float)
    mask_bad = np.logical_not(data.masks[plane] & data.enabled[plane])
    u[mask_bad] = np.nan
    v[mask_bad] = np.nan

    nx, ny = u.shape[1], u.shape[0]
    x = np.arange(nx)
    y = np.arange(ny)
    x = data.scales.x.offset + (x + 0.5) * data.scales.x.slope * data.grid.x
    y = data.scales.y.offset + (y + 0.5) * data.scales.y.slope * data.grid.y
    X, Y = np.meshgrid(x, y)

    u_phys = data.scales.i.offset + u * data.scales.i.slope
    v_phys = data.scales.i.offset + v * data.scales.i.slope
    if data.scales.y.slope < 0:
        v_phys = -v_phys

    speed = np.sqrt(u_phys**2 + v_phys**2)
    return X, Y, u_phys, v_phys, speed, path


@app.cell
def _(plt, X, Y, u_phys, v_phys, speed, path, mo):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    q = ax1.quiver(X, Y, u_phys, v_phys, speed, cmap="viridis", scale=3, width=0.003)
    fig.colorbar(q, ax=ax1, label="Speed [m/s]")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("Velocity Vectors")
    ax1.set_aspect("equal")

    ax2 = axes[1]
    cf = ax2.contourf(X, Y, speed, levels=20, cmap="hot")
    fig.colorbar(cf, ax=ax2, label="Speed [m/s]")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_title("Speed Contour")
    ax2.set_aspect("equal")

    plt.tight_layout()
    mo.md(f"### Frame: `{path.name}`")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
