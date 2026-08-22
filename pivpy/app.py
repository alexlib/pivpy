"""pivpy.app

Interactive Particle Image Velocimetry (PIV) diagnostic and visualization app
built with Marimo.

Run standalone via CLI:
    marimo run pivpy/app.py
    marimo edit pivpy/app.py
    python -m pivpy.app

Or programmatically:
    import pivpy.pivpy
    ds.piv.explore()
"""

from __future__ import annotations

import sys
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    import marimo as mo

    HAS_MARIMO = True
except ImportError:  # pragma: no cover
    mo = None
    HAS_MARIMO = False


if HAS_MARIMO:
    app = mo.App(width="full", app_title="PIVPy Interactive Explorer")

    @app.cell
    def __():
        import marimo as mo_inner
        import matplotlib.pyplot as plt_inner
        import numpy as np_inner
        import pivpy.pivpy  # registers .piv
        from pivpy import io as io_inner
        from pivpy.synthetic import vortex_pair as vp_inner
        from pivpy.compute_funcs import (
            vorticity,
            gamma1,
            gamma2,
            q_criterion,
            okubo_weiss,
            max_shear,
            acceleration,
            energy_spectrum,
            reynolds_decomposition,
            integral_length_scale,
            taylor_microscale,
        )

        return (
            mo_inner,
            plt_inner,
            np_inner,
            io_inner,
            vp_inner,
            vorticity,
            gamma1,
            gamma2,
            q_criterion,
            okubo_weiss,
            max_shear,
            acceleration,
            energy_spectrum,
            reynolds_decomposition,
            integral_length_scale,
            taylor_microscale,
        )

    @app.cell
    def __(mo_inner):
        header = mo_inner.md(r"""
        # 🌊 PIVPy Interactive Explorer
        ### Real-Time Fluid Dynamics Diagnostics, Vortex Topology & Spectral Analysis
        ---
        """)
        return (header,)

    @app.cell
    def __(mo_inner):
        preset_dropdown = mo_inner.ui.dropdown(
            options=[
                "Synthetic Vortex Pair (Moving)",
                "Synthetic Shear Layer",
                "Random Turbulence (Randvec)",
            ],
            value="Synthetic Vortex Pair (Moving)",
            label="Preset Dataset",
        )
        property_dropdown = mo_inner.ui.dropdown(
            options=[
                "vorticity",
                "gamma1",
                "gamma2",
                "q_criterion",
                "okubo_weiss",
                "max_shear",
                "acceleration",
                "magnitude",
            ],
            value="vorticity",
            label="Background Scalar Property",
        )
        cmap_dropdown = mo_inner.ui.dropdown(
            options=["RdBu_r", "Spectral_r", "viridis", "coolwarm", "plasma", "inferno"],
            value="RdBu_r",
            label="Colormap",
        )
        quiver_density_slider = mo_inner.ui.slider(
            start=1, stop=6, step=1, value=2, label="Arrow Stride (Density)"
        )
        quiver_scale_slider = mo_inner.ui.slider(
            start=0.2, stop=5.0, step=0.1, value=1.0, label="Arrow Scale"
        )
        frame_slider_ui = mo_inner.ui.slider(
            start=0, stop=30, step=1, value=0, label="Time Frame (t)"
        )
        return (
            preset_dropdown,
            property_dropdown,
            cmap_dropdown,
            quiver_density_slider,
            quiver_scale_slider,
            frame_slider_ui,
        )

    @app.cell
    def __(header, preset_dropdown, property_dropdown, cmap_dropdown, quiver_density_slider, quiver_scale_slider, frame_slider_ui, mo_inner):
        controls = mo_inner.vstack([
            header,
            mo_inner.hstack([preset_dropdown, property_dropdown, cmap_dropdown], justify="start", gap=2),
            mo_inner.hstack([frame_slider_ui, quiver_density_slider, quiver_scale_slider], justify="start", gap=2),
        ])
        return (controls,)

    @app.cell
    def __(preset_dropdown, vp_inner, io_inner):
        if preset_dropdown.value == "Synthetic Vortex Pair (Moving)":
            active_ds = vp_inner(n_frames=31)
        elif preset_dropdown.value == "Synthetic Shear Layer":
            active_ds = io_inner.create_sample_Dataset(n_frames=31, rows=25, cols=25)
        else:
            active_ds = io_inner.randvec(n=32, nf=31)
        return (active_ds,)

    @app.cell
    def __(
        active_ds,
        frame_slider_ui,
        property_dropdown,
        cmap_dropdown,
        quiver_density_slider,
        quiver_scale_slider,
        plt_inner,
        np_inner,
        mo_inner,
        controls,
        vorticity,
        gamma1,
        gamma2,
        q_criterion,
        okubo_weiss,
        max_shear,
        acceleration,
        energy_spectrum,
        integral_length_scale,
        taylor_microscale,
    ):
        has_t = "t" in active_ds.dims and active_ds.sizes["t"] > 1
        n_frames = active_ds.sizes["t"] if has_t else 1
        t_idx = min(frame_slider_ui.value, n_frames - 1)
        frame = active_ds.isel(t=t_idx) if has_t else active_ds

        prop = property_dropdown.value
        if prop == "vorticity":
            scalar_ds = vorticity(frame, method="circulation")
            scalar = scalar_ds["w"].to_numpy()
            label = r"Vorticity $\omega_z$ [s$^{-1}$]"
        elif prop == "gamma1":
            scalar_ds = gamma1(frame)
            scalar = scalar_ds["w"].to_numpy()
            label = r"$\Gamma_1$ Core Criterion"
        elif prop == "gamma2":
            scalar_ds = gamma2(frame)
            scalar = scalar_ds["w"].to_numpy()
            label = r"$\Gamma_2$ Boundary Criterion"
        elif prop == "q_criterion":
            scalar_ds = q_criterion(frame)
            scalar = scalar_ds["w"].to_numpy()
            label = r"$Q$-Criterion [s$^{-2}$]"
        elif prop == "okubo_weiss":
            scalar_ds = okubo_weiss(frame)
            scalar = scalar_ds["w"].to_numpy()
            label = r"Okubo-Weiss $Q_{OW}$ [s$^{-2}$]"
        elif prop == "max_shear":
            scalar_ds = max_shear(frame)
            scalar = scalar_ds["w"].to_numpy()
            label = r"Max Shear Rate [s$^{-1}$]"
        elif prop == "acceleration":
            scalar_ds = acceleration(frame, unsteady=False)
            scalar = scalar_ds["w"].to_numpy()
            label = r"Convective Acceleration [m/s$^2$]"
        else:
            u_np = frame["u"].to_numpy()
            v_np = frame["v"].to_numpy()
            scalar = np_inner.sqrt(u_np**2 + v_np**2)
            label = r"Velocity Magnitude $|\mathbf{u}|$ [m/s]"

        x = frame["x"].to_numpy()
        y = frame["y"].to_numpy()
        u = frame["u"].to_numpy()
        v = frame["v"].to_numpy()

        fig, (ax1, ax2) = plt_inner.subplots(1, 2, figsize=(14, 5.5), dpi=120, gridspec_kw={"width_ratios": [1.4, 1.0]})

        # Main Quiver + Contour Plot
        vmax = float(np_inner.nanpercentile(np_inner.abs(scalar), 98)) if np_inner.any(scalar) else 1.0
        vmin = -vmax if prop in {"vorticity", "gamma1", "gamma2", "q_criterion", "okubo_weiss"} else 0.0

        cf = ax1.contourf(x, y, scalar, levels=50, cmap=cmap_dropdown.value, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(cf, ax=ax1, shrink=0.85, pad=0.02)
        cb.set_label(label, fontsize=10)

        step = int(quiver_density_slider.value)
        scale_fac = float(quiver_scale_slider.value)
        ax1.quiver(
            x[::step],
            y[::step],
            u[::step, ::step],
            v[::step, ::step],
            color="k",
            alpha=0.7,
            angles="xy",
            scale_units="xy",
            scale=scale_fac,
            width=0.0035,
        )
        ax1.set_title(f"Field Overview at t = {t_idx} (Grid {len(x)}x{len(y)})", fontsize=12, pad=8)
        ax1.set_xlabel("x [mm]")
        ax1.set_ylabel("y [mm]")

        # 1D Radial Energy Spectrum Plot
        spec_ds = energy_spectrum(frame, radial=True)
        k = spec_ds["k"].to_numpy()
        E_rad = spec_ds["E_radial"].to_numpy()
        ax2.loglog(k[1:], E_rad[1:], "o-", color="royalblue", lw=1.8, ms=4, label=r"$E(k)$")
        
        # Reference Kolmogorov -5/3 slope
        if len(k) > 4:
            k_ref = k[1 : len(k) // 2 + 1]
            E_ref = E_rad[1] * (k_ref / k_ref[0]) ** (-5.0 / 3.0)
            ax2.loglog(k_ref, E_ref, "--", color="crimson", lw=1.5, label=r"$k^{-5/3}$ Slope")

        ax2.set_title("Radial Energy Spectrum E(k)", fontsize=12, pad=8)
        ax2.set_xlabel(r"Wavenumber $k$ [rad/m]")
        ax2.set_ylabel(r"Energy $E(k)$ [(m/s)$^2$/(rad/m)]")
        ax2.grid(True, which="both", alpha=0.3, ls=":")
        ax2.legend(frameon=True, fontsize=10)

        fig.tight_layout()

        # Compute diagnostic stats
        L11 = integral_length_scale(frame, component="u", dim="x")
        lambda_T = taylor_microscale(frame, component="u", dim="x", method="curvature")
        u_mean_mag = float(np_inner.mean(np_inner.sqrt(u**2 + v**2)))
        max_vort = float(np_inner.max(np_inner.abs(scalar))) if prop == "vorticity" else float(np_inner.max(np_inner.abs(vorticity(frame)["w"].to_numpy())))

        stats_card = mo_inner.md(f"""
        ### 📊 Instant Diagnostics Summary
        | Diagnostic Metric | Value | Physical Unit |
        |---|---|---|
        | **Mean Velocity $\\langle |\\mathbf{{u}}| \\rangle$** | `{u_mean_mag:.3f}` | m/s |
        | **Peak Vorticity $|\\omega_z|_{{\\max}}$** | `{max_vort:.3f}` | s$^{{-1}}$ |
        | **Integral Length Scale $L_{{11}}$** | `{L11:.3f}` | length units |
        | **Taylor Microscale $\\lambda_T$** | `{lambda_T:.3f}` | length units |
        """)

        dashboard = mo_inner.vstack([
            controls,
            mo_inner.ui.matplotlib(fig),
            stats_card,
        ])
        return (dashboard,)

    @app.cell
    def __(dashboard):
        dashboard
        return
else:
    app = None


def launch_app(
    dataset: Optional[xr.Dataset] = None,
    port: int = 8000,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> None:
    """Launches the interactive Marimo PIVPy Explorer web app.

    Parameters
    ----------
    dataset : xr.Dataset, optional
        Initial dataset to load.
    port : int
        Port number to run the server on (default 8000).
    host : str
        Host address (default '127.0.0.1').
    open_browser : bool
        If True, opens the default web browser on launch.
    """
    if not HAS_MARIMO:
        raise ImportError("Marimo is required to run the interactive explorer. Install via `pip install marimo`.")

    import subprocess
    cmd = [sys.executable, "-m", "marimo", "run", __file__, "--port", str(port), "--host", host]
    if not open_browser:
        cmd.append("--no-browser")
    subprocess.run(cmd)


if __name__ == "__main__":
    if HAS_MARIMO:
        app.run()
    else:
        print("Marimo is required to run pivpy.app. Please install marimo via `pip install marimo`.")
