"""pivpy.synthetic

Analytical and stochastic velocity vector field generators for PIVPy.

This module provides exact analytical and synthetic flow fields formatted
as canonical PIVPy xarray.Datasets (with dims ('y', 'x', 't'), coords x, y, t,
and variables u, v, chc, optional w).

Supported flow models:
- vortex: Lamb-Oseen, Burgers, Rankine, and Vatistas vortices
- multivortex: 2D synthetic turbulence from multiple random Burgers/Lamb vortices
- randvec: Divergence-free random velocity fields with prescribed power spectrum
- channel: Analytical laminar Poiseuille channel flow
- shear_layer: Analytical hyperbolic tangent shear/mixing layer
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union
import numpy as np
import xarray as xr

from pivpy.schema import (
    DELTA_T,
    build_dataset,
    set_default_attrs,
)


def vortex(
    n: Union[int, Tuple[int, int]] = 128,
    r0: float = 10.0,
    vorticity: float = 1.0,
    mode: Literal["burgers", "lamb", "rankine", "vatistas"] = "burgers",
    diver: float = 0.0,
    center: Optional[Tuple[float, float]] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    frame: int = 0,
    n_vatistas: float = 2.0,
) -> xr.Dataset:
    """Generate an analytical 2D vector field containing a centered or offset vortex.

    Parameters
    ----------
    n : int or tuple of (rows, cols)
        Grid dimension. If int, creates an (n, n) grid.
    r0 : float
        Vortex core radius in coordinate units (default 10.0).
    vorticity : float
        Peak vorticity / circulation parameter $\\omega_0$ in $s^{-1}$ (default 1.0).
    mode : {'burgers', 'lamb', 'rankine', 'vatistas'}
        Vortex profile type:
        - 'burgers' or 'lamb': Lamb-Oseen / Burgers Gaussian vorticity profile
        - 'rankine': Solid-body rotation inside core, potential vortex outside
        - 'vatistas': Generalized algebraic vortex profile
    diver : float
        Radial divergence / suction parameter $\\gamma$ (default 0.0).
    center : tuple of (x0, y0), optional
        Vortex center coordinates. Default is domain center.
    dx, dy : float
        Grid spacing in x and y directions (default 1.0).
    frame : int
        Time frame index (default 0).
    n_vatistas : float
        Exponent parameter for Vatistas vortex (default 2.0).

    Returns
    -------
    xr.Dataset
        Canonical PIVPy dataset with variables u, v, chc, coords (x, y, t).
    """
    if isinstance(n, int):
        rows, cols = n, n
    else:
        rows, cols = n

    x_coords = np.arange(cols, dtype=float) * dx
    y_coords = np.arange(rows, dtype=float) * dy
    x2d, y2d = np.meshgrid(x_coords, y_coords)

    if center is None:
        x0 = float(x_coords[-1] + x_coords[0]) / 2.0
        y0 = float(y_coords[-1] + y_coords[0]) / 2.0
    else:
        x0, y0 = center

    rx = x2d - x0
    ry = y2d - y0
    radius = np.sqrt(rx**2 + ry**2)

    # Angular velocity & divergence scales
    omega = float(vorticity) / 2.0
    gamma = float(diver) / 2.0

    u = np.zeros_like(radius, dtype=float)
    v = np.zeros_like(radius, dtype=float)

    mode_lower = mode.lower()
    safe_radius = np.where(radius == 0.0, 1e-12, radius)

    if mode_lower in ("burgers", "lamb"):
        decay = 1.0 - np.exp(-((radius / r0) ** 2))
        circ_factor = omega * (r0**2) / (safe_radius**2) * decay
        u = -circ_factor * ry
        v = circ_factor * rx

        if gamma != 0.0:
            div_factor = gamma * (r0**2) / (safe_radius**2) * decay
            u += div_factor * rx
            v += div_factor * ry

        u[radius == 0.0] = 0.0
        v[radius == 0.0] = 0.0

    elif mode_lower == "rankine":
        inside = radius <= r0
        outside = ~inside

        u[inside] = -omega * ry[inside]
        v[inside] = omega * rx[inside]

        circ_factor = omega * (r0**2) / (safe_radius[outside] ** 2)
        u[outside] = -circ_factor * ry[outside]
        v[outside] = circ_factor * rx[outside]

    elif mode_lower == "vatistas":
        factor = omega / ((1.0 + (radius / r0) ** (2.0 * n_vatistas)) ** (1.0 / n_vatistas))
        u = -factor * ry
        v = factor * rx

    else:
        raise ValueError(f"Unknown vortex mode '{mode}'. Choose from 'burgers', 'lamb', 'rankine', 'vatistas'.")

    chc = np.ones_like(u, dtype=float)

    u_3d = u[:, :, np.newaxis]
    v_3d = v[:, :, np.newaxis]
    chc_3d = chc[:, :, np.newaxis]
    t_coords = np.array([float(frame)], dtype=float)

    ds = build_dataset(
        x=x_coords,
        y=y_coords,
        t=t_coords,
        u=u_3d,
        v=v_3d,
        chc=chc_3d,
        delta_t=float(DELTA_T),
    )
    ds.attrs["flow_model"] = f"vortex_{mode_lower}"
    return ds


def multivortex(
    n_frames: int = 1,
    n: Union[int, Tuple[int, int]] = 128,
    n_vortices: int = 8,
    two_d: bool = True,
    asym: bool = False,
    dx: float = 1.0,
    dy: float = 1.0,
    seed: Optional[int] = None,
) -> xr.Dataset:
    """Generate 2D synthetic turbulence fields composed of multiple random Burgers vortices.

    Parameters
    ----------
    n_frames : int
        Number of time frames to generate (default 1).
    n : int or tuple of (rows, cols)
        Grid dimension (default 128).
    n_vortices : int
        Average number of vortices per frame (default 8).
    two_d : bool
        If True, enforces 2D zero-divergence (gamma = 0). Default True.
    asym : bool
        If True, generates only positive vorticity (cyclonic). Default False.
    dx, dy : float
        Grid spacing (default 1.0).
    seed : int, optional
        Random seed for reproducible realizations.

    Returns
    -------
    xr.Dataset
        Canonical PIVPy dataset with multi-frame turbulent flow field.
    """
    if isinstance(n, int):
        rows, cols = n, n
    else:
        rows, cols = n

    rng = np.random.default_rng(seed)
    x_coords = np.arange(cols, dtype=float) * dx
    y_coords = np.arange(rows, dtype=float) * dy
    x2d, y2d = np.meshgrid(x_coords, y_coords)

    domain_w = float(x_coords[-1] - x_coords[0])
    domain_h = float(y_coords[-1] - y_coords[0])
    diag = np.sqrt(domain_w**2 + domain_h**2)

    n_total_vortices = int(np.ceil(n_vortices * 9))
    frames = []

    for t_idx in range(n_frames):
        u_frame = np.zeros((rows, cols), dtype=float)
        v_frame = np.zeros((rows, cols), dtype=float)

        xc = x_coords[0] + domain_w * (3.0 * rng.random(n_total_vortices) - 1.0)
        yc = y_coords[0] + domain_h * (3.0 * rng.random(n_total_vortices) - 1.0)

        omega = rng.choice([-1.0, 1.0], size=n_total_vortices) * (2.0 + rng.standard_normal(n_total_vortices))
        if asym:
            omega = np.abs(omega)

        div = np.zeros(n_total_vortices, dtype=float) if two_d else 0.5 * rng.standard_normal(n_total_vortices)
        core = 0.015 * (4.0 + rng.standard_normal(n_total_vortices)) * diag
        core = np.maximum(core, 2.0 * min(dx, dy))

        for k in range(n_total_vortices):
            rx = x2d - xc[k]
            ry = y2d - yc[k]
            r2 = rx**2 + ry**2
            safe_r2 = np.where(r2 == 0.0, 1e-12, r2)
            c2 = core[k] ** 2

            decay = (1.0 - np.exp(-r2 / c2)) / safe_r2
            ampl_rot = omega[k] * c2 / 2.0 * decay
            ampl_div = div[k] * c2 / 2.0 * decay

            u_frame += -ampl_rot * ry + ampl_div * rx
            v_frame += ampl_rot * rx + ampl_div * ry

        chc_frame = np.ones_like(u_frame, dtype=float)
        ds_t = build_dataset(
            x=x_coords,
            y=y_coords,
            t=np.array([float(t_idx)], dtype=float),
            u=u_frame[:, :, np.newaxis],
            v=v_frame[:, :, np.newaxis],
            chc=chc_frame[:, :, np.newaxis],
            delta_t=float(DELTA_T),
        )
        frames.append(ds_t)

    ds = xr.concat(frames, dim="t") if len(frames) > 1 else frames[0]
    ds.attrs["flow_model"] = "multivortex"
    return ds


def randvec(
    n: Union[int, Tuple[int, int]] = 128,
    n_frames: int = 1,
    slope: float = 5.0 / 3.0,
    nc: float = 3.0,
    nl: Optional[float] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    seed: Optional[int] = None,
) -> xr.Dataset:
    """Generate divergence-free 2D random velocity fields with prescribed power spectrum.

    Parameters
    ----------
    n : int or tuple of (rows, cols)
        Grid dimension (default 128).
    n_frames : int
        Number of independent realization frames along 't' (default 1).
    slope : float
        Spectral decay exponent $E(k) \\propto k^{-\\text{slope}}$ (default 5/3).
    nc : float
        Small scale cutoff in grid units (default 3.0).
    nl : float, optional
        Large scale cutoff in grid units. Default is n/3.
    dx, dy : float
        Grid spacing (default 1.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Canonical PIVPy dataset with divergence-free random velocity fields.
    """
    if isinstance(n, int):
        rows, cols = n, n
    else:
        rows, cols = n

    if nl is None:
        nl = float(min(rows, cols)) / 3.0

    rng = np.random.default_rng(seed)
    x_coords = np.arange(cols, dtype=float) * dx
    y_coords = np.arange(rows, dtype=float) * dy

    kx = np.fft.fftfreq(cols, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(rows, d=dy) * 2.0 * np.pi
    kx_2d, ky_2d = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx_2d**2 + ky_2d**2)
    safe_k = np.where(k_mag == 0.0, 1.0, k_mag)

    k_c = 2.0 * np.pi / (nc * max(dx, dy))
    k_l = 2.0 * np.pi / (nl * min(dx, dy))

    spec = np.exp(-((k_mag / k_c) ** 2)) * (k_mag**2) / np.sqrt(1.0 + (k_mag / k_l) ** (2.0 * slope + 4.0)) / safe_k
    spec[k_mag == 0.0] = 0.0
    amp = np.sqrt(spec)

    frames = []
    for t_idx in range(n_frames):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(rows, cols))
        complex_noise = np.exp(1j * phase)

        psi_hat = amp * complex_noise / safe_k
        psi_hat[k_mag == 0.0] = 0.0

        u_hat = 1j * ky_2d * psi_hat
        v_hat = -1j * kx_2d * psi_hat

        u_real = np.real(np.fft.ifft2(u_hat))
        v_real = np.real(np.fft.ifft2(v_hat))

        u_real -= np.mean(u_real)
        v_real -= np.mean(v_real)

        chc = np.ones_like(u_real, dtype=float)
        ds_t = build_dataset(
            x=x_coords,
            y=y_coords,
            t=np.array([float(t_idx)], dtype=float),
            u=u_real[:, :, np.newaxis],
            v=v_real[:, :, np.newaxis],
            chc=chc[:, :, np.newaxis],
            delta_t=float(DELTA_T),
        )
        frames.append(ds_t)

    ds = xr.concat(frames, dim="t") if len(frames) > 1 else frames[0]
    ds.attrs["flow_model"] = "randvec"
    return ds


def channel(
    rows: int = 64,
    cols: int = 64,
    u_max: float = 1.0,
    dx: float = 1.0,
    dy: float = 1.0,
    frame: int = 0,
) -> xr.Dataset:
    """Generate an analytical laminar Poiseuille channel flow velocity field.

    Profile: $u(y) = U_{\\max} \\left[1 - \\left(\\frac{y - y_c}{H}\\right)^2\\right]$, $v(y) = 0$.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (default 64x64).
    u_max : float
        Centerline maximum velocity (default 1.0).
    dx, dy : float
        Grid spacing (default 1.0).
    frame : int
        Time frame index (default 0).

    Returns
    -------
    xr.Dataset
        Canonical PIVPy dataset with analytical Poiseuille flow.
    """
    x_coords = np.arange(cols, dtype=float) * dx
    y_coords = np.arange(rows, dtype=float) * dy
    x2d, y2d = np.meshgrid(x_coords, y_coords)

    y_min = y_coords[0]
    y_max = y_coords[-1]
    y_c = (y_min + y_max) / 2.0
    h = (y_max - y_min) / 2.0

    u = u_max * (1.0 - ((y2d - y_c) / h) ** 2)
    v = np.zeros_like(u, dtype=float)
    chc = np.ones_like(u, dtype=float)

    ds = build_dataset(
        x=x_coords,
        y=y_coords,
        t=np.array([float(frame)], dtype=float),
        u=u[:, :, np.newaxis],
        v=v[:, :, np.newaxis],
        chc=chc[:, :, np.newaxis],
        delta_t=float(DELTA_T),
    )
    ds.attrs["flow_model"] = "poiseuille_channel"
    return ds


def shear_layer(
    rows: int = 64,
    cols: int = 64,
    u0: float = 1.0,
    delta: float = 5.0,
    perturbation: float = 0.05,
    dx: float = 1.0,
    dy: float = 1.0,
    frame: int = 0,
) -> xr.Dataset:
    """Generate an analytical hyperbolic tangent shear/mixing layer with Kelvin-Helmholtz perturbation.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (default 64x64).
    u0 : float
        Free-stream velocity magnitude (default 1.0).
    delta : float
        Shear layer thickness in coordinate units (default 5.0).
    perturbation : float
        Relative amplitude of transverse periodic Kelvin-Helmholtz disturbance (default 0.05).
    dx, dy : float
        Grid spacing (default 1.0).
    frame : int
        Time frame index (default 0).

    Returns
    -------
    xr.Dataset
        Canonical PIVPy dataset with analytical shear layer.
    """
    x_coords = np.arange(cols, dtype=float) * dx
    y_coords = np.arange(rows, dtype=float) * dy
    x2d, y2d = np.meshgrid(x_coords, y_coords)

    y_c = (y_coords[0] + y_coords[-1]) / 2.0
    domain_w = x_coords[-1] - x_coords[0]
    k_pert = 2.0 * np.pi / domain_w

    u = u0 * np.tanh((y2d - y_c) / delta)
    v = np.zeros_like(u, dtype=float)

    if perturbation > 0.0:
        pert_v = perturbation * u0 * np.sin(k_pert * x2d) * np.exp(-(((y2d - y_c) / (2.0 * delta)) ** 2))
        pert_u = -perturbation * u0 * np.cos(k_pert * x2d) * ((y2d - y_c) / delta) * np.exp(-(((y2d - y_c) / (2.0 * delta)) ** 2))
        u += pert_u
        v += pert_v

    chc = np.ones_like(u, dtype=float)
    ds = build_dataset(
        x=x_coords,
        y=y_coords,
        t=np.array([float(frame)], dtype=float),
        u=u[:, :, np.newaxis],
        v=v[:, :, np.newaxis],
        chc=chc[:, :, np.newaxis],
        delta_t=float(DELTA_T),
    )
    ds.attrs["flow_model"] = "shear_layer"
    return ds
