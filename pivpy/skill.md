---
name: pivpy
description: >-
  Expert skill for deep Particle Image Velocimetry (PIV) vector field analysis, post-processing,
  vortex topology identification, spatial filtering, outlier cleaning, out-of-core streaming,
  interactive Marimo app exploration, and publication-ready figures/reports.
  Designed to work in tandem with openpiv-skill and scientific reporting workflows.
---

# PIVPy Skill: Deep PIV Analysis, Visualization & Reporting

This skill equips the agent with comprehensive domain expertise, workflows, and code recipes for processing, analyzing, and visualizing Particle Image Velocimetry (PIV) data using **PIVPy** (built on top of `xarray`, `numpy`, `scipy`, and `matplotlib`).

---

## 1. System Architecture & Interoperability

```
+---------------------+       +----------------------+       +-----------------------+
|   Raw Image Pairs   | ----> |    openpiv-skill     | ----> |  Raw Velocity Fields  |
|  (TIF, BMP, PNG)    |       | (cross-correlation)  |       |   (TXT, DAT, VC7, NC) |
+---------------------+       +----------------------+       +-----------+-----------+
                                                                         |
                                                                         v
+------------------------------------------------------------------------------------+
|                                    PIVPy Skill                                     |
|                                                                                    |
|  1. Canonical Ingestion:   build_dataset(), load_directory(), from_openpiv()       |
|  2. Out-of-Core Streaming: stream_directory_to_zarr(), stream_statistics()         |
|  3. Validation & Clean:    normalized_median_test(), clean(), harmonic inpainting  |
|  4. Spatial Filtering:     smooth(method='gaussian' | 'median' | 'butterworth')    |
|  5. Kinematic & Topology:  vorticity, Gamma1, Gamma2, Q-criterion, Okubo-Weiss    |
|  6. Gradient & Strain:     gradient_tensor(), max_shear(), acceleration()          |
|  7. Turbulence & Spectra:  Reynolds decomposition, E(k), dissipation, R_ij(r)     |
|  8. Interactive App:       ds.piv.explore(), Marimo reactive dashboard             |
|  9. Publication Figures:   piv.plot(), marimo notebooks, animations, LaTeX reports|
+------------------------------------------------------------------------------------+
```

---

## 2. Canonical Data Model (`xarray.Dataset`)

Every PIV field in PIVPy is represented as an `xarray.Dataset` containing:
- **Dimensions**: `('y', 'x')` for single frames, `('y', 'x', 't')` for time series / ensembles.
- **Coordinates**:
  - `x`: 1D array of horizontal positions $[x_0, x_1, \dots, x_{N-1}]$.
  - `y`: 1D array of vertical positions $[y_0, y_1, \dots, y_{M-1}]$.
  - `t`: 1D array of time stamps or frame indices $[t_0, t_1, \dots, t_{K-1}]$ (optional).
- **Required Data Variables**:
  - `u`: Horizontal velocity component $[M \times N]$ or $[M \times N \times K]$.
  - `v`: Vertical velocity component $[M \times N]$ or $[M \times N \times K]$.
  - `chc`: Vector validation flag channel ($1.0 = \text{valid}$, $0.0 = \text{spurious/outlier}$).
- **Attributes (`attrs`)**: `units_x`, `units_y`, `units_u`, `units_v`, `units_t`, `dt`, `history`.

---

## 3. Standard PIV Processing Pipeline

### Step 1: Ingestion & Out-of-Core Streaming

```python
import xarray as xr
import pivpy.pivpy  # Registers .piv accessor
from pivpy import io

# Option A: In-memory directory ingestion
ds = io.load_directory("path/to/openpiv_results/", ext=".txt")

# Option B: Out-of-core directory ingestion to chunked Zarr
io.stream_directory_to_zarr("path/to/raw_vc7_or_txt/", "dataset.zarr", chunks={"t": 1})
ds_lazy = io.read_directory_lazy("dataset.zarr")

# Option C: O(1) memory streaming statistics reduction across 10,000+ frames
stats_ds = io.stream_statistics("dataset.zarr")  # or ds.piv.stream_statistics()
```

### Step 1b: Raw LaVision Image Pairs -> PIV -> PIVPy (no vectors yet)

If you only have raw camera image pairs (LaVision `.im7`/`.vc7` buffers, not
pre-computed vectors), read them with `lvpyio`, run cross-correlation
yourself (e.g. with `openpiv`), and only then hand the result to PIVPy. Two
things real acquisitions get right that a first attempt easily gets wrong:

- **`.im7` buffers already hold both frames** - `lvpyio.read_buffer()`
  returns a 2-frame buffer directly; there's no need to split a stacked
  image (`buffer.as_masked_array(0)` / `(1)` are frame A / B).
- **The true PIV pulse separation (`dt`) is in the buffer's own metadata**,
  not something to assume or leave as a placeholder. LaVision timing
  channels are typically stored as `DevDataTrace<N>` / `DevDataAlias<N>`
  pairs; find the channel whose alias mentions "dt" (e.g. `"Reference time
  dt : dt 1"`), and its value is in microseconds:

```python
import lvpyio as lv
import numpy as np
import xarray as xr
import openpiv.pyprocess as pyprocess
from openpiv import tools, scaling, validation, filters
import pivpy  # registers .piv accessor

buffer = lv.read_buffer("B0001.im7")
dt = float(np.asarray(buffer.attributes["DevDataTrace5"]).flat[0]) * 1e-6  # us -> s
frame_a = np.asarray(buffer.as_masked_array(0).data)
frame_b = np.asarray(buffer.as_masked_array(1).data)

u, v, s2n = pyprocess.extended_search_area_piv(
    frame_a.astype(np.int32), frame_b.astype(np.int32),
    window_size=64, overlap=32, search_area_size=96, dt=dt,
    sig2noise_method="peak2peak",
)
x, y = pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=96, overlap=32)
invalid = validation.sig2noise_val(s2n, threshold=1.3)
u, v = filters.replace_outliers(u, v, invalid, method="localmean", max_iter=10, kernel_size=3)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=173.4)  # px/mm from calibration, not a guess
x, y, u, v = tools.transform_coordinates(x, y, u, v)

ds = xr.Dataset(
    data_vars={"u": (("y", "x"), u), "v": (("y", "x"), v), "chc": (("y", "x"), (~invalid).astype(float))},
    coords={"x": x[0, :], "y": y[:, 0]},
)
```

Sanity-check `dt` against known physics before trusting it (e.g. compare the
resulting mean speed to a reported/expected flow rate) - a wrong `dt` still
produces a plausible-looking, uniformly-wrong velocity field.

For a batch (many frames -> one time series), build one `xr.Dataset` per
frame this way and `xr.concat([...], dim="t")`; for a large batch, save the
result to Zarr immediately (`ds.to_zarr("run.zarr", mode="w")`) so later
analysis reloads in under a second instead of re-running PIV.

### Step 2: Outlier Rejection & Inpainting (Normalized Median Test)

```python
# Westerweel & Scarano (2005) Normalized Median Test + Harmonic Inpainting
ds_clean = ds.piv.clean(
    method="normalized_median",
    threshold=2.0,       # Residual threshold (typically 2.0)
    epsilon=0.1,         # Velocity noise floor
    inpaint_method=0,    # 0 = harmonic Laplacian, 1 = nearest, 2 = linear
    radius=1,            # 3x3 stencil (radius=1) or 5x5 stencil (radius=2)
)
```

### Step 3: Spatial Filtering & Denoising

```python
# Gaussian smoothing
ds_smooth = ds_clean.piv.smooth(sigma=1.2, method="gaussian")

# Frequency-domain Butterworth filter
ds_bw = ds_clean.piv.smooth(sigma=8.0, method="butterworth", order=2.0)
```

### Step 4: Vortex Identification & Kinematic Diagnostics

```python
# 1. Circulation-based noise-robust vorticity (77% error reduction vs standard diff)
ds_vort = ds_smooth.piv.vorticity(name="vorticity", method="circulation", radius=2)

# 2. Topology Identification: Gamma1 (vortex center) and Gamma2 (boundary)
ds_g1 = ds_smooth.piv.gamma1(name="gamma1", radius=3)
ds_g2 = ds_smooth.piv.gamma2(name="gamma2", radius=3)

# 3. Galilean-invariant Q-criterion & Okubo-Weiss parameter
ds_q = ds_smooth.piv.q_criterion(name="Q")
ds_ow = ds_smooth.piv.okubo_weiss(name="OW")

# 4. Solid-body rotation subtraction
ds_nobr = ds_smooth.piv.subsbr()
```

### Step 5: Velocity Gradient & Acceleration Analysis

```python
# Full strain rate tensor decomposition
tensor = ds_smooth.piv.gradient_tensor(return_components=True)
# Contains: s_xx, s_yy, s_xy, lambda_1, lambda_2, max_shear, strain_angle

# Total Material Acceleration D(u)/Dt (unsteady + convective)
ds_accel = ds_smooth.piv.acceleration(name="accel", unsteady=True)
```

---

## 4. Advanced Turbulence Statistics & Spectral Analysis

```python
# 1. Reynolds Decomposition & Reynolds Stress Tensor
# Splits velocity into mean and fluctuations: u = u_mean + u_prime
turb = ds.piv.reynolds_decomposition()
# Yields: u_mean, v_mean, u_prime, v_prime, uu_prime, vv_prime, uv_prime, tke, intensity_u, intensity_v

# 2. Kinetic Energy Spectra (2D & 1D Radial with Hann Windowing)
spec = ds.piv.energy_spectrum(window="hann", detrend=True, radial=True)
# Yields: E2D(ky, kx), E_kx(kx), E_ky(ky), E_radial(k)
# Satisfies Parseval relation: integral(E(k) dk) == TKE

# 3. Two-Point Spatial Autocorrelation Function
corr_u = ds.piv.spatial_correlation(component="u", dim="x", normalize=True)
# Yields: R(r) with R(0) = 1.0

# 4. Integral Length Scale (L11)
L11 = ds.piv.integral_length_scale(component="u", dim="x")  # Physical length units

# 5. Taylor Microscale (lambda_T)
lambda_curv = ds.piv.taylor_microscale(component="u", dim="x", method="curvature")
lambda_grad = ds.piv.taylor_microscale(component="u", dim="x", method="gradient")

# 6. Turbulent Dissipation Rate (epsilon)
ds_eps_dir = ds.piv.dissipation(method="direct", nu=1.5e-5, name="eps_dir")
ds_eps_iso = ds.piv.dissipation(method="isotropic", nu=1.5e-5, name="eps_iso")
ds_eps_sgs = ds.piv.dissipation(method="smagorinsky", name="eps_sgs")
```

---

## 5. Interactive Diagnostics & Publication Figures

### Interactive Marimo Web Explorer

```python
# Launch interactive real-time PIV explorer in web browser
ds.piv.explore()

# Or from command-line:
# marimo run pivpy/app.py
# python -m pivpy.app
```

### Beautiful Single-Frame Quiver & Contour Plots

```python
import matplotlib.pyplot as plt

# Generate high-contrast, publication-quality figure
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

ds.piv.plot(
    flow_property="vorticity",  # Background scalar
    cmap="RdBu_r",              # Diverging colormap
    clim=(-15, 15),             # Symmetric color limits
    cbar=True,
    cbar_label=r"Vorticity $\omega_z$ [s$^{-1}$]",
    quiver_scale=1.0,           # Optimized arrow scaling
    quiver_density=2,           # Subsample grid for clear arrow visibility
    quiver_color="k",
    quiver_width=0.003,
    quiver_alpha=0.75,
    ax=ax,
)

ax.set_xlabel(r"$x$ [mm]", fontsize=12)
ax.set_ylabel(r"$y$ [mm]", fontsize=12)
ax.set_title("Vortical Wake Evolution", fontsize=14, pad=10)
fig.tight_layout()
fig.savefig("figure_vortex_wake.pdf", bbox_inches="tight")
```

### PIVlab-Style Image Overlay (Colored Quiver on the Raw Frame)

`ds.piv.plot()`'s `background=` normally draws a *computed* scalar contour
(vorticity/magnitude/etc.), never a raw camera image, and its quiver arrows
are a flat color unless colored by speed. For the classic PIV-software look
(dense/auto-scaled arrows, continuous colormap, raw tracer image behind
them), use `background="image"` with `color_by=`:

```python
fig, ax = ds.piv.plot(
    background="image", image=frame_a,       # raw grayscale camera frame (2D array)
    image_extent=None,                       # None -> uses ds's x/y range
    image_cmap="gray", image_alpha=0.6,
    color_by="v",                            # or "u", "mag"/"speed", any var name
    cmap="viridis",                          # colormap for the arrows (color_by only)
    streamlines=False, quiver_key=False,
)
```

`ds.piv.animate()` supports the same `background="image"`, `color_by`,
`cmap`, `image_cmap`, `image_extent`, `image_alpha` — the frame is drawn
once (assumed the same background for every t), and the quiver's color and
direction update per frame:

```python
anim = ds.piv.animate(
    background="image", image=frame_a, color_by="v", cmap="viridis", interval=100,
)
anim.save("flow_overlay.gif", writer="pillow", fps=10)
```

For quick interactive tuning in a marimo notebook before committing to
values for a batch pipeline, drive these kwargs from `mo.ui` controls
(`mo.ui.dropdown` for `color_by`/`cmap`/`image_cmap`, `mo.ui.slider` for
`image_alpha`, `mo.ui.checkbox` for `streamlines`) — each `.value` read in
a cell that calls `ds.piv.plot(...)` re-renders automatically on change.

### Fluid Dynamics Quiver Animation (MP4 / GIF)

```python
anim = ds.piv.animate(
    flow_property="vorticity",
    cmap="Spectral_r",
    quiver_scale=0.8,
    quiver_density=3,
    quiver_color="midnightblue",
    quiver_width=0.0035,
    quiver_alpha=0.7,
    fps=15,
    blur=0.6,
    save_path="vortex_pair_dynamics.gif",
)
```

---

## 6. Multiphase PIV/PTV Analytics & Spatial Turbulence Budgets (WP6 & WP7)

### Multiphase & Lagrangian-Eulerian Analysis (WP6)
- **Radial Distribution Function ($g(r)$)**: Measures preferential concentration and particle clustering against Poisson null hypothesis with domain boundary correction.
- **Eulerian-Lagrangian Sub-grid Sampling (`interp_at_points`)**: Interpolates continuous fluid fields (velocity, vorticity, $Q$-criterion, shear) onto Lagrangian particle tracks to calculate slip velocity $\mathbf{u}_p - \mathbf{u}_f$.
- **PTV Sparse-to-Dense Grid Reconstruction (`ptv_to_grid`)**: Bins scattered particle tracks onto structured `xarray.Dataset` grids with local particle concentration and velocity statistics.
- **Windowed 1D Spectral Density (`spectrum_1d`)**: Computes 1D spatial and temporal energy spectra with Hanning/Hamming tapering and segment overlap.

### Spatial Turbulence Budgets & Robust Differentiation (WP7)
- **TKE Production Rate ($P_k$)**: Computes $-\overline{u'u'}\frac{\partial \overline{u}}{\partial x} - \overline{u'v'}\left(\frac{\partial \overline{u}}{\partial y} + \frac{\partial \overline{v}}{\partial x}\right) - \overline{v'v'}\frac{\partial \overline{v}}{\partial y}$.
- **Turbulence Intensity Field**: $I_u = u_{rms} / U_{ref}$, $I_v = v_{rms} / U_{ref}$, $I_{total}$.
- **Least-Squares Spatial Derivatives (`lsgradient`)**: 2D polynomial surface fitting to suppress experimental measurement noise during gradient calculation.
- **Grubbs Outlier Filter (`grubbs_filter`)**: Statistical maximum normed residual outlier detection.
- **Transect & Slice Extractor (`extract_profile`)**: Extracts 1D profiles across arbitrary spatial transects with confidence intervals.

---

## 7. Automated Deep PIV Analysis Reports

When generating comprehensive analysis reports for experiments:
1. **Quality Audit Table**: Report total vectors, percentage of valid vs inpainted vectors, mean velocity magnitude, peak Reynolds stresses, and vortex core circulation $\Gamma$.
2. **Multi-Panel Overview**:
   - Panel A: Streamwise & transverse velocity contours ($u, v$).
   - Panel B: Circulation vorticity & $\Gamma_2$ vortex boundary contours.
   - Panel C: $Q$-criterion & Okubo-Weiss topology partitioning.
   - Panel D: Maximum shear strain rate & material acceleration.
   - Panel E: Turbulent kinetic energy & Reynolds shear stress $-\overline{u'v'}$.
   - Panel F: Radial energy spectrum $E(k)$ with Kolmogorov $-5/3$ reference slope.
3. **Integral Quantities Table**: Report integral length scale $L_{11}$, Taylor microscale $\lambda_T$, turbulent Reynolds number $\text{Re}_\lambda = u_{\text{rms}} \lambda_T / \nu$, and dissipation rate $\varepsilon$.
4. **Artifact Generation**: Save high-resolution PNG/PDF figures and export analysis summary markdown artifacts.

