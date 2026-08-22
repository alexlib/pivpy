---
name: pivpy
description: >-
  Expert skill for deep Particle Image Velocimetry (PIV) vector field analysis, post-processing,
  vortex topology identification, spatial filtering, outlier cleaning, and publication-ready figures/reports.
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
|  2. Validation & Clean:    normalized_median_test(), clean(), harmonic inpainting  |
|  3. Spatial Filtering:     smooth(method='gaussian' | 'median' | 'butterworth')    |
|  4. Kinematic & Topology:  vorticity, Gamma1, Gamma2, Q-criterion, Okubo-Weiss    |
|  5. Gradient & Strain:     gradient_tensor(), max_shear(), acceleration()          |
|  6. Turbulence & Spectra:  Reynolds decomposition, E(k), dissipation, R_ij(r)     |
|  7. Publication Figures:   piv.plot(), marimo notebooks, animations, LaTeX reports|
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

### Step 1: Ingestion from OpenPIV or Files

```python
import xarray as xr
import pivpy.pivpy  # Registers .piv accessor
from pivpy import io

# Option A: From OpenPIV output files / directories
ds = io.load_directory("path/to/openpiv_results/", extension=".txt")

# Option B: From OpenPIV arrays directly in memory
from pivpy.schema import build_dataset
ds = build_dataset(x=x, y=y, u=u, v=v, chc=flags, dt=0.001)

# Option C: Out-of-core streaming from Zarr archive
ds = io.open_zarr("dataset.zarr")
```

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

## 4. Publication-Ready Visualizations

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

## 5. Automated Deep PIV Analysis Reports

When generating comprehensive analysis reports for experiments:
1. **Quality Audit Table**: Report total vectors, percentage of valid vs inpainted vectors, mean velocity magnitude, peak Reynolds stresses, and vortex core circulation $\Gamma$.
2. **Multi-Panel Overview**:
   - Panel A: Streamwise & transverse velocity contours ($u, v$).
   - Panel B: Circulation vorticity & $\Gamma_2$ vortex boundary contours.
   - Panel C: $Q$-criterion & Okubo-Weiss topology partitioning.
   - Panel D: Maximum shear strain rate & material acceleration.
3. **Artifact Generation**: Save high-resolution PNG/PDF figures and export analysis summary markdown artifacts.
