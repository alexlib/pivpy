# PIVPy Future Analysis Plan

Status: planning document (not yet implemented).
Scope: analysis features to be added to `pivpy` (MIT), informed by a comparison with
the GPL-licensed `uPrime` project. **Only algorithms and ideas are reused here; no code
from uPrime is copied.** POD, DMD and 3D FFT spectra are intentionally excluded and
will be implemented in separate packages.

Related docs:
- `docs/architecture/pivmat-parity-roadmap.md`
- `docs/architecture/zarr-migration.md`
- `ROADMAP.md`

---

## Guiding principles

1. **License hygiene.** pivpy stays MIT. Every new feature is implemented from the
   cited literature and from pivpy's own existing code, never ported from uPrime's
   GPL-3.0 source.
2. **Data model.** All functions operate on the canonical `xr.Dataset` schema
   (`coords x, y, (t)`; vars `u, v, (w)`, `chc`; attrs units/dt) and are exposed both
   in `compute_funcs.py` and as thin methods on the `.piv` accessor (`pivpy/pivpy.py`),
   following the `c`/`g` prefix convention.
3. **Out-of-core friendly.** Where feasible, support Dask-backed / Zarr-backed inputs
   and keep memory O(grid) rather than O(grid × time) (Welford-style), consistent with
   `io.stream_statistics`.
4. **Naming.** Keep PIVMat-style `*f` names for parity additions; new non-PIVMat
   features use descriptive snake_case names plus an accessor method.
5. **Testing.** Each feature ships with synthetic-field tests (known analytical answer)
   plus a regression test on bundled sample data, mirroring the existing suite.

## Implementation order

1. TKE budget
2. Convergence analysis
3. Vortex identification (detection + spatial probability)
4. FFT spectra (2D only)
5. Stress and TKE variants (std-dev, line profiles)
6. Anisotropy invariants + barycentric / Lumley triangle

POD, DMD and 3D spectra are out of scope for this plan.

---

## 1. TKE budget

### Goal
Compute the terms of the turbulent kinetic energy (TKE) budget on a 2D velocity field
ensemble, following standard formulations (Pope 2000).

### Terms
For incompressible flow with velocity $(u,v)$, mean $(\bar u,\bar v)$ and fluctuations
$(u',v')$, on a uniform grid:

- **Mean advection** (convection of TKE by the mean flow): $\bar u_j \partial_j k$
- **Turbulent transport**: $\partial_j \overline{u_j' k}$ (or split into pressure
  transport + turbulent diffusion)
- **Production**: $P = -\overline{u_i' u_j'}\, \partial_j \bar u_i$
- **Dissipation**: $\varepsilon = \nu\, \overline{(\partial_j u_i')^2}$ (or
  $\nu\,\overline{\partial_k u_i' \partial_k u_i'}$)
- **Viscous diffusion**: $\nu\, \partial_j \partial_j k$
- **Pressure transport**: $-\frac{1}{\rho}\partial_i \overline{p' u_i'}$ (optional,
  requires pressure; usually reported as residual)
- **Unsteady term** (time-resolved data): $\partial k/\partial t$
- **Residual** = production − dissipation + transport − advection (for closure check)

### API sketch
```python
# compute_funcs.py
def tke_budget(
    ds: xr.Dataset,
    *,
    nu: float | None = None,  # kinematic viscosity; if None, dissipation skipped
    rho: float = 1.0,
    method: str = "central",  # gradient scheme: central | least_squares
    smooth: bool = False,
) -> xr.Dataset:
    """Return a Dataset with data vars: k, production, dissipation, advection,
    turbulent_transport, viscous_diffusion, (pressure_transport), residual, ..."""
```
Accessor: `ds.piv.tke_budget(nu=..., ...)`. Also expose individual terms
(`ds.piv.tke_production()`, etc.) for users who want a subset.

### Implementation notes
- Reuse existing `gradient_tensor`, `reynolds_decomposition`, `reynolds_stresses`,
  `tke_turb` from `compute_funcs.py`.
- Handle NaNs/inpainting via existing `inpaint_missing_2d` or mask; report filled
  fraction in attrs.
- Uniform-grid assumption; document. Non-uniform grids: defer (reuse `non_uniform_spectra`
  machinery later).
- For time-resolved data, compute unsteady term from temporal derivative of $k$.

### Tests
- Synthetic: uniform shear flow with known $P$ and $\varepsilon$; channel-flow-like
  field with zero advection; residual ≈ 0 for a constructed field satisfying
  the budget.
- Sample data: run on `pivpy/data` sample; check shapes/units attrs; no NaNs in valid region.

### Dependencies
numpy, scipy, xarray (already). No new deps.

---

## 2. Convergence analysis

### Goal
Quantify how many snapshots are needed for turbulence statistics (mean velocity,
Reynolds stresses, TKE) to converge, given a time series of PIV fields.

### API sketch
```python
def convergence_analysis(
    ds: xr.Dataset,
    *,
    quantities: list[str] = ["u", "v", "uu", "vv", "uv", "k"],
    kernel: str = "cumulative",  # cumulative | moving (window)
    window: int | None = None,
    threshold: float = 0.05,  # relative change threshold
    point: tuple[int, int] | None = None,  # grid point; None = spatial mean
) -> xr.Dataset:
    """Returns per-quantity series vs. N plus estimated convergence N."""
```
Accessor: `ds.piv.convergence(...)`.

### Implementation notes
- Reuse Welford accumulation from `io.stream_statistics`; extend to track per-N
  estimates of the requested quantities (not just final).
- `find_convergence_n`: smallest N such that relative change of the quantity over
  the last `m` snapshots stays below `threshold` for `p` consecutive steps.
- Support both point-wise and spatially-averaged convergence.
- Out-of-core: stream from Zarr/directory without loading all frames.

### Tests
- Synthetic: field with known mean; convergence N should equal the period/length
  of the deterministic component.
- Monotonicity: convergence N non-increasing as threshold relaxes.

---

## 3. Vortex identification (detection + spatial probability)

### Goal
Beyond per-point criteria (already implemented: Γ₁, Γ₂, Q, Okubo-Weiss, circulation
vorticity), add:
- **Vortex detection**: connected-component labeling of regions satisfying a criterion
  threshold and sign, returning vortex centers, area, circulation, peak vorticity.
- **Spatial probability**: probability that a given grid point belongs to a vortex
  (of a given sign) across the ensemble.

### API sketch
```python
def detect_vortices(
    ds: xr.Dataset,
    *,
    criterion: str = "gamma2",  # gamma1 | gamma2 | q | ow | circulation
    threshold: float = 0.5,  # |criterion| >= threshold for vortex region
    sign: str = "all",  # positive | negative | all
    min_area: int = 4,  # min connected-component size (grid cells)
) -> list[dict]:  # list of {center, area, circulation, peak, ...}
    ...


def spatial_vortex_probability(
    ds: xr.Dataset,
    *,
    criterion: str = "gamma2",
    threshold: float = 0.5,
    sign: str = "all",
) -> xr.DataArray:  # probability per grid point
    ...
```
Accessors: `ds.piv.detect_vortices(...)`, `ds.piv.vortex_probability(...)`.

### Implementation notes
- Use `scipy.ndimage.label` / `measure.label` (scipy already a dependency) for
  connected components; avoid new heavy deps.
- Circulation per vortex via line integral around the labeled region boundary (reuse
  gradient/circulation helpers).
- Spatial probability: fraction of snapshots where point is in a vortex region;
  stream over time to keep memory O(grid).

### Tests
- Synthetic: single Burgers/Lamb-Oseen vortex → exactly one detected vortex at known
  center; probability = 1 at center for that snapshot, 0 elsewhere.
- Multi-vortex field → correct count within tolerance.
- Sign filtering works.

---

## 4. FFT spectra (2D only)

### Goal
Spatial wavenumber spectra $E(k)$ via FFT, 2D only (3D is out of scope here). Complement
the existing `specf`/`spec2f`/`energy_spectrum` with a faster FFT-based path and
azimuthal averaging.

### API sketch
```python
def fft_spectrum_2d(
    ds: xr.Dataset,
    *,
    component: str = "u",  # u | v | (u,v) combined
    window: str = "hann",  # hann | none
    detrend: bool = True,
    azimuthal: bool = True,  # radial average -> E(k)
    Lx: float | None = None,
    Ly: float | None = None,  # domain sizes (from attrs if None)
) -> xr.Dataset:  # k, E(k) (and optionally 2D spectrum)
    ...
```
Accessor: `ds.piv.fft_spectrum(...)`.

### Implementation notes
- Use `numpy.fft` (no pyFFTW dependency for the 2D path; keep optional fft extra for
  performance only).
- Handle non-square grids; document wavenumber binning.
- Reuse `_kx_ky_from_coords`-style helpers already in `compute_funcs.py`.

### Tests
- Synthetic: isotropic turbulence-like field with known $-5/3$ slope over a range;
  check slope within tolerance.
- Single Fourier mode → delta peak at correct wavenumber.

---

## 5. Stress and TKE variants

### Goal
Extended Reynolds-stress/TKE variants beyond the existing `reynolds_stresses`,
`reynolds_stress`, `tke`, `tke_turb`, `rms`:
- Per-component and tensor standard deviations / uncertainties
  (`reynolds_stress_std`, `tke_std`).
- Line profiles of stresses along an arbitrary line (reuse `sample_along_line` idea).
- Optional normalization (by $U_\infty^2$, by $k$, etc.).

### API sketch
```python
def reynolds_stress_std(ds, *, ddof: int = 1) -> xr.Dataset: ...
def tke_std(ds, *, mode: str = "2d") -> xr.DataArray: ...
def stress_line_profile(ds, x0, y0, x1, y1, *, component: str = "uu") -> xr.Dataset: ...
```
Accessors added accordingly.

### Implementation notes
- Reuse existing `reynolds_stresses`, `extract_line_profile`-style sampling, and
  `probe` helpers.
- Keep units attrs consistent.

### Tests
- Synthetic: constant field → std = 0; known-variance field → matches analytic.

---

## 6. Anisotropy

### Goal
Anisotropy invariant analysis: anisotropy tensor $b_{ij}$, invariants II/III,
barycentric coordinates, and Lumley triangle data, from the Reynolds stress tensor.

### API sketch
```python
def anisotropy_tensor(ds) -> xr.Dataset:   # b_ij fields
def anisotropy_invariants(ds) -> xr.Dataset:  # -II, III per point
def barycentric_map(ds) -> xr.Dataset:        # (x_b, y_b) per point for Lumley triangle
```
Accessors: `ds.piv.anisotropy(...)`, etc. Plotting helper in `graphics.py` for the
Lumley triangle (optional).

### Implementation notes
- Build on `reynolds_stresses`/`stresstensor` already in pivpy.
- Handle 2C (planar) by assuming $w'=0$ or using available components; document.
- No new dependencies.

### Tests
- Synthetic: isotropic turbulence → invariants at origin (II=III=0), barycentric at
  center; 1C/2C limits at triangle vertices.

---

## Explicitly out of scope (other packages)
- **POD** (snapshot POD + reconstruction)
- **DMD** (with gap handling)
- **3D FFT spectra**

These will be implemented in dedicated packages and are not part of this plan.

---

## Milestones

- M1: TKE budget + tests + accessor + docs (1 PR)
- M2: Convergence analysis (1 PR)
- M3: Vortex detection + spatial probability (1 PR)
- M4: FFT 2D spectra (1 PR)
- M5: Stress/TKE variants (1 PR)
- M6: Anisotropy (1 PR)

Each PR follows repo conventions: logic in `compute_funcs.py`, thin accessor in
`pivpy.py` with `c`-prefix import, tests in `tests/`, docs in `docs_mkdocs/`.

## License note
This plan describes algorithms and APIs only. Implementation must be original work
based on the cited literature and on pivpy's existing code, not derived from
uPrime's GPL-3.0 source. uPrime is acknowledged as prior art / related tool.