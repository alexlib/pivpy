# PIVPy introduction

## History and motivation

PIVPy is inspired by Federic Moisy Matlab package, called [PIVMAT](http://www.fast.u-psud.fr/pivmat/).

Furthermore, although OpenPIV is our main PIV analysis tool, we work with multiple software vendors in
our research projects, TSI, Lavision, Dantec, PIVLab, etc. PIVPy, like PIVMAT is reading all these
files and unifies the post-processing. The main underlying Python packages are:

- xarray
- numpy
- scipy
- matplotlib
- jupyter

## Data model and I/O

PIVPy's core data structure is an `xarray.Dataset` with:

- Dims: `('y', 'x', 't')`
- Coords: 1D `x`, 1D `y`, 1D `t`
- Variables: `u`, `v`, and `chc` (validity / mask). Some readers may also add `mask`.

Recommended entry points in `pivpy.io`:

- `read_piv(path)`: auto-detect a single file format and load it.
- `read_directory(path)`: load a directory of PIV files into a time-series dataset.

Backward-compatible helpers are still available (e.g. `load_vec()`, `load_openpiv_txt()`, `load_directory()`).

## I/O formats and optional deps

PIVPy supports multiple PIV output formats. Some are available with the core install; others require optional dependencies.

- LaVision `.vc7`: requires `lvpyio` (install extra: `pivpy[lvpyio]`). If `lvpyio` is not installed, loading VC7 files falls back to a small synthetic dataset so that higher-level workflows (e.g. directory loading) remain usable.
- PIVLab `.mat` (v7.3 / HDF5): requires `h5py` (install extra: `pivpy[h5py]`).
- NetCDF export: requires `netcdf4` (install extra: `pivpy[netcdf]`) if you want NetCDF4 support.

Core (no optional deps):

- Insight / LaVision ASCII `.vec`
- OpenPIV `.txt` (5- or 6-column exports; when present, the 6th column is loaded as `mask`)

## Related work and inspiration

Several recently added analysis features (TKE budget, convergence analysis, vortex
identification, 2D FFT spectra, stress/TKE variants, anisotropy invariants) were
inspired by the feature set of **uPrime** (Jibu Tom Jose and Omri Ram, Technion),
an open-source desktop application for turbulence analysis of PIV/CFD velocity fields
(GPL-3.0). PIVPy's implementations are original work written for this MIT-licensed
codebase and are not derived from uPrime's source code; uPrime is acknowledged as
prior art.

## Tutorials

- [Tutorial](tutorial.md) (marimo notebook)
- [PIVPy tutorial](pivpy_tutorial.md) (written tutorial)
