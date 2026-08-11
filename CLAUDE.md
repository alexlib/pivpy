# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PIVPy is a Python package for post-processing Particle Image Velocimetry (PIV) data. It merges three
predecessor packages (Vecpy, alexlib/pivpy-xarray, ronshnapp/vecpy). Core data structure is an
`xarray.Dataset` (coords `x, y, t`; data vars `u, v, chc`) extended via a registered `.piv` accessor,
plus a large library of PIVMat-toolbox-inspired analysis functions.

## Commands

Install (uv is the primary workflow; pip also works):

    uv venv
    uv pip install -e '.[full]'   # full pulls in lvpyio, readim, netcdf4, vortexfitting, h5py

Core install no longer requires the `netcdf4` or `vortexfitting` C-extension packages: NetCDF I/O
goes through `h5netcdf` (built on `h5py`, already a hard dep) by default, and `vortexfitting`
(only used by `pivpy.interfacing.pivpyTOvf`) is optional and fails with a clear `ImportError` if
missing. This keeps the base install installable on Python builds where those packages don't yet
ship wheels (e.g. free-threaded/no-GIL CPython) — verified working end-to-end, `zarr`'s
`numcodecs` codec re-enables the GIL at import time (it hasn't declared free-threading safety
yet), so this isn't a true GIL-free run, but nothing crashes and the full test suite passes.
Install the classic netCDF4 C library engine explicitly with `pip install pivpy[netcdf]` if you
need it.

Run all tests:

    uv run pytest -q

Run a single test file / test:

    uv run pytest tests/test_io.py -q
    uv run pytest tests/test_io.py::test_read_piv_vec -q

Run against a specific managed Python version (sandboxed, no persistent venv):

    uv run --isolated --managed-python -p python3.14 --with-editable . pytest -q

Build docs (Sphinx, notebooks under docs/source converted to .rst):

    uv pip install -r docs/requirements.txt
    uv run sphinx-build -b html docs/source/ docs/build/html

There is no configured lint/format command in this repo (no ruff/black config present) — `mypy.ini` exists
but is minimal (just enables the numpy typing plugin); there's no CI-enforced mypy run to match.

## Architecture

- `pivpy/pivpy.py` — the `PIVAccessor` class registered as `Dataset.piv` via `@xr.register_dataset_accessor("piv")`.
  This is the primary public API surface (e.g. `ds.piv.average`, `ds.piv.vorticity()`, `ds.piv.crop()`,
  `ds.piv.averf()`, `ds.piv.filterf()`). Accessor methods are thin wrappers that delegate the actual math to
  `pivpy/compute_funcs.py`, imported at the top with a `c`/`g` prefix convention (`cinterpf`, `cprobef`,
  `gquiver`, `gshowf`, ...) to disambiguate from the accessor's own method names.
- `pivpy/compute_funcs.py` (~3500 lines) — the computational core: filtering (`filter2d`, `bwfilter2d`),
  correlation (`corrx`, `corrm`, `corrf`), spectra (`specf`, `spec2f`, `tempspecf`), statistics (`statf`,
  `jpdfscal`, `histf`), interpolation/hole-filling (`interpolat_zeros_2d`, `inpaint_missing_2d`, `interpf`),
  and PIVMat-named transforms (`shiftf`, `smoothf`, `truncf`, `subsbr`, `zeropadf`, etc). Most functions
  operate on `xr.Dataset | list[xr.Dataset]` and are designed to mirror the MATLAB PIVMat toolbox API/naming
  (see README's "PIVMat-inspired methods" section for the mapping).
- `pivpy/schema.py` — the canonical dataset schema shared by every reader: `build_dataset()` (the one
  constructor readers should call instead of hand-rolling `xr.Dataset(...)`), `validate()`/`is_valid()`,
  and `set_default_attrs()`. See `docs/architecture/zarr-migration.md` for the design rationale.
- `pivpy/io.py` — file I/O. Two layers coexist intentionally:
  - **Current**: `read_piv(filepath, format=None, **kwargs)` auto-detects format and dispatches through a
    `PIVReader` ABC registry (`PIVReaderRegistry`, `register_reader`) with concrete readers per format
    (`InsightVECReader`, `OpenPIVReader`, `Davis8Reader`, `LaVisionVC7Reader`, `PIVLabReader`, `NetCDFReader`,
    `ZarrReader`). Each reader has a canonical `.name` (e.g. `"insight"`, `"openpiv"`, `"vc7"`) used both by
    `format=` dispatch (`PIVReaderRegistry.get_by_name`) and auto-detection, so a custom reader registered
    via `register_reader()` is selectable either way. Pair with `save_piv()` (supports `format="netcdf"`
    (default), `"csv"`, `"zarr"`). For large/out-of-core directories, prefer `convert_directory_to_zarr()` +
    `read_directory_lazy()` over the eager `read_directory()` (which loads every frame into memory before
    concatenating).
  - **Legacy**: `load_vec`, `load_openpiv_txt`, `load_vc7`, `loadvec`, `openvec`, `openim7`, etc. — kept for
    backward compatibility; new format support should go through the `PIVReader` registry, not this layer.
  - Also home to `batchf()` (PIVMat-style batch processing over filename glob patterns, accepting either a
    callable or an accessor method name as the function to apply).
- `pivpy/pivmat_compat.py` — compatibility shims specifically for reading PIVMat-format data.
- `pivpy/graphics.py` — plotting (quiver, streamplot, histograms, movies); accessor methods in `pivpy.py`
  delegate to these with a `g` prefix, mirroring the `compute_funcs.py` pattern.
- `pivpy/davis_readim.py` — LaVision Davis/im7 format reading support.
- `pivpy/update.py` — PyPI version-check helper (`pivpy.check_update()`).
- `pivpy/data/` — bundled sample datasets used throughout the test suite and README examples (e.g.
  `pivpy/data/openpiv_vec/exp1_001_b.vec`).

## Conventions worth knowing before editing

- New analysis functions belong in `compute_funcs.py`; expose them on the accessor in `pivpy.py` by importing
  with the `c`-prefix convention and adding a thin method — don't put heavy logic directly in `pivpy.py`.
- New file formats belong in `io.py` as a `PIVReader` subclass registered via `register_reader`, not as a new
  standalone `load_*`/`open*` function (those are legacy-only).
- Function names ending in `f` (e.g. `averf`, `spaverf`, `filterf`, `corrf`) intentionally mirror PIVMat
  toolbox naming/behavior for users porting MATLAB workflows — keep that naming convention for further
  PIVMat-parity additions.
