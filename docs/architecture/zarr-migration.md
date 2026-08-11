# PIVPy: Zarr-first storage + intuitive API — architecture plan

## Status

- **Phase 1 (PoC)**: done — `ZarrReader`, `save_piv(format="zarr")`, round-trip/lazy tests.
- **Phase 2**: done — `pivpy/schema.py` (`build_dataset`/`validate`/`is_valid`), all 6 readers
  migrated to `build_dataset`, `OpenPIVReader` mask→chc fold, `LaVisionVC7Reader` raises
  `ImportError` instead of silently fabricating data, registry/`read_piv` dispatch dedup
  (`PIVReader.name` + `PIVReaderRegistry.get_by_name`), `convert_directory_to_zarr` +
  `read_directory_lazy`, `pivpy_schema_version` attr, schema-conformance/out-of-core/attrs tests.
  `zarr`/`dask` are hard dependencies.
- **Phase 3**: partly done —
  - `__mul__`/`__div__` no longer mutate the caller's Dataset in place (return a new Dataset);
    `__div__` was also renamed to `__truediv__` since Python 3 never calls `__div__` for `/` (it
    was dead code).
  - Scalar-producing methods (`vorticity`, `strain`, `divergence`, `acceleration`,
    `kinetic_energy`, `tke`, `reynolds_stress`, `rms`) now warn (`UserWarning`, via
    `compute_funcs.warn_if_overwriting_scalar`) when their default `name="w"` is about to silently
    overwrite an existing variable — the collision itself is still allowed (non-breaking default),
    it's just visible now.
  - `crop()`/`vec2scal()` now emit a `DeprecationWarning` documenting that reassigning
    `self._obj` internally will go away in a future release — first step of the deprecation cycle,
    behavior unchanged.
  - The return-type contract (Dataset vs `(fig, ax)` vs raw tuple for `azprofile`) is now
    documented in `PIVAccessor`'s class docstring.
  - `save_piv`'s default `format` flipped from `"netcdf"` to `"zarr"` — `zarr`/`dask` are always
    available, `netcdf4` no longer is (see dependency note below). `save_piv(ds, path)` with no
    `format=` now writes a Zarr store (a directory), not a `.nc` file.
  - Still open: actually removing the `self._obj` reassignment in `crop`/`vec2scal` (kept as a
    `DeprecationWarning` for now — deliberately not removed in the same pass the warning was
    added; give it a release before changing behavior).
  - Also landed as part of this pass (dependency hygiene, prompted by wanting to validate the
    package installs cleanly on newer Python builds): `netcdf4` and `vortexfitting` moved from
    hard dependencies to optional extras (`pivpy[netcdf]`, `pivpy[vortexfitting]`); default NetCDF
    I/O now goes through `h5netcdf` (built on the already-required `h5py`) instead of the
    `netcdf4` C-extension package. Verified the package installs and the full test suite passes
    on a free-threaded (no-GIL) CPython 3.14 build — `zarr`'s `numcodecs` codec re-enables the GIL
    at import time (undeclared free-threading safety), so it's not a GIL-free run in practice yet,
    but nothing breaks.

## Context

PIVPy's current storage story is NetCDF/HDF5-based and fully eager (no lazy loading, no
`dask`/`zarr` anywhere — verified: neither package is even installed). For the repo's stated
goal — becoming the default post-PIV tool for grad students and the community, handling very
large datasets from Davis VC7 / OpenPIV / PIVLab out-of-core — this doesn't scale: loading a
directory of thousands of frames currently means holding every frame in memory before a single
`xr.concat`. Users also report the API isn't intuitive; investigation below shows why (inconsistent
return types, silent in-place mutation, a still-live "overwrite variable w" footgun, two formats
of the same validity concept). This plan targets both problems together, since fixing storage
without fixing the API (or vice versa) won't move the "default tool" needle by itself.

Deliverable for this pass: an in-repo design doc (this plan, committed as
`docs/architecture/zarr-migration.md`) plus a small, additive proof-of-concept that proves the
Zarr direction works end-to-end. No accessor rewrite yet — that's staged for later.

## Current-state findings (why this plan looks the way it does)

- **Storage**: `save_piv()` (`pivpy/io.py:918`) only supports `netcdf` (bare `dataset.to_netcdf()`,
  zero encoding/chunking options) and `csv`. `NetCDFReader.read()` uses plain `xr.open_dataset()`
  — always eager. HDF5 appears only via `h5py` inside `PIVLabReader` for `.mat` v7.3 files, and via
  the optional `lvpyio` for VC7 — both fully materialize into memory. No `chunks=` anywhere.
- **Large-dataset risk**: `read_directory()` (`io.py:901`) builds a Python list of *every* per-file
  Dataset before one `xr.concat(dim="t")` — for "thousands of Davis VC7 frames" this is a direct
  memory-blowup path. `batchf()` (`io.py:963`) is the one file-by-file-safe utility, but it only
  supports reduce-per-file scripting, not "give me one lazy Dataset over a big directory."
- **Reader architecture**: `PIVReader` ABC + `PIVReaderRegistry` (`io.py:402-856`) is a genuinely
  good, extensible plugin pattern (subclass, register, done) — but `read_piv(format=...)`
  (`io.py:866`) has a *second*, independent hardcoded string-dispatch table, so a custom reader
  registered via `register_reader()` can only ever be found by auto-detection, never selected
  explicitly by name. Two sources of truth for "what formats exist."
- **Correctness trap**: `LaVisionVC7Reader` (`io.py:629`) silently falls back to a *fabricated
  synthetic dataset* when the optional `lvpyio` package is missing, instead of raising — anyone
  without `lvpyio` who runs real workflows gets fake data with no error.
- **Data model drift**: dim order `('y','x','t')` is consistently enforced, but each of the 6
  readers hand-builds its own `xr.Dataset(...)` literal rather than sharing one constructor
  (`from_arrays()` exists but only 3/6 readers use it) — attrs values/keys differ by reader.
  `OpenPIVReader` additionally creates a *second* validity variable, `mask`, alongside the
  canonical `chc`, with overlapping but not identical semantics.
- **Accessor API** (`Dataset.piv`, `pivpy/pivpy.py`, ~55 public methods): three different return
  shapes depending on method (`Dataset` for most, raw tuple for `azprofile`, `(fig, ax)` for
  plotting methods) with no documented contract. `__mul__`/`__div__` (`pivpy.py:2277-2317`) mutate
  the underlying DataArray buffers **in place** (`self._obj["u"] *= scalar`) — visible to any other
  alias of the same Dataset, the most surprising mutation in the class. `crop()`/`vec2scal()`
  reassign `self._obj` (accessor-local mutation, inconsistent with the ~50 other methods that just
  return). The historical "all scalar methods overwrote `data['w']`" bug (CHANGES_SUMMARY.md) is
  mitigated (every such method now takes `name: str = "w"`) but not prevented — default is still
  the collision-prone `"w"`.
- **Tests**: 25 focused files, good per-feature coverage, but zero round-trip serialization tests
  (existing netCDF save test only checks the file exists, and explicitly catches/skips known
  `RuntimeError`s rather than asserting success), zero out-of-core/large-dataset tests
  (`create_sample_Dataset` defaults to `n_frames=2` but already accepts arbitrary `n_frames`), zero
  attrs-preservation tests, zero cross-reader schema-conformance tests.
- **Ecosystem research**: the closest analog is
  [`movement`](https://movement.neuroinformatics.dev/) (neuroinformatics-unit) — an xarray-based
  motion-tracking package with an explicit, documented "native dataset schema" that many
  third-party tracker formats normalize into, input-agnostic I/O, presently NetCDF-based save/load.
  PIVPy's reader registry is structurally similar but lacks an equally explicit schema doc.
  `cf_xarray` shows a cheap, adoptable way to make attrs machine-discoverable (`standard_name`/
  `units`) without needing full CF compliance. Zarr + `xr.open_zarr(chunks=...)` + dask is the
  standard modern pattern for out-of-core xarray data (chunked binary + lightweight JSON metadata,
  cloud/parallel-friendly) — chunking along `t` (one or a few frames per chunk) fits PIV's typical
  access pattern (whole x/y frame at a time, or a temporal reduction). Icechunk (a transactional
  layer on top of Zarr: ACID transactions, snapshots/time-travel) is a credible *future* upgrade,
  not needed for v1.

## Proposed design

### 1. Canonical dataset schema — new `pivpy/schema.py`

Doesn't belong in `io.py` (per-format parsing) or `pivpy.py` (compute accessor) — it's a distinct
concern ("what is a valid pivpy Dataset") importable by both.

- `build_dataset(x, y, u, v, t, chc=None, delta_t=0.0, files=None, **extra_vars) -> xr.Dataset`:
  the one canonical constructor. Extends/replaces `from_arrays()` in place; all 6 readers migrate
  to call it instead of hand-rolling `xr.Dataset(...)`.
- `validate(ds)` / `is_valid(ds)`: checks required dims/vars/coord monotonicity — usable internally
  by readers/tests and exposed as `Dataset.piv.validate()`.
- `set_default_attrs()` moves here from `io.py` (schema policy, not I/O mechanics); `io.py`
  re-exports for backward compat.
- **Resolve `chc`/`mask`**: keep `chc` (1=valid) as the one canonical validity variable.
  `OpenPIVReader` folds its `mask` column into `chc` at read time instead of emitting a second
  variable — release-noted as "OpenPIV-read Datasets no longer contain `mask`; use `chc`."
- **Attrs**: don't chase full CF compliance (PIV isn't geospatial — no need for `axis`/
  `grid_mapping`/bounds). Do add `units` (mostly already present) and a small set of
  domain-specific `standard_name` constants (`"u_velocity"`, `"v_velocity"`, `"validity_flag"`)
  defined once in `schema.py`, cheap and makes `cf_xarray`-style discovery possible later without
  committing to it now. Add one new global attr, `pivpy_schema_version = "1"` — lets future readers
  detect and warn on old stores as the schema evolves; this is the single "known ceiling" worth
  flagging now rather than a config system.

### 2. Zarr-first storage (`pivpy/io.py`)

- `save_piv(dataset, filepath, format="netcdf", chunks=None, mode="w", **kwargs)`: add a `"zarr"`
  branch (`dataset.chunk(chunks or {"t": 1}).to_zarr(path, mode=mode, consolidated=True)`). Default
  `format` stays `"netcdf"` for this pass (no surprise for existing callers) — flipping the default
  to `"zarr"` is a later-phase, deliberately breaking-ish change.
- `ZarrReader(PIVReader)`: `can_read` matches `.zarr` suffix or a directory containing zarr
  metadata; `read(filepath, chunks=None, **kwargs)` does `xr.open_zarr(path, chunks=chunks)`.
  Registered into `PIVReaderRegistry._register_builtin_readers()` — reuses the existing plugin
  pattern, doesn't replace it.
- **`PIVReader.read()` gains `chunks=` as an additive kwarg** (already accepts `**kwargs`, so
  non-breaking); only Zarr/NetCDF readers act on it, per-file text/vec readers ignore it (single
  frame, chunking is moot).
- **Registry/dispatch dedup** (small, independent, do alongside): give each `PIVReader` a `name`
  and add `PIVReaderRegistry.get_by_name(fmt)`; `read_piv(format=...)` calls that instead of its own
  hardcoded `if/elif` table — this is what makes a custom `register_reader()` actually selectable
  by explicit format string, not just auto-detection.
- **`LaVisionVC7Reader` fallback fix** (small, independent, do alongside): raise
  `ImportError("... pip install pivpy[lvpyio]")` when `lvpyio` is missing, matching the pattern
  `PIVLabReader` already uses for `h5py`. Move synthetic-fallback data generation to tests only
  (`pytest.importorskip("lvpyio")` where currently relied upon).
- **Large-directory story** (Phase 2, not this pass): `convert_directory_to_zarr(directory,
  zarr_path, pattern=..., chunks=...)` streams frame-by-frame like `batchf()` already does,
  appending into a Zarr store and discarding each in-memory frame — this is the actual answer to
  "thousands of frames," not a lazy multi-file-text reader (there's no laziness win reading
  inherently-eager per-frame text/vec files one at a time). `read_directory_lazy(directory,
  chunks=None)` then opens the resulting Zarr store lazily. Existing eager `read_directory()` stays
  as-is for small directories/back-compat — not deprecated in this pass.
- **Dependencies**: add `zarr` and `dask` to `pyproject.toml` `[project.dependencies]` (not gated
  behind an optional extra — gating the primary storage path behind an extra reproduces the
  `lvpyio` footgun pattern). `zarr==3.3.0` confirmed resolvable via `uv`.

### 3. Accessor API cleanup (staged — not this pass, planned for Phase 3)

Safe/non-breaking now-or-soon:
- Fix `__mul__`/`__div__` to stop mutating `self._obj` in place; return a new Dataset instead
  (`out = self._obj.copy(); out["u"] = out["u"] * scalar; return out`).
- Route all scalar-producing methods (`vorticity`, `strain`, `divergence`, `acceleration`,
  `kinetic_energy`, `tke`, `reynolds_stress`, `rms`, `vec2scal`) through one shared
  `_assign_scalar(ds, name, data)` helper in `compute_funcs.py` that **warns** (not raises) on
  silent overwrite of an existing same-named variable — root-cause fix in one place rather than
  patching 8+ call sites, keeps the `name="w"` default unchanged (non-breaking) but makes the
  footgun visible.
- Document the existing three-bucket return contract (field-transform → `Dataset`; plotting →
  `(fig, ax)`; `azprofile` → raw tuple) explicitly in one table at the top of `pivpy.py`.

Needs a deprecation cycle (Phase 3, with `DeprecationWarning` for ≥1 release first):
- `crop()`/`vec2scal()` no longer reassigning `self._obj`, standardizing on "always return a new
  Dataset."
- Any future default-format flip (`save_piv` → zarr) or `read_directory` behavior change.

### 4. Test plan

New files, following the existing one-file-per-feature convention:
- `tests/test_schema.py` — parametrized over all 6 readers' sample fixtures, asserts
  `schema.validate(ds)` passes and required dims/vars/attrs match; asserts `OpenPIVReader` output
  never contains `mask` post-fix.
- `tests/test_zarr_roundtrip.py` — save→reload via `assert_identical` (not just file-exists), plus
  a lazy-loading assertion (`isinstance(ds.u.data, dask.array.Array)`).
- Strengthen the existing netCDF round-trip test in `tests/test_io.py` (currently only checks the
  file exists and swallows `RuntimeError`) to actually reload and compare; only skip on
  `ImportError` for the netCDF4 engine, not on write failures.
- `tests/test_out_of_core.py` (Phase 2) — synthetic 500+-frame dataset via
  `create_sample_Dataset(n_frames=500)` (already supports arbitrary `n_frames`), written to Zarr
  with `chunks={"t": 1}`, reopened lazily, asserts dask-backed-ness and that a `.mean("t")`
  reduction doesn't require an upfront full materialization call.
- `tests/test_accessor_attrs.py` (Phase 2) — ~8-10 representative accessor methods, asserts
  `delta_t`/`units`/`files` survive the call.

### 5. Staged rollout

- **Phase 1 (this pass)**: `pivpy/schema.py` foundations are *designed* here but the PoC itself
  (§6) ships the minimal zarr slice without the full schema module — see below.
- **Phase 2**: migrate all readers to `build_dataset`; `convert_directory_to_zarr` +
  `read_directory_lazy`; promote `zarr`/`dask` to hard dependencies; `OpenPIVReader` mask→chc fold;
  `pivpy_schema_version` attr; cross-reader schema conformance test; out-of-core test.
- **Phase 3**: flip `save_piv` default to `"zarr"`; deprecate `self._obj`-mutating accessor
  methods; decide on `azprofile` return-shape consistency.

### 6. Proof-of-concept for this pass (build now)

Minimal and additive — no accessor changes, no `schema.py` yet (not needed to prove the Zarr path
works):

1. `pyproject.toml`: add `zarr` and `dask` to `dependencies`.
2. `pivpy/io.py`: `_save_zarr(dataset, path, chunks=None, mode="w")` helper; wire `format="zarr"`
   into `save_piv` (default `format` stays `"netcdf"`, no behavior change for existing callers).
3. `pivpy/io.py`: `ZarrReader(PIVReader)` — `can_read` matches `.zarr` suffix / zarr-store
   directory; `read(filepath, chunks=None, **kwargs)` → `xr.open_zarr(path, chunks=chunks)`.
   Register it in `PIVReaderRegistry._register_builtin_readers()`.
4. `tests/test_zarr_roundtrip.py`:
   - round-trip: `create_sample_Dataset()` → `save_piv(..., format="zarr")` → `read_piv()` back →
     `xr.testing.assert_identical` (or `assert_allclose` if float tolerance needed).
   - lazy loading: `create_sample_Dataset(n_frames=10)` → save with `chunks={"t": 1}` → reopen with
     `chunks="auto"` → assert `isinstance(ds.u.data, dask.array.Array)`.

Explicitly deferred past this pass: `schema.py`, `convert_directory_to_zarr`,
`read_directory_lazy`, the registry-dispatch dedup fix, and the `LaVisionVC7Reader` fallback fix
(both correctness bugs worth doing, but independent of Zarr and can land as their own small PR).

## Verification

- `uv run pytest tests/test_zarr_roundtrip.py -q` — new PoC tests pass.
- `uv run pytest -q` — full suite still green (no regressions from the additive `ZarrReader` /
  `save_piv` branch or new dependencies).
- Manually: `python -c "from pivpy import io; ds = io.create_sample_Dataset(n_frames=10); io.save_piv(ds, 'x.zarr', format='zarr', chunks={'t':1}); ds2 = io.read_piv('x.zarr', chunks='auto'); print(type(ds2.u.data))"` — confirms a dask-backed array comes back.
- Commit this plan to `docs/architecture/zarr-migration.md` so it's discoverable by contributors,
  not just left in scratch/plan-mode state.
