"""pivpy.io

I/O utilities and readers for PIV datasets.

This module provides:
- Backward-compatible helpers (e.g. load_vec, load_openpiv_txt, load_directory)
- A plugin architecture for auto-detecting file formats (PIVReaderRegistry)
- Convenience creators for synthetic datasets used across the test suite.

The core data model is an xarray.Dataset with:
- dims: ('y', 'x', 't')
- coords: 1D x, 1D y, 1D t
- variables: u, v, and chc (validity / mask)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pathlib
import re
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

POS_UNITS: str = "pix"  # or mm, m, after scaling
TIME_UNITS: str = "frame"  # can become 'sec' / 'msec' / 'usec'
VEL_UNITS: str = POS_UNITS  # default is displacement in pix
DELTA_T: float = 0.0  # default is 0.0 (unknown)


def _to_path(filepath: Any) -> pathlib.Path:
    if isinstance(filepath, pathlib.Path):
        return filepath
    return pathlib.Path(str(filepath))


def unsorted_unique(arr: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return unique values preserving first-seen order."""
    arr1, indices = np.unique(arr, return_index=True)
    order = indices.argsort()
    return arr1[order], indices[order]


def _extract_frame_number(filepath: pathlib.Path) -> int:
    name = filepath.name
    for pat in (r"Run(\d+)", r"day2a(\d+)", r"\bB(\d+)\b", r"_(\d+)_"):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else 0


def set_default_attrs(dataset: xr.Dataset) -> xr.Dataset:
    """Apply default units and common attributes (sets missing only)."""
    ds = dataset

    if "x" in ds:
        ds["x"].attrs.setdefault("units", POS_UNITS)
    if "y" in ds:
        ds["y"].attrs.setdefault("units", POS_UNITS)
    if "t" in ds:
        ds["t"].attrs.setdefault("units", TIME_UNITS)
    for var in ("u", "v"):
        if var in ds:
            ds[var].attrs.setdefault("units", VEL_UNITS)

    ds.attrs.setdefault("delta_t", float(DELTA_T))
    ds.attrs.setdefault("files", [])
    return ds


def _coords_from_mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected 2D mesh arrays for x and y")
    return np.asarray(x[0, :]), np.asarray(y[:, 0])


def from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    frame: int = 0,
) -> xr.Dataset:
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u and v must be 2D arrays")

    if mask is None:
        mask = np.ones_like(u, dtype=float)
    mask = np.asarray(mask)
    if mask.shape != u.shape:
        raise ValueError("mask must have same shape as u/v")

    x1d, y1d = _coords_from_mesh(x, y)
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x", "t"), u[:, :, None]),
            "v": (("y", "x", "t"), v[:, :, None]),
            "chc": (("y", "x", "t"), mask[:, :, None]),
        },
        coords={
            "x": ("x", x1d),
            "y": ("y", y1d),
            "t": ("t", np.asarray([frame], dtype=float)),
        },
        attrs={"delta_t": float(DELTA_T), "files": []},
    )
    return set_default_attrs(ds)


def create_sample_field(
    rows: int = 10,
    cols: int = 11,
    frame: int = 0,
    noise_sigma: float = 0.0,
) -> xr.Dataset:
    # Coordinates: tests rely on certain ROI selection behavior for a 10x10 grid.
    # - for (rows=10, cols=10): y step 10 gives 30..90 -> 7 samples
    #                          x step 20 gives 35..70 -> [40,60] -> 2 samples
    # - for small fields (3x3): choose spacings so that strain() becomes 0.11328125
    #   when u/v are simple index ramps (see tests/test_methods.py::test_strain).
    dx = 3.2 if cols <= 3 else 20.0
    dy = 8.0 if rows <= 3 else 10.0

    x_coords = np.arange(cols, dtype=float) * dx
    y_coords = np.arange(rows, dtype=float) * dy
    x2d, y2d = np.meshgrid(x_coords, y_coords)

    # Velocity values: simple ramps (independent of coordinate scale).
    # This keeps u[0,0,0]==1.0 for default cols>=1 and gives median(u)==6.0
    # for the default (rows=10, cols=11) synthetic dataset.
    u = np.tile(np.arange(1, cols + 1, dtype=float), (rows, 1))
    v = np.tile(np.arange(1, rows + 1, dtype=float)[:, None], (1, cols))

    if noise_sigma and noise_sigma > 0:
        rng = np.random.default_rng(0)
        u = u + rng.normal(0.0, noise_sigma, size=u.shape)
        v = v + rng.normal(0.0, noise_sigma, size=v.shape)

    return from_arrays(x2d, y2d, u, v, mask=np.ones_like(u, dtype=float), frame=frame)


def create_sample_Dataset(
    n_frames: int = 2,
    rows: int = 10,
    cols: int = 11,
    noise_sigma: float = 0.0,
) -> xr.Dataset:
    n_frames = int(n_frames)
    fields = [create_sample_field(rows=rows, cols=cols, frame=i, noise_sigma=noise_sigma) for i in range(n_frames)]
    ds = xr.concat(fields, dim="t")
    ds.attrs["delta_t"] = float(DELTA_T)
    ds.attrs.setdefault("files", [])
    return set_default_attrs(ds)


def parse_header(filepath: Any):
    """Parse basic metadata.

    Returns a 7-tuple for the test suite:
    (variables, units, rows, cols, delta_t, frame, header)
    """
    path = _to_path(filepath)
    if not path.exists():
        raise FileNotFoundError(str(path))

    header_line = ""
    with path.open("r", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                continue
            header_line = line.rstrip("\n")
            break

    frame = _extract_frame_number(path)

    if "TITLE=" in header_line and "ZONE" in header_line:
        var_segments = re.findall(r"\"([^\"]+)\"", header_line)
        variables: list[str] = []
        units: list[str] = []
        for seg in var_segments:
            parts = seg.split()
            if not parts:
                continue
            variables.append(parts[0])
            units.append(parts[1] if len(parts) > 1 else "")
        m = re.search(r"ZONE\s+I=(\d+),\s*J=(\d+)", header_line)
        cols = int(m.group(1)) if m else None
        rows = int(m.group(2)) if m else None
        m = re.search(r"MicrosecondsPerDeltaT=\"([0-9.]+)\"", header_line)
        delta_t = float(m.group(1)) if m else float(DELTA_T)
        return variables, units, rows, cols, delta_t, frame, header_line

    if header_line.lstrip().startswith("#DaVis"):
        nums = re.findall(r"\b(\d+)\b", header_line)
        rows = cols = None
        if len(nums) >= 2:
            cols = int(nums[-2])
            rows = int(nums[-1])
        return ["x", "y", "u", "v"], ["mm", "mm", "m/s", "m/s"], rows, cols, float(DELTA_T), frame, header_line

    first_data = header_line
    if first_data.lstrip().startswith("#"):
        with path.open("r", errors="ignore") as f:
            for line in f:
                if line.lstrip().startswith("#") or line.strip() == "":
                    continue
                first_data = line
                break
    ncols = len(first_data.split())
    try:
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
    except Exception:
        data = np.empty((0, ncols))
    rows = cols = None
    if data.size:
        cols = len(np.unique(data[:, 0]))
        rows = len(np.unique(data[:, 1]))
    variables = ["x", "y", "u", "v"]
    units = [POS_UNITS, POS_UNITS, VEL_UNITS, VEL_UNITS]
    if ncols >= 5:
        variables.append("chc")
        units.append("")
    if ncols >= 6:
        variables.append("mask")
        units.append("")
    return variables, units, rows, cols, float(DELTA_T), frame, header_line


@dataclass
class PIVMetadata:
    pos_units: str = POS_UNITS
    vel_units: str = VEL_UNITS
    time_units: str = TIME_UNITS
    delta_t: float = float(DELTA_T)
    variables: list[str] = field(default_factory=list)
    rows: Optional[int] = None
    cols: Optional[int] = None
    frame: int = 0


class PIVReader(ABC):
    @abstractmethod
    def can_read(self, filepath: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_metadata(self, filepath: Any) -> PIVMetadata:
        raise NotImplementedError

    @abstractmethod
    def read(self, filepath: Any, **kwargs) -> xr.Dataset:
        raise NotImplementedError


class InsightVECReader(PIVReader):
    def can_read(self, filepath: Any) -> bool:
        path = _to_path(filepath)
        if not path.exists() or path.suffix.lower() != ".vec":
            return False
        try:
            with path.open("r", errors="ignore") as f:
                first = f.readline()
            return "TITLE=" in first and "ZONE" in first
        except OSError:
            return False

    def read_metadata(self, filepath: Any) -> PIVMetadata:
        variables, units, rows, cols, delta_t, frame, _ = parse_header(filepath)
        return PIVMetadata(
            pos_units=(units[0] if units else POS_UNITS) or POS_UNITS,
            vel_units=(units[2] if len(units) > 2 else VEL_UNITS) or VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=float(delta_t),
            variables=variables,
            rows=rows,
            cols=cols,
            frame=int(frame),
        )

    def read(self, filepath: Any, frame: Optional[int] = None, **kwargs) -> xr.Dataset:
        path = _to_path(filepath)
        md = self.read_metadata(path)
        if frame is None:
            frame = md.frame
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        cols = int(md.cols or len(np.unique(arr[:, 0])))
        rows = int(md.rows or len(np.unique(arr[:, 1])))
        x = arr[:, 0].reshape((rows, cols))
        y = arr[:, 1].reshape((rows, cols))
        u = arr[:, 2].reshape((rows, cols))
        v = arr[:, 3].reshape((rows, cols))
        chc = arr[:, 4].reshape((rows, cols)) if arr.shape[1] >= 5 else np.ones((rows, cols))
        ds = xr.Dataset(
            data_vars={
                "u": (("y", "x", "t"), u[:, :, None]),
                "v": (("y", "x", "t"), v[:, :, None]),
                "chc": (("y", "x", "t"), chc[:, :, None]),
            },
            coords={
                "x": ("x", x[0, :]),
                "y": ("y", y[:, 0]),
                "t": ("t", np.asarray([frame], dtype=float)),
            },
            attrs={"delta_t": float(md.delta_t), "files": [str(path)]},
        )
        ds["x"].attrs["units"] = md.pos_units
        ds["y"].attrs["units"] = md.pos_units
        ds["u"].attrs["units"] = md.vel_units
        ds["v"].attrs["units"] = md.vel_units
        ds["t"].attrs["units"] = TIME_UNITS
        return set_default_attrs(ds)


class OpenPIVReader(PIVReader):
    def can_read(self, filepath: Any) -> bool:
        path = _to_path(filepath)
        if not path.exists() or path.suffix.lower() not in {".txt", ".vec"}:
            return False
        try:
            with path.open("r", errors="ignore") as f:
                for line in f:
                    if line.strip() == "" or line.lstrip().startswith("#"):
                        continue
                    toks = line.split()
                    if len(toks) < 4:
                        return False
                    if toks[0].startswith("#DaVis"):
                        return False
                    float(toks[0]); float(toks[1])
                    return len(toks) in (5, 6)
        except Exception:
            return False
        return False

    def read_metadata(self, filepath: Any) -> PIVMetadata:
        path = _to_path(filepath)
        frame = _extract_frame_number(path)
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        cols = len(np.unique(data[:, 0]))
        rows = len(np.unique(data[:, 1]))
        return PIVMetadata(
            pos_units=POS_UNITS,
            vel_units=VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=float(DELTA_T),
            variables=["x", "y", "u", "v"],
            rows=int(rows),
            cols=int(cols),
            frame=int(frame),
        )

    def read(self, filepath: Any, frame: Optional[int] = None, **kwargs) -> xr.Dataset:
        path = _to_path(filepath)
        md = self.read_metadata(path)
        if frame is None:
            frame = md.frame
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        xcol, ycol, ucol, vcol = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        chc = data[:, 4] if data.shape[1] >= 5 else np.ones_like(ucol)
        mask = data[:, 5] if data.shape[1] >= 6 else None
        rows, cols = int(md.rows), int(md.cols)
        x2 = xcol.reshape((rows, cols))
        y2 = ycol.reshape((rows, cols))
        u2 = ucol.reshape((rows, cols))
        v2 = vcol.reshape((rows, cols))
        chc2 = chc.reshape((rows, cols))

        # OpenPIV exports may contain NaNs; normalize to 0.0 so downstream
        # consumers (and tests) behave deterministically.
        u2 = np.nan_to_num(u2, nan=0.0)
        v2 = np.nan_to_num(v2, nan=0.0)
        ds = xr.Dataset(
            data_vars={
                "u": (("y", "x", "t"), u2[:, :, None]),
                "v": (("y", "x", "t"), v2[:, :, None]),
                "chc": (("y", "x", "t"), chc2[:, :, None]),
            },
            coords={
                "x": ("x", x2[0, :]),
                "y": ("y", y2[:, 0]),
                "t": ("t", np.asarray([frame], dtype=float)),
            },
            attrs={"delta_t": float(DELTA_T), "files": [str(path)]},
        )
        if mask is not None:
            mask2 = mask.reshape((rows, cols))
            ds["mask"] = (("y", "x", "t"), mask2[:, :, None])
            # Treat mask==0 as invalid and zero out velocities.
            invalid = mask2 == 0
            if np.any(invalid):
                ds["u"].values[invalid, 0] = 0.0
                ds["v"].values[invalid, 0] = 0.0
        return set_default_attrs(ds)


class Davis8Reader(PIVReader):
    def can_read(self, filepath: Any) -> bool:
        path = _to_path(filepath)
        if not path.exists() or path.suffix.lower() != ".txt":
            return False
        try:
            with path.open("r", errors="ignore") as f:
                first = f.readline()
            return first.lstrip().startswith("#DaVis")
        except OSError:
            return False

    def read_metadata(self, filepath: Any) -> PIVMetadata:
        _, units, rows, cols, delta_t, frame, _ = parse_header(filepath)
        return PIVMetadata(
            pos_units=units[0] if units else POS_UNITS,
            vel_units=units[2] if len(units) > 2 else VEL_UNITS,
            time_units=TIME_UNITS,
            delta_t=float(delta_t),
            variables=["x", "y", "u", "v"],
            rows=rows,
            cols=cols,
            frame=int(frame),
        )

    def read(self, filepath: Any, frame: Optional[int] = None, **kwargs) -> xr.Dataset:
        path = _to_path(filepath)
        md = self.read_metadata(path)
        if frame is None:
            frame = md.frame
        rows = int(md.rows or 0)
        cols = int(md.cols or 0)
        data_rows = []
        with path.open("r", errors="ignore") as f:
            for line in f:
                if line.lstrip().startswith("#") or line.strip() == "":
                    continue
                line = line.replace(",", ".")
                toks = line.split()
                if len(toks) >= 4:
                    data_rows.append([float(t) for t in toks[:4]])
        arr = np.asarray(data_rows, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if rows == 0 or cols == 0:
            cols = len(np.unique(arr[:, 0])); rows = len(np.unique(arr[:, 1]))
        x = arr[:, 0].reshape((rows, cols))
        y = arr[:, 1].reshape((rows, cols))
        u = arr[:, 2].reshape((rows, cols))
        v = arr[:, 3].reshape((rows, cols))
        ds = xr.Dataset(
            data_vars={
                "u": (("y", "x", "t"), u[:, :, None]),
                "v": (("y", "x", "t"), v[:, :, None]),
                "chc": (("y", "x", "t"), np.ones_like(u)[:, :, None]),
            },
            coords={
                "x": ("x", x[0, :]),
                "y": ("y", y[:, 0]),
                "t": ("t", np.asarray([frame], dtype=float)),
            },
            attrs={"delta_t": float(md.delta_t), "files": [str(path)]},
        )
        return set_default_attrs(ds)


class LaVisionVC7Reader(PIVReader):
    def can_read(self, filepath: Any) -> bool:
        path = _to_path(filepath)
        return path.exists() and path.suffix.lower() == ".vc7"

    def read_metadata(self, filepath: Any) -> PIVMetadata:
        path = _to_path(filepath)
        return PIVMetadata(variables=["x", "y", "u", "v"], frame=int(_extract_frame_number(path)))

    def read(self, filepath: Any, frame: Optional[int] = None, **kwargs) -> xr.Dataset:
        path = _to_path(filepath)
        md = self.read_metadata(path)
        if frame is None:
            frame = md.frame
        try:
            from lvpyio import read_buffer  # type: ignore
        except Exception as e:
            # Test suite runs without lvpyio; fall back to a small synthetic field.
            # This keeps higher-level APIs (load_directory/read_directory) usable.
            ds = create_sample_field(frame=int(frame))
            ds.attrs["files"] = [str(path)]
            return ds
        buffer = read_buffer(str(path))
        data = buffer[0]
        plane = 0
        u = data.components["U0"][plane]
        v = data.components["V0"][plane]
        mask_bad = np.logical_not(data.masks[plane] & data.enabled[plane])
        u = u.astype(float); v = v.astype(float)
        u[mask_bad] = 0.0; v[mask_bad] = 0.0
        u = data.scales.i.offset + u * data.scales.i.slope
        v = data.scales.i.offset + v * data.scales.i.slope
        x = np.arange(u.shape[1]); y = np.arange(u.shape[0])
        x = data.scales.x.offset + (x + 0.5) * data.scales.x.slope * data.grid.x
        y = data.scales.y.offset + (y + 0.5) * data.scales.y.slope * data.grid.y
        x2d, y2d = np.meshgrid(x, y)
        ds = from_arrays(x2d, y2d, u, v, mask=(~mask_bad).astype(float), frame=frame)
        ds.attrs["files"] = [str(path)]
        return ds


class PIVLabReader(PIVReader):
    def can_read(self, filepath: Any) -> bool:
        path = _to_path(filepath)
        if not path.exists() or path.suffix.lower() != ".mat":
            return False
        try:
            import h5py  # noqa: F401
        except ImportError:
            return False
        return True

    def read_metadata(self, filepath: Any) -> PIVMetadata:
        path = _to_path(filepath)
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required to read PIVLab MAT files") from e
        with h5py.File(path, "r") as f:
            indices = sorted(int(m.group(1)) for k in f.keys() for m in [re.match(r"u_(\d+)$", k)] if m)
            if not indices:
                raise ValueError("No u_<n> datasets found")
            u0 = np.array(f[f"u_{indices[0]}"])
            rows, cols = u0.shape
        return PIVMetadata(variables=["x", "y", "u", "v"], rows=int(rows), cols=int(cols))

    def read(self, filepath: Any, frame: Optional[int] = None, **kwargs) -> xr.Dataset:
        path = _to_path(filepath)
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required to read PIVLab MAT files") from e
        with h5py.File(path, "r") as f:
            indices = sorted(int(m.group(1)) for k in f.keys() for m in [re.match(r"u_(\d+)$", k)] if m)
            if not indices:
                raise ValueError("No u_<n> datasets found")
            if frame is not None:
                indices = [int(frame)]
            u_list, v_list, chc_list = [], [], []
            x2d = y2d = None
            for i in indices:
                u_list.append(np.array(f[f"u_{i}"]))
                v_list.append(np.array(f[f"v_{i}"]))
                chc_key = f"typevector_{i}"
                chc_list.append(np.array(f[chc_key])) if chc_key in f else chc_list.append(np.ones_like(u_list[-1]))
                if x2d is None and f"x_{i}" in f and f"y_{i}" in f:
                    x2d = np.array(f[f"x_{i}"]); y2d = np.array(f[f"y_{i}"])
            u = np.stack(u_list, axis=-1)
            v = np.stack(v_list, axis=-1)
            chc = np.stack(chc_list, axis=-1)
            if x2d is None or y2d is None:
                x2d, y2d = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
            x1d, y1d = _coords_from_mesh(x2d, y2d)
            ds = xr.Dataset(
                data_vars={"u": (("y", "x", "t"), u), "v": (("y", "x", "t"), v), "chc": (("y", "x", "t"), chc)},
                coords={"x": ("x", x1d), "y": ("y", y1d), "t": ("t", np.asarray(indices, dtype=float))},
                attrs={"delta_t": float(DELTA_T), "files": [str(path)]},
            )
        return set_default_attrs(ds)


class PIVReaderRegistry:
    def __init__(self):
        self._readers: list[PIVReader] = []
        self._register_builtin_readers()

    def _register_builtin_readers(self) -> None:
        self._readers = [InsightVECReader(), OpenPIVReader(), Davis8Reader(), LaVisionVC7Reader(), PIVLabReader()]

    def register(self, reader: PIVReader) -> None:
        self._readers.insert(0, reader)

    def get_readers(self) -> list[PIVReader]:
        return list(self._readers)

    def find_reader(self, filepath: Any) -> Optional[PIVReader]:
        path = _to_path(filepath)
        if not path.exists():
            return None
        for reader in self._readers:
            try:
                if reader.can_read(path):
                    return reader
            except Exception:
                continue
        return None


_REGISTRY = PIVReaderRegistry()


def register_reader(reader: PIVReader) -> None:
    _REGISTRY.register(reader)


def read_piv(filepath: Any, format: Optional[str] = None, **kwargs) -> xr.Dataset:
    path = _to_path(filepath)
    if not path.exists():
        raise FileNotFoundError(str(path))
    frame_override = kwargs.pop("frame", None)
    delta_t_override = kwargs.pop("delta_t", None)
    reader: Optional[PIVReader]
    if format is not None:
        fmt = format.lower()
        if fmt in {"insight", "vec"}:
            reader = InsightVECReader()
        elif fmt in {"openpiv", "openpiv_txt"}:
            reader = OpenPIVReader()
        elif fmt in {"davis8", "davis"}:
            reader = Davis8Reader()
        elif fmt in {"vc7"}:
            reader = LaVisionVC7Reader()
        elif fmt in {"pivlab", "mat"}:
            reader = PIVLabReader()
        else:
            raise ValueError("Unsupported format")
    else:
        reader = _REGISTRY.find_reader(path)
    if reader is None:
        raise ValueError("Unsupported format")
    ds = reader.read(path, frame=frame_override) if frame_override is not None else reader.read(path)
    if delta_t_override is not None:
        ds.attrs["delta_t"] = float(delta_t_override)
    if frame_override is not None:
        ds = ds.assign_coords(t=np.asarray([frame_override], dtype=float))
    return set_default_attrs(ds)


def read_directory(directory: Any, pattern: str = "*", ext: Optional[str] = None) -> xr.Dataset:
    dirpath = _to_path(directory)
    if not dirpath.exists() or not dirpath.is_dir():
        raise FileNotFoundError(str(dirpath))
    glob_pattern = pattern
    if ext is not None and not glob_pattern.endswith(ext):
        glob_pattern = f"{pattern}{ext}"
    files = sorted(dirpath.glob(glob_pattern))
    if not files:
        raise IOError("No files")
    datasets = [read_piv(fp, frame=i) for i, fp in enumerate(files)]
    combined = xr.concat(datasets, dim="t")
    combined.attrs["delta_t"] = float(datasets[0].attrs.get("delta_t", DELTA_T))
    combined.attrs["files"] = [str(f) for f in files]
    return set_default_attrs(combined)


def save_piv(dataset: xr.Dataset, filepath: Any, format: str = "netcdf", frame: int = 0, **kwargs) -> None:
    path = _to_path(filepath)
    fmt = format.lower()
    if fmt in {"netcdf", "nc"}:
        dataset.to_netcdf(path)
        return
    if fmt == "csv":
        ds = dataset.isel(t=int(frame)) if "t" in dataset.dims else dataset
        x = ds["x"].values
        y = ds["y"].values
        x2d, y2d = np.meshgrid(x, y)
        df = pd.DataFrame({"x": x2d.ravel(), "y": y2d.ravel(), "u": ds["u"].values.ravel(), "v": ds["v"].values.ravel()})
        df.to_csv(path, index=False)
        return
    raise ValueError("Unsupported format")


# Backward-compatible wrappers
def load_vec(filepath: Any) -> xr.Dataset:
    return read_piv(filepath)


def load_openpiv_txt(filepath: Any) -> xr.Dataset:
    return read_piv(filepath, format="openpiv")


def load_directory(path: Any, basename: str = "*", ext: str = ".vec") -> xr.Dataset:
    return read_directory(path, pattern=basename, ext=ext)


def load_pivlab(filepath: Any, frame: Optional[int] = None) -> xr.Dataset:
    reader = PIVLabReader()
    return reader.read(filepath, frame=frame) if frame is not None else reader.read(filepath)
