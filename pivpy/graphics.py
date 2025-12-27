"""pivpy.graphics

Plotting helpers used by the test suite and the xarray accessor in
``pivpy/pivpy.py``.

Important behavioral expectations (tests rely on these):

- ``quiver()`` and ``streamplot()`` return ``(fig, ax)``
- ``showf()`` exists
- ``showscal()`` accepts ``flow_property=`` as an alias and can compute a scalar
    via the ``.piv.vec2scal()`` accessor when needed
"""

from __future__ import annotations

import warnings
import pathlib
import glob
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from matplotlib.quiver import Quiver
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def quiver(
    data: xr.Dataset,
    quiverKey: str | float | int = "Q",
    scalingFactor: float = 1.0,
    widthFactor: float = 0.002,
    ax: Axes | None = None,
    arrowColor: str = "k",
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Creates a quiver plot from the dataset

    Parameters
    ----------
    data : xarray.Dataset
        dataset with u, v, x, y
    quiverKey : str
        key for the quiver plot, by default "Q"
    scalingFactor : float
        scaling factor for the arrows, by default 1.0
    widthFactor : float
        width factor for the arrows, by default 0.002
    ax : matplotlib.axes.Axes | None
        matplotlib axes, by default None
    arrowColor : str
        color of the arrows, by default "k"

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes used for plotting.

    Examples
    --------
    >>> from pivpy import io, graphics
    >>> d = io.loadvec("test.vec")
    >>> graphics.quiver(d)
    """
    from pivpy.graphics_utils import dataset_to_array

    # ------------------------------------------------------------------
    # Backwards-compatible API shims (for older notebooks/examples)
    # ------------------------------------------------------------------
    # Old signature patterns commonly used:
    # - quiver(ds, arrScale=10, nthArr=2, aspectratio=0.5, add_guide=False)
    # - quiver(ds, 5)  # positional arrScale
    # - quiver(ds, colorbar=True, cmap='Reds', width=0.0075, streamlines=True)
    if isinstance(quiverKey, (int, float)):
        # Treat the 2nd positional argument as arrScale.
        if scalingFactor == 1.0 and "arrScale" not in kwargs:
            scalingFactor = float(quiverKey)
        quiverKey = "Q"

    # Legacy aliases
    if "arrScale" in kwargs and scalingFactor == 1.0:
        scalingFactor = float(kwargs.pop("arrScale"))
    if "width" in kwargs and widthFactor == 0.002:
        widthFactor = float(kwargs.pop("width"))

    nthArr = kwargs.pop("nthArr", None)
    aspectratio = kwargs.pop("aspectratio", None)
    add_guide = kwargs.pop("add_guide", True)
    streamlines = bool(kwargs.pop("streamlines", False))
    colorbar = bool(kwargs.pop("colorbar", False))
    colorbar_orient = kwargs.pop("colorbar_orient", "vertical")
    cmap = kwargs.pop("cmap", None)
    units = kwargs.pop("units", None)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x, y, u, v = dataset_to_array(data)

    # Subsample vectors for display
    if nthArr is not None:
        try:
            step = int(nthArr)
        except Exception:
            step = 1
        if step and step > 1:
            x = x[::step, ::step]
            y = y[::step, ::step]
            u = u[::step, ::step]
            v = v[::step, ::step]

    # Prefer xarray attrs if present; otherwise use empty strings.
    xUnits = str(getattr(data.get("x", None), "attrs", {}).get("units", ""))
    yUnits = str(getattr(data.get("y", None), "attrs", {}).get("units", ""))
    if units and isinstance(units, (list, tuple)) and len(units) >= 2:
        xUnits = str(units[0])
        yUnits = str(units[1])

    quiver_kwargs: dict = {
        "scale": scalingFactor,
        "width": widthFactor,
    }

    # If colorbar requested, color by vector magnitude
    if colorbar:
        mag = np.sqrt(u**2 + v**2)
        Q = ax.quiver(x, y, u, v, mag, cmap=cmap, **quiver_kwargs, **kwargs)
        plt.colorbar(Q, ax=ax, orientation=colorbar_orient)
    else:
        Q = ax.quiver(x, y, u, v, color=arrowColor, **quiver_kwargs, **kwargs)

    # Aspect handling
    if aspectratio is None:
        ax.set_aspect("equal")
    elif isinstance(aspectratio, str) and aspectratio.lower() == "auto":
        ax.set_aspect("auto")
    else:
        try:
            ax.set_aspect(float(aspectratio))
        except Exception:
            ax.set_aspect("equal")

    if add_guide:
        ax.quiverkey(
            Q,
            0.9,
            0.9,
            1,
            str(quiverKey),
            labelpos="E",
            coordinates="figure",
        )
    ax.set_xlabel(f"x [{xUnits}]")
    ax.set_ylabel(f"y [{yUnits}]")

    if streamlines:
        try:
            # Use any remaining streamplot-like kwargs if provided.
            sp_density = float(kwargs.pop("density", 1.0)) if "density" in kwargs else 1.0
            sp_linewidth = float(kwargs.pop("linewidth", 1.0)) if "linewidth" in kwargs else 1.0
            sp_arrowsize = float(kwargs.pop("arrowsize", 1.0)) if "arrowsize" in kwargs else 1.0
            streamplot(data, density=sp_density, linewidth=sp_linewidth, arrowsize=sp_arrowsize, ax=ax)
        except Exception:
            # Keep quiver usable even if streamplot fails.
            pass

    return fig, ax

def statvec_disp(stat_u: dict, stat_v: dict, ax=None, title: str | None = None):
    """Display vector statistics returned by ``statf`` (PIVMAT-style helper)."""

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    labels = ["mean", "std", "rms", "min", "max"]
    uvals = [float(stat_u.get(k, np.nan)) for k in labels]
    vvals = [float(stat_v.get(k, np.nan)) for k in labels]

    x = np.arange(len(labels))
    ax.plot(x, uvals, "o-", label="u")
    ax.plot(x, vvals, "o-", label="v")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("value")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def vsf_disp(vsf: xr.Dataset, ax=None, title: str | None = None):
    """Display vector structure function results (PIVMAT-style helper)."""

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    if "r" in vsf.coords:
        r = np.asarray(vsf["r"].values)
    else:
        raise ValueError("vsf_disp expects a Dataset with coordinate 'r'")

    for name in ["SLL", "SNN", "STT", "S2", "S3"]:
        if name in vsf.data_vars:
            ax.loglog(r, np.asarray(vsf[name].values), label=name)

    ax.set_xlabel("r")
    ax.set_ylabel("VSF")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def vectorplot(
    data: xr.Dataset,
    arrowColor: str = "k",
    arrowScale: float = 1.0,
    arrowWidth: float = 0.002,
) -> tuple[Figure, Axes]:
    """
    vectorplot plots the vector field

    Parameters
    ----------
    data : xarray.Dataset
        dataset with u, v, x, y
    arrowColor : str
        color of the arrows, by default "k"
    arrowScale : float
        scaling factor for the arrows, by default 1.0
    arrowWidth : float
        width factor for the arrows, by default 0.002

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes used for plotting.

    Examples
    --------
    >>> from pivpy import io, graphics
    >>> d = io.loadvec("test.vec")
    >>> graphics.vectorplot(d)
    """
    return quiver(
        data,
        arrowColor=arrowColor,
        scalingFactor=arrowScale,
        widthFactor=arrowWidth,
    )


def showscal(
    data: xr.Dataset,
    property: str = "w",
    **kwargs,
) -> None:
    """
    showscal plots the scalar field

    Parameters
    ----------
    data : xarray.Dataset
        dataset with u, v, x, y
    property : str
        property to plot, by default "w"
    **kwargs
        additional keyword arguments for pcolormesh

    Examples
    --------
    >>> from pivpy import io, graphics
    >>> d = io.loadvec("test.vec")
    >>> graphics.showscal(d, "chc")
    """
    from pivpy.graphics_utils import dataset_to_array

    # Backwards-compat: tests call showscal(..., flow_property="curl")
    flow_property = kwargs.pop("flow_property", None)

    ds = data
    if flow_property is not None and property not in ds:
        try:
            ds = ds.piv.vec2scal(flow_property)
        except Exception:
            # Fall back to plotting what we have.
            ds = data

    x, y, _, _ = dataset_to_array(ds)
    if property in ds:
        plt.pcolormesh(x, y, ds[property].isel(t=0) if "t" in ds[property].dims else ds[property], **kwargs)
        plt.colorbar(label=property)
        plt.axis("equal")
    else:
        warnings.warn(f"Property {property} not found in dataset")


def streamplot(
    data: xr.Dataset,
    density: float = 1.0,
    linewidth: float = 1.0,
    arrowsize: float = 1.0,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple["Figure", "Axes"]:
    """
    streamplot plots the streamlines of the vector field

    Parameters
    ----------
    data : xarray.Dataset
        dataset with u, v, x, y
    density : float
        density of the streamlines, by default 1.0
    linewidth : float
        linewidth of the streamlines, by default 1.0
    arrowsize : float
        size of the arrows, by default 1.0
    ax : matplotlib.axes.Axes
        Matplotlib axes (or None), by default None
    **kwargs
        additional keyword arguments for streamplot

    Examples
    --------
    >>> from pivpy import io, graphics
    >>> d = io.loadvec("test.vec")
    >>> graphics.streamplot(d)
    """
    from pivpy.graphics_utils import dataset_to_array

    x, y, u, v = dataset_to_array(data)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    xUnits = str(getattr(data.get("x", None), "attrs", {}).get("units", ""))
    yUnits = str(getattr(data.get("y", None), "attrs", {}).get("units", ""))

    # Matplotlib requires strictly increasing x and y.
    x1 = x[0, :]
    y1 = y[:, 0]
    u2 = u
    v2 = v

    if y1.size >= 2 and y1[0] > y1[-1]:
        y1 = y1[::-1]
        u2 = u2[::-1, :]
        v2 = v2[::-1, :]

    if x1.size >= 2 and x1[0] > x1[-1]:
        x1 = x1[::-1]
        u2 = u2[:, ::-1]
        v2 = v2[:, ::-1]

    ax.streamplot(
        x1,
        y1,
        u2,
        v2,
        density=density,
        linewidth=linewidth,
        arrowsize=arrowsize,
        **kwargs,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{xUnits}]")
    ax.set_ylabel(f"y [{yUnits}]")

    return fig, ax


def showf(data: xr.Dataset, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """Display a vector or scalar field (PIVMAT-inspired dispatcher).

    This is a lightweight Python analogue of PIVMAT's ``showf``:

    - If the dataset contains ``u`` and ``v``, it displays a vector field.
    - Otherwise, it displays a scalar field (default variable ``w``).

    Parameters
    ----------
    data:
        Dataset to display.
    background:
        Optional scalar background shown behind vectors.

        Use ``None``/``''``/``'off'`` for no background. If ``background`` matches
        an existing variable name, that variable is displayed. Otherwise, it is
        interpreted as a vector-derived flow property and computed using
        ``data.piv.vec2scal(background, name='w')``.
    scalar:
        Scalar variable to display if the dataset is scalar-only.
    ax:
        Optional axes.
    **kwargs:
        Forwarded to :func:`quiver` (vector mode) or matplotlib pcolormesh
        (scalar mode). Recognized scalar kwargs: ``cmap``, ``clim``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes used for plotting.
    """

    background = kwargs.pop("background", None)
    scalar = kwargs.pop("scalar", "w")
    ax = kwargs.pop("ax", None)

    is_vector = "u" in data and "v" in data
    if is_vector:
        if background is None or str(background).strip() == "" or str(background).lower() == "off":
            return quiver(data, ax=ax, **kwargs)
        # Vector field with scalar background
        fig, ax = quiver(data, ax=ax, **kwargs)
        try:
            _plot_scalar_background(ax, data, background=background, clim=kwargs.get("clim", None), cmap=kwargs.get("cmap", None))
        except Exception:
            # Keep showf usable even if background fails.
            pass
        return fig, ax

    # Scalar field
    return _showscal_axes(data, property=str(scalar), ax=ax, **kwargs)


def _showscal_axes(
    data: xr.Dataset,
    property: str = "w",
    *,
    ax: Axes | None = None,
    clim: tuple[float, float] | str | None = None,
    cmap: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    from pivpy.graphics_utils import dataset_to_array

    ds = data
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x, y, _, _ = dataset_to_array(ds if "t" not in ds.dims else ds.isel(t=0))
    if property not in ds:
        raise KeyError(f"Property {property} not found in dataset")
    da = ds[property]
    z = np.asarray(da.isel(t=0).values if "t" in da.dims else da.values)

    plot_kwargs: dict = {"shading": "auto"}
    if cmap is not None:
        plot_kwargs["cmap"] = cmap

    m = ax.pcolormesh(x, y, z, **plot_kwargs, **kwargs)
    if clim is not None and isinstance(clim, (tuple, list)) and len(clim) == 2:
        m.set_clim(float(clim[0]), float(clim[1]))
    plt.colorbar(m, ax=ax, label=property)
    ax.set_aspect("equal")
    return fig, ax


def _plot_scalar_background(
    ax: Axes,
    data: xr.Dataset,
    *,
    background: str,
    clim: tuple[float, float] | str | None,
    cmap: str | None,
) -> "plt.collections.QuadMesh":
    from pivpy.graphics_utils import dataset_to_array

    ds = data
    bg = str(background)
    if bg in ds:
        bg_var = bg
        ds_bg = ds
    else:
        import pivpy.pivpy  # registers the .piv accessor

        ds_bg = ds.copy()
        ds_bg = ds_bg.piv.vec2scal(bg, name="__bg")
        bg_var = "__bg"

    x, y, _, _ = dataset_to_array(ds_bg if "t" not in ds_bg.dims else ds_bg.isel(t=0))
    da = ds_bg[bg_var]
    z = np.asarray(da.isel(t=0).values if "t" in da.dims else da.values)

    plot_kwargs: dict = {"shading": "auto"}
    if cmap is not None:
        plot_kwargs["cmap"] = cmap

    m = ax.pcolormesh(x, y, z, **plot_kwargs)
    if clim is not None and isinstance(clim, (tuple, list)) and len(clim) == 2:
        m.set_clim(float(clim[0]), float(clim[1]))
    return m


class _FrameCollector:
    def __init__(self):
        self.frames: list[np.ndarray] = []

    def grab(self, fig: Figure) -> None:
        fig.canvas.draw()
        self.frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())


def _format_title(
    title: str | None,
    *,
    i: int,
    t: float | None,
    file: str | None,
) -> str | None:
    if title is None:
        return None
    try:
        return str(title).format(i=i, t=t, file=file)
    except Exception:
        return str(title)


class _VectorMovieRenderer:
    def __init__(
        self,
        *,
        first: xr.Dataset,
        ax: Axes | None,
        background: str | None,
        title: str | None,
        clim: tuple[float, float] | str | None,
        cmap: str | None,
        quiver_kwargs: dict,
    ):
        from pivpy.graphics_utils import dataset_to_array

        self.background = background
        self.title = title
        self.clim = clim
        self.cmap = cmap

        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = ax.figure

        # Parse a subset of existing quiver() compat kwargs.
        self.nthArr = quiver_kwargs.pop("nthArr", None)
        self.aspectratio = quiver_kwargs.pop("aspectratio", None)
        self.add_guide = quiver_kwargs.pop("add_guide", True)
        self.streamlines = bool(quiver_kwargs.pop("streamlines", False))
        self.colorbar = bool(quiver_kwargs.pop("colorbar", False))
        self.colorbar_orient = quiver_kwargs.pop("colorbar_orient", "vertical")
        self.arrowColor = quiver_kwargs.pop("arrowColor", "k")
        self.scalingFactor = float(quiver_kwargs.pop("scalingFactor", 1.0))
        self.widthFactor = float(quiver_kwargs.pop("widthFactor", 0.002))
        self.quiverKey = quiver_kwargs.pop("quiverKey", "Q")

        x, y, u, v = dataset_to_array(first)
        if self.nthArr is not None:
            step = max(1, int(self.nthArr))
            x = x[::step, ::step]
            y = y[::step, ::step]
            u = u[::step, ::step]
            v = v[::step, ::step]

        self._bg_mesh = None
        if background is not None and str(background).strip() != "" and str(background).lower() != "off":
            try:
                self._bg_mesh = _plot_scalar_background(self.ax, first, background=str(background), clim=clim, cmap=cmap)
            except Exception:
                self._bg_mesh = None

        quiver_plot_kwargs: dict = {"scale": self.scalingFactor, "width": self.widthFactor}
        if self.colorbar:
            mag = np.sqrt(u**2 + v**2)
            self._Q = self.ax.quiver(x, y, u, v, mag, cmap=cmap, **quiver_plot_kwargs, **quiver_kwargs)
            plt.colorbar(self._Q, ax=self.ax, orientation=self.colorbar_orient)
        else:
            self._Q = self.ax.quiver(x, y, u, v, color=self.arrowColor, **quiver_plot_kwargs, **quiver_kwargs)

        if self.aspectratio is None:
            self.ax.set_aspect("equal")
        elif isinstance(self.aspectratio, str) and self.aspectratio.lower() == "auto":
            self.ax.set_aspect("auto")
        else:
            try:
                self.ax.set_aspect(float(self.aspectratio))
            except Exception:
                self.ax.set_aspect("equal")

        if self.add_guide:
            try:
                self.ax.quiverkey(self._Q, 0.9, 0.9, 1, str(self.quiverKey), labelpos="E", coordinates="figure")
            except Exception:
                pass

        if self.streamlines:
            try:
                streamplot(first, ax=self.ax)
            except Exception:
                pass

    @property
    def figax(self) -> tuple[Figure, Axes]:
        return self.fig, self.ax

    def update(self, ds: xr.Dataset, *, i: int, file: str | None = None) -> None:
        from pivpy.graphics_utils import dataset_to_array

        x, y, u, v = dataset_to_array(ds)
        if self.nthArr is not None:
            step = max(1, int(self.nthArr))
            x = x[::step, ::step]
            y = y[::step, ::step]
            u = u[::step, ::step]
            v = v[::step, ::step]

        if self._bg_mesh is not None and self.background is not None:
            try:
                bg = str(self.background)
                if bg in ds:
                    da = ds[bg]
                    z = np.asarray(da.isel(t=0).values if "t" in da.dims else da.values)
                else:
                    import pivpy.pivpy  # registers the .piv accessor

                    tmp = ds.copy()
                    tmp = tmp.piv.vec2scal(bg, name="__bg")
                    da = tmp["__bg"]
                    z = np.asarray(da.isel(t=0).values if "t" in da.dims else da.values)

                if self.nthArr is not None:
                    step = max(1, int(self.nthArr))
                    z = z[::step, ::step]

                self._bg_mesh.set_array(z.ravel())
                if isinstance(self.clim, str) and self.clim.lower() == "each":
                    finite = z[np.isfinite(z)]
                    if finite.size:
                        self._bg_mesh.set_clim(float(np.min(finite)), float(np.max(finite)))
            except Exception:
                pass

        self._Q.set_UVC(u, v)
        if self.colorbar:
            mag = np.sqrt(u**2 + v**2)
            self._Q.set_array(mag.ravel())
            if isinstance(self.clim, str) and self.clim.lower() == "each":
                finite = mag[np.isfinite(mag)]
                if finite.size:
                    self._Q.set_clim(float(np.min(finite)), float(np.max(finite)))

        t_val: float | None
        if "t" in ds.coords and ds["t"].size:
            try:
                t_val = float(ds["t"].values[0])
            except Exception:
                t_val = None
        else:
            t_val = None

        ttl = _format_title(self.title, i=i, t=t_val, file=file)
        if ttl is not None:
            self.ax.set_title(ttl)


class _ScalarMovieRenderer:
    def __init__(
        self,
        *,
        first: xr.Dataset,
        ax: Axes | None,
        variable: str,
        title: str | None,
        clim: tuple[float, float] | str | None,
        cmap: str | None,
        pcolormesh_kwargs: dict,
    ):
        self.variable = variable
        self.title = title
        self.clim = clim
        self.cmap = cmap

        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = ax.figure

        ds0 = first.isel(t=0) if "t" in first.dims else first
        if "x" not in ds0.coords or "y" not in ds0.coords:
            raise KeyError("Scalar movie rendering requires 1D coords 'x' and 'y'")

        x1 = np.asarray(ds0["x"].values)
        y1 = np.asarray(ds0["y"].values)
        x2d, y2d = np.meshgrid(x1, y1)

        if variable not in ds0:
            raise KeyError(f"Variable {variable} not found in dataset")
        da = ds0[variable]
        z = np.asarray(da.values)

        kwargs = dict(pcolormesh_kwargs)
        kwargs.setdefault("shading", "auto")
        if cmap is not None:
            kwargs.setdefault("cmap", cmap)

        self._mesh = self.ax.pcolormesh(x2d, y2d, z, **kwargs)
        self._cbar = plt.colorbar(self._mesh, ax=self.ax, label=variable)

        if isinstance(clim, (tuple, list)) and len(clim) == 2:
            self._mesh.set_clim(float(clim[0]), float(clim[1]))
        self.ax.set_aspect("equal")

    @property
    def figax(self) -> tuple[Figure, Axes]:
        return self.fig, self.ax

    def update(self, ds: xr.Dataset, *, i: int, file: str | None = None) -> None:
        if self.variable not in ds:
            raise KeyError(f"Variable {self.variable} not found in dataset")
        da = ds[self.variable]
        z = np.asarray(da.isel(t=0).values if "t" in da.dims else da.values)
        self._mesh.set_array(z.ravel())
        if isinstance(self.clim, str) and self.clim.lower() == "each":
            finite = z[np.isfinite(z)]
            if finite.size:
                self._mesh.set_clim(float(np.min(finite)), float(np.max(finite)))

        t_val: float | None
        if "t" in ds.coords and ds["t"].size:
            try:
                t_val = float(ds["t"].values[0])
            except Exception:
                t_val = None
        else:
            t_val = None

        ttl = _format_title(self.title, i=i, t=t_val, file=file)
        if ttl is not None:
            self.ax.set_title(ttl)


def _resolve_files(pattern: str | list[str] | tuple[str, ...]) -> list[pathlib.Path]:
    pats = [str(p) for p in pattern] if isinstance(pattern, (list, tuple)) else [str(pattern)]

    expanded: list[str] = []
    for pat in pats:
        if "[" in pat and "]" in pat:
            try:
                from pivpy.pivmat_compat import expandstr

                expanded.extend(expandstr(pat))
                continue
            except Exception:
                pass
        expanded.append(pat)

    files: list[pathlib.Path] = []
    for pat in expanded:
        matches = [pathlib.Path(p) for p in glob.glob(pat)]
        if matches:
            files.extend(matches)
        else:
            p = pathlib.Path(pat)
            if p.exists() and p.is_file():
                files.append(p)

    # Stable order, de-duplicate.
    uniq: dict[str, pathlib.Path] = {}
    for f in sorted(files, key=lambda x: str(x)):
        uniq[str(f)] = f
    return list(uniq.values())


def _movie_from_iter(
    datasets: "Iterable[tuple[xr.Dataset, str | None]]",
    *,
    output: str | pathlib.Path | None,
    show: str = "auto",
    background: str | None = None,
    scalar: str = "w",
    fps: int = 10,
    dpi: int = 150,
    writer: str | None = None,
    codec: str | None = None,
    title: str | None = None,
    clim: tuple[float, float] | str | None = None,
    cmap: str | None = None,
    return_frames: bool = False,
    close: bool = True,
    ax: Axes | None = None,
    **kwargs,
) -> list[np.ndarray] | None:
    """Core movie loop shared by to_movie() and imvectomovie()."""

    iterator = iter(datasets)
    first, first_file = next(iterator)

    mode = str(show).lower() if show is not None else "auto"
    if mode == "auto":
        mode = "vector" if ("u" in first and "v" in first) else "scalar"
    if mode not in {"vector", "scalar"}:
        raise ValueError("show must be 'auto', 'vector', or 'scalar'")

    # Split kwargs: keep them for artist creation.
    pcolormesh_kwargs = {}
    quiver_kwargs = dict(kwargs)

    if mode == "scalar":
        # pcolormesh accepts lots of kwargs; don't forward quiver compat kwargs.
        for k in list(quiver_kwargs.keys()):
            if k in {"nthArr", "aspectratio", "add_guide", "streamlines", "colorbar", "colorbar_orient", "arrowColor", "scalingFactor", "widthFactor", "quiverKey"}:
                quiver_kwargs.pop(k)
        pcolormesh_kwargs = quiver_kwargs
        quiver_kwargs = {}

    if mode == "vector":
        renderer = _VectorMovieRenderer(
            first=first,
            ax=ax,
            background=background,
            title=title,
            clim=clim,
            cmap=cmap,
            quiver_kwargs=quiver_kwargs,
        )
    else:
        renderer = _ScalarMovieRenderer(
            first=first,
            ax=ax,
            variable=scalar,
            title=title,
            clim=clim,
            cmap=cmap,
            pcolormesh_kwargs=pcolormesh_kwargs,
        )

    fig, _ = renderer.figax

    collector = _FrameCollector() if return_frames else None

    movie_writer = None
    if output is not None:
        out_path = pathlib.Path(str(output))
        ext = out_path.suffix.lower()
        if writer is None:
            if ext in {".gif"}:
                writer = "pillow"
            else:
                writer = "ffmpeg"

        from matplotlib import animation

        if str(writer).lower() == "pillow":
            movie_writer = animation.PillowWriter(fps=int(fps))
        elif str(writer).lower() == "ffmpeg":
            if codec is not None:
                movie_writer = animation.FFMpegWriter(fps=int(fps), codec=str(codec))
            else:
                movie_writer = animation.FFMpegWriter(fps=int(fps))
        else:
            raise ValueError("writer must be None, 'ffmpeg', or 'pillow'")

        movie_writer.setup(fig, str(out_path), dpi=int(dpi))

    # First frame
    renderer.update(first, i=0, file=first_file)
    if collector is not None:
        collector.grab(fig)
    if movie_writer is not None:
        movie_writer.grab_frame()

    # Remaining frames
    frame_index = 1
    for ds, fname in iterator:
        renderer.update(ds, i=frame_index, file=fname)
        if collector is not None:
            collector.grab(fig)
        if movie_writer is not None:
            movie_writer.grab_frame()
        frame_index += 1

    if movie_writer is not None:
        try:
            movie_writer.finish()
        except Exception:
            # Some writers raise on finish if encoding failed.
            raise

    if close:
        plt.close(fig)

    return collector.frames if collector is not None else None


def to_movie(
    data: xr.Dataset,
    output: str | pathlib.Path | None,
    *,
    show: str = "auto",
    background: str | None = None,
    scalar: str = "w",
    fps: int = 10,
    dpi: int = 150,
    writer: str | None = None,
    codec: str | None = None,
    title: str | None = None,
    clim: tuple[float, float] | str | None = None,
    cmap: str | None = None,
    return_frames: bool = False,
    close: bool = True,
    ax: Axes | None = None,
    **kwargs,
) -> list[np.ndarray] | None:
    """Save an in-memory Dataset as a movie (fast artist-updating renderer).

    Parameters
    ----------
    data:
        Dataset to render. If it contains a ``t`` dimension, each ``t`` step is
        rendered as one frame.
    output:
        Output path (e.g. ``'movie.mp4'``, ``'movie.gif'``). If ``None`` and
        ``return_frames=True``, returns a list of RGBA frames.
    show:
        ``'auto'`` (default), ``'vector'``, or ``'scalar'``.
    background:
        In vector mode, optionally draw a scalar background behind vectors.
        See :func:`showf`.
    scalar:
        Variable name to render in scalar mode.
    writer:
        ``None`` (infer from extension), ``'ffmpeg'``, or ``'pillow'``.
    clim:
        Either a fixed ``(vmin, vmax)`` tuple, or ``'each'`` to auto-scale per
        frame.
    return_frames:
        If True, returns a list of RGBA arrays (uint8) for each frame.

    Notes
    -----
    MP4/AVI writing requires FFmpeg. GIF writing requires Pillow.
    """

    ds = data
    if "t" in ds.dims:
        datasets = ((ds.isel(t=i), None) for i in range(int(ds.sizes["t"])))
    else:
        datasets = ((ds, None),)

    return _movie_from_iter(
        datasets,
        output=output,
        show=show,
        background=background,
        scalar=scalar,
        fps=fps,
        dpi=dpi,
        writer=writer,
        codec=codec,
        title=title,
        clim=clim,
        cmap=cmap,
        return_frames=return_frames,
        close=close,
        ax=ax,
        **kwargs,
    )


def imvectomovie(
    filename: str | list[str] | tuple[str, ...],
    output: str | pathlib.Path | None,
    *,
    format: str | None = None,
    show: str = "auto",
    background: str | None = None,
    scalar: str = "w",
    fps: int = 10,
    dpi: int = 150,
    writer: str | None = None,
    codec: str | None = None,
    title: str | None = None,
    clim: tuple[float, float] | str | None = None,
    cmap: str | None = None,
    verbose: bool = False,
    return_frames: bool = False,
    close: bool = True,
    ax: Axes | None = None,
    **kwargs,
) -> list[np.ndarray] | None:
    """Convert a series of vector/scalar files into a movie (PIVMAT-inspired).

    This is the Python analogue of PIVMAT's ``imvectomovie``:
    it loads files one-by-one from disk (no big in-memory list) and writes each
    frame directly to the movie writer.

    Parameters
    ----------
    filename:
        File pattern (glob) or list of patterns/paths.
    output:
        Output movie file (e.g. ``.mp4`` / ``.gif``). If ``None`` and
        ``return_frames=True``, returns a list of RGBA frames.
    format:
        Optional explicit format for ``pivpy.io.read_piv``.
    verbose:
        If True, prints each file loaded.

    Notes
    -----
    Other parameters match :func:`to_movie`.
    """

    import pivpy.io as pio

    files = _resolve_files(filename)
    if not files:
        raise FileNotFoundError("No files matched the given pattern")

    def gen():
        for i, fp in enumerate(files):
            if verbose:
                print(f"Loading file #{i + 1}/{len(files)}: {fp}")
            ds = pio.read_piv(fp, format=format, frame=i)
            yield ds, str(fp)

    return _movie_from_iter(
        gen(),
        output=output,
        show=show,
        background=background,
        scalar=scalar,
        fps=fps,
        dpi=dpi,
        writer=writer,
        codec=codec,
        title=title,
        clim=clim,
        cmap=cmap,
        return_frames=return_frames,
        close=close,
        ax=ax,
        **kwargs,
    )


def jpdfscal_disp(
    jpdf: xr.Dataset,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Display a joint PDF computed by :func:`pivpy.compute_funcs.jpdfscal`.

    This mimics PIVMAT's ``jpdfscal_disp``: it plots ``log10(hi)`` as filled
    contours and draws dashed zero lines.

    Parameters
    ----------
    jpdf:
        Dataset produced by ``jpdfscal`` with coords ``bin1``/``bin2`` and data
        variable ``hi``.
    ax:
        Optional axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if "hi" not in jpdf:
        raise KeyError("jpdf dataset must contain variable 'hi'")
    if "bin1" not in jpdf.coords or "bin2" not in jpdf.coords:
        raise KeyError("jpdf dataset must contain coords 'bin1' and 'bin2'")

    bin1 = np.asarray(jpdf["bin1"].values, dtype=float)
    bin2 = np.asarray(jpdf["bin2"].values, dtype=float)
    hi = np.asarray(jpdf["hi"].values, dtype=float)

    # Avoid log10(0) warnings by masking zeros.
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(hi > 0, np.log10(hi), np.nan)

    levels = [0, 1, 2, 3, 4, 5, 6]
    m = ax.contourf(bin1, bin2, z.T, levels=levels)

    namew1 = str(jpdf.attrs.get("namew1", "s1"))
    namew2 = str(jpdf.attrs.get("namew2", "s2"))
    unitw1 = str(jpdf.attrs.get("unitw1", ""))
    unitw2 = str(jpdf.attrs.get("unitw2", ""))

    xlab = f"{namew1} ({unitw1})" if unitw1 else namew1
    ylab = f"{namew2} ({unitw2})" if unitw2 else namew2
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"Log joint PDF of {namew1} and {namew2}")

    ax.plot([bin1[0], bin1[-1]], [0, 0], "k--")
    ax.plot([0, 0], [bin2[0], bin2[-1]], "k--")
    plt.colorbar(m, ax=ax)
    return fig, ax


def contour_plot(
    data: xr.Dataset,
    property: str = "mag",
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Contour/heatmap style plot for notebooks (compat shim).

    Many legacy notebooks call ``graphics.contour_plot(ds, colorbar=True)``.
    This helper defaults to plotting the vector magnitude ``mag`` computed from
    ``u`` and ``v``.

    Parameters
    ----------
    data:
        Dataset containing at least ``u`` and ``v``.
    property:
        Variable to plot. Special value ``"mag"`` plots ``sqrt(u^2+v^2)``.
    ax:
        Optional matplotlib axis.
    **kwargs:
        Passed through to matplotlib. Recognized compat kwargs:
        ``colorbar`` (bool), ``colorbar_orient`` ("vertical"/"horizontal"),
        ``cmap``, ``levels``.
    """

    from pivpy.graphics_utils import dataset_to_array

    colorbar = bool(kwargs.pop("colorbar", False))
    colorbar_orient = kwargs.pop("colorbar_orient", "vertical")
    cmap = kwargs.pop("cmap", None)
    levels = kwargs.pop("levels", None)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x, y, u, v = dataset_to_array(data)

    if property == "mag":
        z = np.sqrt(u**2 + v**2)
    elif property in data:
        da = data[property]
        z = np.asarray(da.isel(t=0).values if "t" in da.dims else da.values)
    else:
        raise KeyError(f"Property {property} not found in dataset")

    plot_kwargs: dict = {}
    if cmap is not None:
        plot_kwargs["cmap"] = cmap
    if levels is not None:
        plot_kwargs["levels"] = levels

    # Use contourf if levels provided, otherwise pcolormesh.
    if levels is not None:
        m = ax.contourf(x, y, z, **plot_kwargs, **kwargs)
    else:
        m = ax.pcolormesh(x, y, z, shading="auto", **plot_kwargs, **kwargs)

    if colorbar:
        plt.colorbar(m, ax=ax, orientation=colorbar_orient)

    ax.set_aspect("equal")
    return fig, ax


def histogram(data: xr.Dataset, bins: int = 50, ax: plt.Axes | None = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """Plot histograms of u and v for quick diagnostics."""

    # Backwards compatibility: matplotlib removed `normed` in favor of `density`.
    if "normed" in kwargs and "density" not in kwargs:
        kwargs["density"] = kwargs.pop("normed")

    ds = data.isel(t=0) if "t" in data.dims else data
    u = np.asarray(ds["u"].values).ravel()
    v = np.asarray(ds["v"].values).ravel()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.hist(u[~np.isnan(u)], bins=bins, alpha=0.5, label="u", **kwargs)
    ax.hist(v[~np.isnan(v)], bins=bins, alpha=0.5, label="v", **kwargs)
    ax.legend()

    return fig, ax


def histscal_disp(
    data: xr.Dataset,
    smooth: int = 0,
    bin: np.ndarray | None = None,
    opt: str = "ngl",
    *,
    variable: str = "w",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes] | list[tuple[plt.Figure, plt.Axes]]:
    """Display histogram(s) for a scalar field (PIVMAT-inspired).

    Parameters
    ----------
    data:
        Dataset containing the scalar variable.
    smooth:
        Number of consecutive frames to average. Use ``0`` to average over all
        frames (default). If ``smooth>1``, returns a list of figures (one per
        chunk).
    bin:
        Optional bin centers.
    opt:
        Option string. Supported letters:
        ``n`` (normalize to PDF), ``g`` (Gaussian fit), ``l`` (log y-axis).
        To include zeros in the underlying histogram computation, include
        ``'0'`` in ``opt``.
    variable:
        Scalar variable name (default: ``'w'``). If missing and ``'u'`` exists,
        falls back to ``'u'`` for convenience.
    ax:
        Optional axes to plot into (only used when producing a single plot).

    Returns
    -------
    tuple or list of tuple
        ``(fig, ax)`` or a list of ``(fig, ax)`` when multiple chunks are plotted.
    """

    ds = data
    if variable not in ds and "u" in ds and variable == "w":
        variable = "u"
    if variable not in ds:
        raise KeyError(f"Variable {variable} not found in dataset")

    opt_l = str(opt).lower() if opt is not None else ""
    normalize = "n" in opt_l
    gaussian = "g" in opt_l
    logy = "l" in opt_l
    include_zeros = "0" in opt_l

    if "t" in ds.dims:
        nframe = int(ds.sizes.get("t", 1))
    else:
        nframe = 1

    if smooth is None:
        smooth = 0
    smooth_i = int(smooth)
    if smooth_i <= 0:
        smooth_i = nframe

    results: list[tuple[plt.Figure, plt.Axes]] = []
    n_chunks = max(1, nframe // smooth_i)

    for chunk in range(n_chunks):
        t0 = chunk * smooth_i
        t1 = (chunk + 1) * smooth_i
        sub = ds.isel(t=slice(t0, t1)) if "t" in ds.dims else ds

        hds = sub.piv.histf(variable=variable, bin=bin, opt="0" if include_zeros else "")
        centers = np.asarray(hds["bin"].values, dtype=float)
        counts = np.asarray(hds["h"].values, dtype=float)

        delta = float(centers[1] - centers[0]) if centers.size >= 2 else 1.0
        y = counts.copy()
        if normalize:
            denom = float(np.sum(y)) * delta
            if denom != 0:
                y = y / denom

        # Stats for axis limits and Gaussian overlay.
        vals = np.asarray(sub[variable].values, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if not include_zeros:
            vals = vals[vals != 0]
        mean = float(np.mean(vals)) if vals.size else 0.0
        std = float(np.std(vals)) if vals.size else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0

        if ax is not None and n_chunks == 1:
            fig = ax.figure
            use_ax = ax
        else:
            fig, use_ax = plt.subplots()

        use_ax.plot(centers, y, "ro", label="hist")
        use_ax.axvline(0.0, color="k", linewidth=1.0)
        use_ax.set_xlabel(variable)
        use_ax.set_ylabel("pdf" if normalize else "Histogram")

        if logy:
            # Matplotlib warns if attempting log scale with no positive values.
            if np.any(y > 0):
                use_ax.set_yscale("log")

        # x-limits like PIVMAT: +/- 15*std around 0 or around mean.
        if mean < std:
            use_ax.set_xlim(-15.0 * std, 15.0 * std)
        else:
            use_ax.set_xlim(mean - 15.0 * std, mean + 15.0 * std)

        if gaussian:
            xg = np.linspace(float(centers[0]), float(centers[-1]), 2000)
            gauss = 1.0 / (np.sqrt(2.0 * np.pi) * std) * np.exp(-0.5 * (xg**2) / (std**2))
            use_ax.plot(xg, gauss, "b-", label="gauss")

        if "t" in ds.dims:
            use_ax.set_title(f"{variable}: frames {t0}..{t1-1}")
        else:
            use_ax.set_title(f"{variable} histogram")

        if gaussian:
            use_ax.legend()

        results.append((fig, use_ax))

    return results[0] if len(results) == 1 else results


def histvec_disp(
    data: xr.Dataset,
    smooth: int = 0,
    bin: np.ndarray | None = None,
    opt: str = "ngl",
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes] | list[tuple[plt.Figure, plt.Axes]]:
    """Display histogram(s) for a 2D vector field (PIVMAT-inspired).

    Vector components are taken from ``('u','v')`` if present, otherwise
    ``('vx','vy')``.

    Parameters
    ----------
    data:
        Dataset containing vector components.
    smooth, bin, opt, ax:
        Same semantics as :func:`histscal_disp`.

    Returns
    -------
    tuple or list of tuple
        ``(fig, ax)`` or a list of ``(fig, ax)`` when multiple chunks are plotted.
    """

    ds = data
    if "u" in ds and "v" in ds:
        xname, yname = "u", "v"
    elif "vx" in ds and "vy" in ds:
        xname, yname = "vx", "vy"
    else:
        raise KeyError("histvec_disp requires ('u','v') or ('vx','vy')")

    opt_l = str(opt).lower() if opt is not None else ""
    normalize = "n" in opt_l
    gaussian = "g" in opt_l
    logy = "l" in opt_l
    include_zeros = "0" in opt_l

    if "t" in ds.dims:
        nframe = int(ds.sizes.get("t", 1))
    else:
        nframe = 1

    if smooth is None:
        smooth = 0
    smooth_i = int(smooth)
    if smooth_i <= 0:
        smooth_i = nframe

    results: list[tuple[plt.Figure, plt.Axes]] = []
    n_chunks = max(1, nframe // smooth_i)

    for chunk in range(n_chunks):
        t0 = chunk * smooth_i
        t1 = (chunk + 1) * smooth_i
        sub = ds.isel(t=slice(t0, t1)) if "t" in ds.dims else ds

        hds = sub.piv.histf(variable=None, bin=bin, opt="0" if include_zeros else "")
        centers = np.asarray(hds["bin"].values, dtype=float)
        hx = np.asarray(hds["hx"].values, dtype=float)
        hy = np.asarray(hds["hy"].values, dtype=float)

        delta = float(centers[1] - centers[0]) if centers.size >= 2 else 1.0
        if normalize:
            sx = float(np.sum(hx)) * delta
            sy = float(np.sum(hy)) * delta
            if sx != 0:
                hx = hx / sx
            if sy != 0:
                hy = hy / sy

        # Stats for Gaussian overlay.
        def _stats(arr: xr.DataArray) -> tuple[float, float]:
            v = np.asarray(arr.values, dtype=float).ravel()
            v = v[np.isfinite(v)]
            if not include_zeros:
                v = v[v != 0]
            if v.size == 0:
                return 0.0, 1.0
            m = float(np.mean(v))
            s = float(np.std(v))
            if not np.isfinite(s) or s == 0.0:
                s = 1.0
            return m, s

        mx, sx = _stats(sub[xname])
        my, sy = _stats(sub[yname])

        if ax is not None and n_chunks == 1:
            fig = ax.figure
            use_ax = ax
        else:
            fig, use_ax = plt.subplots()

        use_ax.plot(centers, hx, "ro", label=xname)
        use_ax.plot(centers, hy, "bs", label=yname)
        use_ax.axvline(0.0, color="k", linewidth=1.0)
        use_ax.set_xlabel("value")
        use_ax.set_ylabel("pdf" if normalize else "Histogram")
        use_ax.legend()

        if logy:
            # Matplotlib warns if attempting log scale with no positive values.
            if np.any(hx > 0) or np.any(hy > 0):
                use_ax.set_yscale("log")

        if gaussian:
            xg = np.linspace(float(centers[0]), float(centers[-1]), 2000)
            gx = float(np.max(hx)) * np.exp(-0.5 * ((xg - mx) ** 2) / (sx**2))
            gy = float(np.max(hy)) * np.exp(-0.5 * ((xg - my) ** 2) / (sy**2))
            use_ax.plot(xg, gx, "r-", linewidth=1.0)
            use_ax.plot(xg, gy, "b-", linewidth=1.0)

        if "t" in ds.dims:
            use_ax.set_title(f"vector hist: frames {t0}..{t1-1}")
        else:
            use_ax.set_title("vector histogram")

        results.append((fig, use_ax))

    return results[0] if len(results) == 1 else results


def autocorrelation_plot(
    data: xr.Dataset,
    variable: str = "u",
    spatial_average: bool = True,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot a simple temporal autocorrelation for a variable.

    If `spatial_average=True` and t exists, average over x/y before correlating.
    Otherwise, flatten all dimensions.
    """

    if ax is None:
        _, ax = plt.subplots()

    if variable not in data:
        raise KeyError(f"Variable {variable} not in dataset")

    da = data[variable]
    if "t" in da.dims:
        if spatial_average:
            series = da.mean(dim=[d for d in da.dims if d != "t"]).values
        else:
            series = da.values.reshape((-1, da.sizes["t"]))
            series = series.reshape(-1)
    else:
        series = da.values.reshape(-1)

    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    if series.size == 0:
        return ax

    series = series - np.mean(series)
    corr = np.correlate(series, series, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr / (corr[0] if corr[0] != 0 else 1.0)

    ax.plot(corr, **kwargs)
    ax.set_title(f"Autocorrelation: {variable}")
    ax.set_xlabel("lag")
    ax.set_ylabel("corr")
    return ax


def display_vector_field(
    data: xr.Dataset,
    arrowColor: str = "k",
    arrowScale: float = 1.0,
    arrowWidth: float = 0.002,
) -> tuple["Figure", "Axes"]:
    """
    display_vector_field is a wrapper for quiver() for backwards compatibility

    Parameters
    ----------
    data : xarray.Dataset
        dataset with u, v, x, y
    arrowColor : str
        color of the arrows, by default "k"
    arrowScale : float
        scaling factor for the arrows, by default 1.0
    arrowWidth : float
        width factor for the arrows, by default 0.002

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes used for plotting.

    Examples
    --------
    >>> from pivpy import io, graphics
    >>> d = io.loadvec("test.vec")
    >>> graphics.display_vector_field(d)
    """
    return quiver(
        data,
        arrowColor=arrowColor,
        scalingFactor=arrowScale,
        widthFactor=arrowWidth,
    )
