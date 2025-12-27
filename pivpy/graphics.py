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
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if TYPE_CHECKING:
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
    """Simple vector-field display (compat shim).

    Historically, `showf` existed as a convenience wrapper. For the tests we only
    need it to be callable without error.
    """

    return quiver(data, **kwargs)


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
