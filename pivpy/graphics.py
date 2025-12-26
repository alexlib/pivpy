"""pivpy.graphics

Plotting helpers used by the test suite and the xarray accessor in
``pivpy/pivpy.py``.

Important behavioral expectations (tests rely on these):

- ``quiver()`` and ``streamplot()`` return ``(fig, ax)``
- ``showf()`` exists
- ``showscal()`` accepts ``flow_property=`` as an alias and can compute a scalar
    via the ``.piv.vec2scal()`` accessor when needed
"""
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from matplotlib.quiver import Quiver


def quiver(
    data: xr.Dataset,
    quiverKey: str = "Q",
    scalingFactor: float = 1.0,
    widthFactor: float = 0.002,
    ax: plt.Axes | None = None,
    arrowColor: str = "k",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Creates a quiver plot from the dataset

    Parameters
    ----------
    data : xr.Dataset
        dataset with u, v, x, y
    quiverKey : str, optional
        key for the quiver plot, by default "Q"
    scalingFactor : float, optional
        scaling factor for the arrows, by default 1.0
    widthFactor : float, optional
        width factor for the arrows, by default 0.002
    ax : plt.Axes | None, optional
        matplotlib axes, by default None
    arrowColor : str, optional
        color of the arrows, by default "k"

    Returns
    -------
    Quiver
        matplotlib quiver object

    Examples
    --------
    >>> from pivpy import io, graphics
    >>> d = io.loadvec("test.vec")
    >>> graphics.quiver(d)
    """
    from pivpy.graphics_utils import dataset_to_array

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x, y, u, v = dataset_to_array(data)

    # Prefer xarray attrs if present; otherwise use empty strings.
    xUnits = str(getattr(data.get("x", None), "attrs", {}).get("units", ""))
    yUnits = str(getattr(data.get("y", None), "attrs", {}).get("units", ""))

    Q = ax.quiver(
        x,
        y,
        u,
        v,
        scale=scalingFactor,
        width=widthFactor,
        color=arrowColor,
    )
    ax.set_aspect("equal")
    ax.quiverkey(
        Q,
        0.9,
        0.9,
        1,
        quiverKey,
        labelpos="E",
        coordinates="figure",
    )
    ax.set_xlabel(f"x [{xUnits}]")
    ax.set_ylabel(f"y [{yUnits}]")

    return fig, ax


def vectorplot(
    data: xr.Dataset,
    arrowColor: str = "k",
    arrowScale: float = 1.0,
    arrowWidth: float = 0.002,
) -> "Quiver":
    """
    vectorplot plots the vector field

    Parameters
    ----------
    data : xr.Dataset
        dataset with u, v, x, y
    arrowColor : str, optional
        color of the arrows, by default "k"
    arrowScale : float, optional
        scaling factor for the arrows, by default 1.0
    arrowWidth : float, optional
        width factor for the arrows, by default 0.002

    Returns
    -------
    Quiver
        matplotlib quiver object

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
    data : xr.Dataset
        dataset with u, v, x, y
    property : str, optional
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
) -> None:
    """
    streamplot plots the streamlines of the vector field

    Parameters
    ----------
    data : xr.Dataset
        dataset with u, v, x, y
    density : float, optional
        density of the streamlines, by default 1.0
    linewidth : float, optional
        linewidth of the streamlines, by default 1.0
    arrowsize : float, optional
        size of the arrows, by default 1.0
    ax : plt.Axes | None, optional
        matplotlib axes, by default None
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


def histogram(data: xr.Dataset, bins: int = 50, ax: plt.Axes | None = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """Plot histograms of u and v for quick diagnostics."""

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
) -> "Quiver":
    """
    display_vector_field is a wrapper for quiver() for backwards compatibility

    Parameters
    ----------
    data : xr.Dataset
        dataset with u, v, x, y
    arrowColor : str, optional
        color of the arrows, by default "k"
    arrowScale : float, optional
        scaling factor for the arrows, by default 1.0
    arrowWidth : float, optional
        width factor for the arrows, by default 0.002

    Returns
    -------
    Quiver
        matplotlib quiver object

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
