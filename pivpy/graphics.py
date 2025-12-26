"""
This module contains all graphical tools for PIVPy
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
) -> "Quiver":
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
        _, ax = plt.subplots()

    x, y, u, v = dataset_to_array(data)
    
    # Import here to avoid circular import
    try:
        from pivpy.io import POS_UNITS, VEL_UNITS
    except ImportError:
        POS_UNITS = {"m": 1, "mm": 1e3, "um": 1e6, "micron": 1e6}
        VEL_UNITS = {"m/s": 1, "mm/s": 1e3, "um/s": 1e6, "micron/s": 1e6}

    xUnits = data.attrs.get("xUnits", "m")
    yUnits = data.attrs.get("yUnits", "m")
    velUnits = data.attrs.get("velUnits", "m/s")
    velUnitsScaling = VEL_UNITS.get(velUnits, 1.0)
    xUnitsScaling = POS_UNITS.get(xUnits, 1.0)
    yUnitsScaling = POS_UNITS.get(yUnits, 1.0)

    # convert to the right units
    x = x * xUnitsScaling
    y = y * yUnitsScaling
    u = u * velUnitsScaling
    v = v * velUnitsScaling

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

    return Q


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

    x, y, _, _ = dataset_to_array(data)
    if property in data:
        plt.pcolormesh(x, y, data[property], **kwargs)
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
    
    # Import here to avoid circular import
    try:
        from pivpy.io import POS_UNITS, VEL_UNITS
    except ImportError:
        POS_UNITS = {"m": 1, "mm": 1e3, "um": 1e6, "micron": 1e6}
        VEL_UNITS = {"m/s": 1, "mm/s": 1e3, "um/s": 1e6, "micron/s": 1e6}

    xUnits = data.attrs.get("xUnits", "m")
    yUnits = data.attrs.get("yUnits", "m")
    velUnits = data.attrs.get("velUnits", "m/s")
    velUnitsScaling = VEL_UNITS.get(velUnits, 1.0)
    xUnitsScaling = POS_UNITS.get(xUnits, 1.0)
    yUnitsScaling = POS_UNITS.get(yUnits, 1.0)

    # convert to the right units
    x = x * xUnitsScaling
    y = y * yUnitsScaling
    u = u * velUnitsScaling
    v = v * velUnitsScaling

    if ax is None:
        _, ax = plt.subplots()

    ax.streamplot(
        x[0, :],
        y[:, 0],
        u,
        v,
        density=density,
        linewidth=linewidth,
        arrowsize=arrowsize,
        **kwargs,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{xUnits}]")
    ax.set_ylabel(f"y [{yUnits}]")


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
