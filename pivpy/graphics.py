"""
pivpy.graphics
==============
This module is responsible for plotting and visualization 
using matplotlib and xarray built-in methods
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import xarray as xr

from pivpy.pivpy import dataset_to_array


def quiver(
    data: xr.Dataset,
    arrScale: int = 20,
    threshold: float = None,
    width: float = 0.0025,
    bg_image: Union[Path, xr.DataArray] = None,
    bg_alpha: float = 1.0,
    fig_ax=None,
    colorbar=False,
    clim=None,
    **kwargs,
):
    """
    piv.quiver(data) creates quiver plot
    Inputs:
        data - an xarray Dataset with dimensions (x,y) and
                    data arrays (u,v,w,chc). Typically piv.vec

        arrScale - use to adjust arrow scale (default is 20)
        threshold - if provided, creates a filtered quiver plot
                   removing vectors based on a property provided
                   as the third input, e.g. threshold = ('chc',0.1)
                   will only display vectors where chc > 0.1

        width : float - arrow width
        bg_image : xarray, optional. It is used for the background image
        bg_alpha : transparency of the image
        **kwargs : can be used to pass additional arguments to matplotlib
                   quiver function, e.g. cmap='gray', clim = [0,1], etc.
    """

    data = dataset_to_array(data)

    # Import here to avoid circular import with io.py
    if len(units) == 0:
        try:
            from pivpy.io import POS_UNITS, VEL_UNITS
            pos_units = data.x.attrs.get("units", POS_UNITS)
            vel_units = data.u.attrs.get("units", VEL_UNITS)
        except ImportError:
            # Fallback if io module doesn't define these constants
            pos_units = data.x.attrs.get("units", "pix")
            vel_units = data.u.attrs.get("units", "pix")
    else:
        pos_units = units[0]
        vel_units = units[2]

    # if only some arrows are requested
    if threshold is not None:
        attr, level = threshold
        data = data.where(data[attr] > level, drop=True)

    if fig_ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    else:
        ax = fig_ax

    if bg_image is not None:
        if isinstance(bg_image, Path):
            img = plt.imread(bg_image)
            ax.imshow(img, alpha=bg_alpha, cmap=kwargs.pop("cmap", "gray"))
        else:
            bg_image.plot.imshow(ax=ax, alpha=bg_alpha, add_colorbar=False)

    # Create quiver plot with the data
    if colorbar:
        # Calculate magnitude for color mapping
        import numpy as np

        magnitude = np.sqrt(data.u**2 + data.v**2)
        Q = ax.quiver(
            data.x,
            data.y,
            data.u,
            data.v,
            magnitude,
            scale=arrScale,
            width=width,
            **kwargs,
        )
        cbar = plt.colorbar(Q, ax=ax)
        cbar.set_label(f"Velocity magnitude [{vel_units}]")

        if clim is not None:
            Q.set_clim(clim)
    else:
        ax.quiver(
            data.x, data.y, data.u, data.v, scale=arrScale, width=width, **kwargs
        )

    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{pos_units}]")
    ax.set_ylabel(f"y [{pos_units}]")
    ax.invert_yaxis()

    return ax


def showf(data: xr.Dataset, property: str = "chc", **kwargs):
    """
    showf(data, property = 'chc') displays an image
    of property of the velocity field
    Input:
        data : xr.Dataset - dataset with dimensions (x,y,t), e.g.
                            typically vec
        property : str - name of the property to display, e.g.
                         chc, u, v, w, etc.
    """

    data = dataset_to_array(data)

    if property in ("u", "v", "w"):
        data[property].plot.imshow(**kwargs)
    else:
        data[property].plot.imshow(**kwargs)

    return plt.gca()


def quivert(
    data: xr.Dataset,
    frame: int = 0,
    arrScale: int = 20,
    threshold: float = None,
    width: float = 0.0025,
    **kwargs,
):
    """
    piv.quiver(data, frame = 0) creates quiver plot at a
     specific time instance
    Inputs:
        data - an xarray Dataset with dimensions (x,y,t) and
                    data arrays (u,v,w,chc). Typically piv.vec
        frame - integer, frame to plot

    see piv.quiver for additional inputs and properties
    """
    data = dataset_to_array(data)

    if "t" in data.dims:
        data = data.isel(t=frame)

    return quiver(data, arrScale, threshold, width, **kwargs)


def streamplot(
    data: xr.Dataset,
    threshold: float = None,
    bg_image: Union[Path, xr.DataArray] = None,
    bg_alpha: float = 1.0,
    fig_ax=None,
    colorbar=False,
    clim=None,
    **kwargs,
):
    """
    piv.streamplot(data) creates streamline plot
    Inputs:
        data - an xarray Dataset with dimensions (x,y) and
                    data arrays (u,v,w,chc). Typically piv.vec

        threshold - if provided, creates a filtered streamline plot
                   removing vectors based on a property provided
                   as the third input, e.g. threshold = ('chc',0.1)
                   will only display vectors where chc > 0.1

        bg_image : xarray, optional. It is used for the background image
        bg_alpha : transparency of the image
        **kwargs : can be used to pass additional arguments to matplotlib
                   streamplot function, e.g. color='k', density=1.5, etc.
    """

    data = dataset_to_array(data)

    # Import here to avoid circular import with io.py
    if len(units) == 0:
        try:
            from pivpy.io import POS_UNITS, VEL_UNITS
            pos_units = data.x.attrs.get("units", POS_UNITS)
            vel_units = data.u.attrs.get("units", VEL_UNITS)
        except ImportError:
            # Fallback if io module doesn't define these constants
            pos_units = data.x.attrs.get("units", "pix")
            vel_units = data.u.attrs.get("units", "pix")
    else:
        pos_units = units[0]
        vel_units = units[2]

    # if only some arrows are requested
    if threshold is not None:
        attr, level = threshold
        data = data.where(data[attr] > level, drop=True)

    if fig_ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    else:
        ax = fig_ax

    if bg_image is not None:
        if isinstance(bg_image, Path):
            img = plt.imread(bg_image)
            ax.imshow(img, alpha=bg_alpha, cmap=kwargs.pop("cmap", "gray"))
        else:
            bg_image.plot.imshow(ax=ax, alpha=bg_alpha, add_colorbar=False)

    # Create streamplot with the data
    if colorbar:
        # Calculate magnitude for color mapping
        import numpy as np

        magnitude = np.sqrt(data.u**2 + data.v**2)
        strm = ax.streamplot(
            data.x.values,
            data.y.values,
            data.u.values,
            data.v.values,
            color=magnitude.values,
            **kwargs,
        )
        cbar = plt.colorbar(strm.lines, ax=ax)
        cbar.set_label(f"Velocity magnitude [{vel_units}]")

        if clim is not None:
            strm.lines.set_clim(clim)
    else:
        ax.streamplot(
            data.x.values, data.y.values, data.u.values, data.v.values, **kwargs
        )

    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{pos_units}]")
    ax.set_ylabel(f"y [{pos_units}]")
    ax.invert_yaxis()

    return ax
