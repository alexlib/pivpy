# -*- coding: utf-8 -*-
"""
Various plots, mostly wraping xarray.plot 

"""
import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr
import pandas as pd
from pivpy.io import POS_UNITS, VEL_UNITS
from typing import List
import warnings


def quiver(
    data: xr.DataArray,
    arrow_scale: float = 25.0,
    threshold: float = None,
    nth_arrow: int = 1,
    aspectratio: str = "equal",
    colorbar: bool = False,
    colorbar_orient: str = "vertical",
    units: List = [],
    streamlines: bool = False,
    cmap: str = 'RdBu',
    **kwargs,
):
    """Creates quiver plot of velocity field from xarray Dataset
    
    Args:
        data (xr.DataArray): PIV velocity field data
        arrow_scale (float, optional): Arrow scaling factor. Defaults to 25.0.
        threshold (float, optional): Maximum velocity magnitude to display. Defaults to None.
        nth_arrow (int, optional): Display every nth arrow for subsampling. Defaults to 1.
        aspectratio (str, optional): Aspect ratio of the plot. Defaults to "equal".
        colorbar (bool, optional): Whether to show colorbar. Defaults to False.
        colorbar_orient (str, optional): Orientation of colorbar ('vertical' or 'horizontal'). 
            Defaults to "vertical".
        units (List, optional): List of units [pos_unit, pos_unit, vel_unit, vel_unit]. 
            Defaults to [].
        streamlines (bool, optional): Whether to overlay streamlines. Defaults to False.
        cmap (str, optional): Matplotlib colormap name (e.g., 'jet', 'hot', 'RdBu', 'Reds'). 
            Defaults to 'RdBu'.
        **kwargs: Additional keyword arguments passed to xarray.plot.quiver
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Backward compatibility for old parameter names
    if 'arrScale' in kwargs:
        warnings.warn("'arrScale' is deprecated, use 'arrow_scale' instead", DeprecationWarning)
        arrow_scale = kwargs.pop('arrScale')
    if 'nthArr' in kwargs:
        warnings.warn("'nthArr' is deprecated, use 'nth_arrow' instead", DeprecationWarning)
        nth_arrow = kwargs.pop('nthArr')
    
    data = dataset_to_array(data)

    pos_units = data.x.attrs["units"] if len(units) == 0 else units[0]
    vel_units = data.u.attrs["units"] if len(units) == 0 else units[2]

    # subsampling number of vectors
    data = data.sel(x=data.x[::nth_arrow], y=data.y[::nth_arrow])  

    # clip data to the threshold
    if threshold is not None:
        data["u"] = xr.where(data["u"] > threshold, threshold, data["u"])
        data["v"] = xr.where(data["v"] > threshold, threshold, data["v"])

    data["s"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    
    # Sort y in increasing order if needed (streamplot requires increasing coordinates)
    # This is needed when streamlines=True
    if streamlines and len(data.y) > 1 and data.y[0] > data.y[-1]:
        data = data.sortby('y')
    

    if len(plt.get_fignums()) == 0:  # if no figure is open
        fig, ax = plt.subplots()  # open a new figure
    else:
        fig = plt.gcf()
        ax = fig.gca()

    # quiver itself
    Q = data.plot.quiver(
        x="x",
        y="y",
        u="u",
        v="v",
        hue="s",
        units="width",
        scale=np.max(data["s"].values * arrow_scale),
        headwidth=2,
        cmap=cmap,
        ax=ax,
        **kwargs,
    )

    if colorbar is False:
        cb = Q.colorbar
        if cb:
            cb.remove()
        plt.draw()
    else:
        if colorbar_orient == "horizontal":
            cb = Q.colorbar
            cb.remove()
            cb = fig.colorbar(Q, orientation=colorbar_orient, ax=ax)

    if streamlines:  # contours or streamlines
        strm = data.plot.streamplot(
            x="x",
            y="y",
            u="u",
            v="v",
            hue="s",
            cmap="hot",
            linewidth=1,
            ax=ax,
        )
        strm.colorbar.remove()

        # if colorbar:
        #     cbar = fig.colorbar(
        #         strm,
        #         orientation=colorbar_orient,
        #         fraction=0.1,
        #     )
        #     cbar.set_label(r"$ V \, (" + vel_units + r")$")

    ax.set_xlabel(f"x ({pos_units})")
    ax.set_ylabel(f"y ({pos_units})")
    ax.set_aspect(aspectratio)
    # ax.invert_yaxis()

    return fig, ax


def streamplot(
    data: xr.DataArray,
    threshold: float = None,
    aspectratio: str = "equal",
    colorbar: bool = False,
    colorbar_orient: str = "vertical",
    units: List = [],
    cmap: str = 'hot',
    linewidth: float = 1.0,
    density: float = 1.0,
    **kwargs,
):
    """Creates streamplot of velocity field from xarray Dataset
    
    Args:
        data (xr.DataArray): PIV velocity field data
        threshold (float, optional): Maximum velocity magnitude to display. Defaults to None.
        aspectratio (str, optional): Aspect ratio of the plot. Defaults to "equal".
        colorbar (bool, optional): Whether to show colorbar. Defaults to False.
        colorbar_orient (str, optional): Orientation of colorbar ('vertical' or 'horizontal'). 
            Defaults to "vertical".
        units (List, optional): List of units [pos_unit, pos_unit, vel_unit, vel_unit]. 
            Defaults to [].
        cmap (str, optional): Matplotlib colormap name (e.g., 'jet', 'hot', 'RdBu', 'Reds'). 
            Defaults to 'hot'.
        linewidth (float, optional): Width of streamlines. Defaults to 1.0.
        density (float, optional): Density of streamlines. Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to xarray.plot.streamplot
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    data = dataset_to_array(data)

    pos_units = data.x.attrs["units"] if len(units) == 0 else units[0]
    vel_units = data.u.attrs["units"] if len(units) == 0 else units[2]

    # clip data to the threshold
    if threshold is not None:
        data["u"] = xr.where(data["u"] > threshold, threshold, data["u"])
        data["v"] = xr.where(data["v"] > threshold, threshold, data["v"])

    data["s"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    
    # Sort y in increasing order if needed (streamplot requires increasing coordinates)
    if len(data.y) > 1 and data.y[0] > data.y[-1]:
        data = data.sortby('y')

    if len(plt.get_fignums()) == 0:  # if no figure is open
        fig, ax = plt.subplots()  # open a new figure
    else:
        fig = plt.gcf()
        ax = fig.gca()

    # streamplot itself
    strm = data.plot.streamplot(
        x="x",
        y="y",
        u="u",
        v="v",
        hue="s",
        cmap=cmap,
        linewidth=linewidth,
        density=density,
        ax=ax,
        **kwargs,
    )

    if colorbar is False:
        if strm.colorbar is not None:
            strm.colorbar.remove()
        plt.draw()
    else:
        if colorbar_orient == "horizontal":
            if strm.colorbar is not None:
                strm.colorbar.remove()
            cb = fig.colorbar(strm, orientation=colorbar_orient, ax=ax)

    ax.set_xlabel(f"x ({pos_units})")
    ax.set_ylabel(f"y ({pos_units})")
    ax.set_aspect(aspectratio)

    return fig, ax


def histogram(data, normed=False):
    """Creates histograms of velocity components
    
    Args:
        data (xr.Dataset): PIV dataset with u and v velocity components
        normed (bool, optional): Whether to normalize the histogram. Defaults to False.
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects with two histogram subplots
    """

    u = np.asarray(data.u).flatten()
    v = np.asarray(data.v).flatten()

    f, ax = plt.subplots(2)

    ax[0].hist(u, bins=np.int32(np.sqrt(len(u)) * 0.5), density=normed)
    ax[0].set_xlabel(f"u ({data.u.attrs['units']})")

    ax[1] = plt.subplot2grid((2, 1), (1, 0))
    ax[1].hist(v, bins=np.int32(np.sqrt(len(v) * 0.5)), density=normed)
    ax[1].set_xlabel(f"v ({data.v.attrs['units']})")
    plt.tight_layout()
    return f, ax


def contour_plot(
    data: xr.DataArray,
    threshold: float = None,
    contour_levels: List[float] = None,
    colorbar: bool = False,
    logscale: bool = False,
    aspectratio: str = "equal",
    units: List[str] = [],
    **kwargs,
):
    """Creates contour plot of scalar field from xarray Dataset
    
    Args:
        data (xr.DataArray): PIV dataset with scalar field 'w' or vector fields
        threshold (float, optional): Maximum value to clip the data. Defaults to None.
        contour_levels (List[float], optional): Specific contour levels to plot. Defaults to None.
        colorbar (bool, optional): Whether to show colorbar. Defaults to False.
        logscale (bool, optional): Whether to use logarithmic scale. Defaults to False.
        aspectratio (str, optional): Aspect ratio of the plot. Defaults to "equal".
        units (List[str], optional): List of units for axes labels. Defaults to [].
        **kwargs: Additional keyword arguments (including deprecated 'contourLevels')
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Backward compatibility for old parameter name
    if 'contourLevels' in kwargs:
        warnings.warn("'contourLevels' is deprecated, use 'contour_levels' instead", DeprecationWarning)
        if contour_levels is None:  # Only use if not already specified
            contour_levels = kwargs.pop('contourLevels')
        else:
            kwargs.pop('contourLevels')  # Remove to avoid confusion
    
    data = dataset_to_array(data)

    if "w" not in data.var():
        data = data.piv.vec2scal("ke")

    if threshold is not None:
        data["w"] = xr.where(data["w"] > threshold, threshold, data["w"])

    f, ax = plt.subplots()
    # data.plot.contourf(x='x',y='y',row='y',col='x', ax=ax)

    if contour_levels is None:
        levels = np.linspace(
            np.min(data["w"].values),
            np.max(data["w"].values),
            10,
        )
    else:
        levels = contour_levels  # vector of levels to set

    if logscale:
        data["w"] = np.abs(data["w"])

        c = data["w"].plot.contourf(
            x="x",
            y="y",
            levels=levels,
            cmap=plt.get_cmap("RdYlBu"),
            norm=colors.LogNorm(),
            ax=ax,
        )
    else:
        c = data["w"].plot.contourf(
            x="x",
            y="y",
            levels=levels,
            cmap=plt.get_cmap("RdYlBu"),
            ax=ax,
        )

    if not colorbar:
        # cbar = c.colorbar(c, orientation=colbar)
        c.colorbar.remove()
        # cbar.set_label(propUnits)

    ax.set_aspect(aspectratio)

    return f, ax


def showf(data, flow_property="ke", **kwargs):
    """Shows velocity field as quiver plot over a scalar background
    
    Args:
        data (xr.Dataset): PIV dataset with velocity fields
        flow_property (str, optional): Scalar flow property to show as background 
            (e.g., 'ke', 'vorticity', 'strain'). Defaults to "ke".
        **kwargs: Additional keyword arguments passed to quiver and contour_plot
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = showscal(data, flow_property=flow_property, **kwargs)
    fig, ax = quiver(data, **kwargs)


def showscal(data, flow_property="ke", **kwargs):
    """Creates contour plot of a scalar flow property field
    
    Args:
        data (xr.Dataset): PIV dataset with velocity or scalar fields
        flow_property (str, optional): Flow property to visualize. 
            Options: 'ke' (kinetic energy), 'vorticity', 'strain', 'divergence', etc.
            Defaults to "ke".
        **kwargs: Additional keyword arguments passed to contour_plot
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    data = data.piv.vec2scal(flow_property=flow_property)
    fig, ax = contour_plot(data, **kwargs)
    return fig, ax


def animate(data: xr.Dataset,
            arrow_scale: int = 1,
            savepath: str = None,
            units: str = "pix/dt"):
    """Animates flow fields and saves to MP4 format
    
    Args:
        data (xr.Dataset): PIV dataset with time-series velocity fields
        arrow_scale (int, optional): Arrow scaling factor for quiver plot. Defaults to 1.
        savepath (str, optional): Path to save the animation MP4 file. Defaults to None.
        units (str, optional): Units for velocity. Defaults to "pix/dt".
        
    Returns:
        FuncAnimation: matplotlib animation object
    """
    arrowscale = arrow_scale  # Use consistent naming internally
    X, Y = np.meshgrid(data.x, data.y)
    X = X.T
    Y = Y.T
    U, V = data.u[:, :, 0], data.v[:, :, 0]  # first frame
    fig, ax = plt.subplots(1, 1)
    M = np.sqrt(U**2 + V**2)

    Q = ax.quiver(
        X[::3, ::3],
        Y[::3, ::3],
        U[::3, ::3],
        V[::3, ::3],
        M[::3, ::3],
        units="inches",
        scale=arrowscale,
    )

    cb = plt.colorbar(Q)

    # units = data.attrs["units"]

    cb.ax.set_ylabel(f"velocity ({units})")

    text = ax.text(
        0.2,
        1.05,
        "1/" + str(len(data.t)),
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    def update_quiver(num, Q, data, text):
        """_summary_

        Args:
            num (_type_): _description_
            Q (_type_): _description_
            data (_type_): _description_
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        U, V = data.u[:, :, num], data.v[:, :, num]

        M = np.sqrt(U[::3, ::3] ** 2 + V[::3, ::3] ** 2)
        Q.set_UVC(U[::3, ::3], V[::3, ::3], M[::3, ::3])
        text.set_text(str(num + 1) + "/" + str(len(data.t)))
        return Q

    anim = FuncAnimation(
        fig,
        update_quiver,
        fargs=(Q, data, text),
        frames=len(data.t),
        blit=False,
    )
    mywriter = FFMpegWriter()
    if savepath:
        p = os.getcwd()
        os.chdir(savepath)
        anim.save("im.mp4", writer=mywriter)
        os.chdir(p)
    else:
        anim.save("im.mp4", writer=mywriter)


def dataset_to_array(data: xr.Dataset, t_index: int = 0):
    """converts xarray Dataset to array"""
    if "t" in data.dims:
        warnings.warn(
            "Warning: this function uses the first \
               frame, otherwise use: data.isel(t=N)"
        )
        return data.isel(t=t_index)
    
    
    return data


def autocorrelation_plot(
    data: xr.Dataset,
    variable: str = "u",
    spatial_average: bool = True,
    **kwargs,
):
    """Creates autocorrelation plot of a specified variable from xarray Dataset
    
    This function plots the autocorrelation of a time series extracted from the dataset.
    The autocorrelation shows how correlated a signal is with itself at different time lags.
    
    Args:
        data (xr.Dataset): PIV dataset with velocity or scalar fields
        variable (str, optional): Variable name to plot autocorrelation for 
            (e.g., 'u', 'v', 'w', 'c', or any other data variable). Defaults to "u".
        spatial_average (bool, optional): If True and time dimension exists, compute 
            spatial average before temporal autocorrelation. If False, flatten all 
            dimensions. Defaults to True for proper temporal analysis.
        **kwargs: Additional keyword arguments passed to pandas.plotting.autocorrelation_plot
        
    Returns:
        matplotlib.axes.Axes: The axes object containing the autocorrelation plot
        
    Raises:
        ValueError: If the specified variable is not found in the dataset
        
    Example:
        >>> data = io.load_vec(filename)
        >>> # Temporal autocorrelation (spatial average over time)
        >>> autocorrelation_plot(data, variable='u', spatial_average=True)
        >>> # Autocorrelation of flattened data (all dimensions)
        >>> autocorrelation_plot(data, variable='u', spatial_average=False)
        >>> # For scalar fields like vorticity
        >>> data = data.piv.vec2scal('curl')
        >>> autocorrelation_plot(data, variable='w')
        
    Note:
        When spatial_average=True and time dimension 't' exists, the function 
        computes the spatial average over x and y dimensions first, then analyzes 
        temporal autocorrelation. This provides proper temporal correlation analysis.
        When spatial_average=False, all dimensions are flattened, which may mix 
        spatial and temporal variations.
    """
    if variable not in data.data_vars:
        available_vars = list(data.data_vars)
        raise ValueError(
            f"Variable '{variable}' not found in dataset. "
            f"Available variables: {available_vars}"
        )
    
    # Extract the variable
    var_data = data[variable]
    
    # Determine how to extract the time series
    if spatial_average and 't' in var_data.dims:
        # Compute spatial average to get proper temporal series
        spatial_dims = [dim for dim in var_data.dims if dim != 't']
        if spatial_dims:
            var_data = var_data.mean(dim=spatial_dims)
        series_data = var_data.values
    else:
        # Flatten all dimensions (original behavior from gist)
        series_data = var_data.values.flatten()
    
    # Create pandas Series
    series = pd.Series(series_data)
    
    # Create the autocorrelation plot
    ax = pd.plotting.autocorrelation_plot(series, **kwargs)
    
    # Get units if available
    units = data[variable].attrs.get("units", "")
    
    # Update the plot title and labels
    title_suffix = " (spatial avg)" if (spatial_average and 't' in data[variable].dims) else ""
    ax.set_title(f"Autocorrelation of {variable}{title_suffix}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    
    if units:
        ax.set_title(f"Autocorrelation of {variable}{title_suffix} ({units})")
    
    return ax
