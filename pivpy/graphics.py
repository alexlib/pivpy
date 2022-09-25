# -*- coding: utf-8 -*-
"""
Various plots, mostly wraping xarray.plot 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr
import os
from pivpy.io import POS_UNITS, VEL_UNITS
from typing import List
import warnings


def quiver(
    data: xr.DataArray,
    arrScale: float=25.0,
    threshold: float=None,
    nthArr: int=1,
    aspectratio: str="equal",
    colorbar: bool=False,
    colorbar_orient: str="vertical",
    units: List=[],
    streamlines: bool=False
):
    """
    Generates a quiver plot of a 'data' xarray DataArray object (single frame
    from a dataset)
    Inputs:
        data - xarray DataArray of the type defined in pivpy, one of the
        frames in the Dataset
            selected by default using .isel(t=0)
        threshold - values above the threshold will be set equal to threshold
        arrScale - use to change arrow scales
        nthArr - use to plot only every nth arrow from the array
        contourLevels - use to specify the maximum value (abs) of contour plots
        colorbar - True/False wether to generate a colorbar or not
        logscale - if true then colorbar is on log scale
        aspectratio - set auto or equal for the plot's apearence
        colorbar_orient - 'horizontal' or 'vertical' orientation of the colorbar
        (if colbar is True)
    Outputs:
        none
    Usage:
        graphics.quiver(data, arrScale = 0.2, threshold = Inf, n)
    """

    data = dataset_to_array(data)


    pos_units = data.x.attrs["units"] if len(units)==0 else units[0]
    vel_units = data.u.attrs["units"] if len(units)==0 else units[2]

    # clip data to the threshold
    if threshold is not None:
        data["u"] = xr.where(data["u"] > threshold, threshold, data["u"])
        data["v"] = xr.where(data["v"] > threshold, threshold, data["v"])

    data['s'] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)

    if len(plt.get_fignums()) == 0:  # if no figure is open
        fig, ax = plt.subplots()  # open a new figure
    else:
        fig = plt.gcf()
        ax = fig.gca()

    # quiver itself
    Q = data.plot.quiver(
            x='x',
            y='y',
            u='u',
            v='v',
            hue='s',
            units='width',
            scale=np.max(data['s'].values * arrScale),
            headwidth=2,
            ax=ax,
            )

    if colorbar is False:
        # cbar = fig.colorbar(Q, shrink=0.9, orientation=colbar_orient)
        cb = Q.colorbar
        cb.remove()
        plt.draw()
    else:
        if colorbar_orient == 'horizontal':
            cb = Q.colorbar
            cb.remove()
            cb = fig.colorbar(Q, orientation=colorbar_orient, ax=ax)


    if streamlines:  # contours or streamlines
        strm = data.plot.streamplot(
            x='x',
            y='y',
            u='u',
            v='v',
            hue='s',
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


def histogram(data, normed=False):
    """
    this function will plot a normalized histogram of
    the velocity data.
    Input:
        data : xarray DataSet with ['u','v'] attrs['units']
        normed : (optional) default is False to present normalized
        histogram
    """

    u = np.asarray(data.u).flatten()
    v = np.asarray(data.v).flatten()

    f, ax = plt.subplots(2)

    ax[0].hist(u, bins = np.int32(np.sqrt(len(u)) * 0.5), density=normed)
    ax[0].set_xlabel(f"u ({data.u.attrs['units']})")

    ax[1] = plt.subplot2grid((2, 1), (1, 0))
    ax[1].hist(v, bins = np.int32(np.sqrt(len(v) * 0.5)), density=normed)
    ax[1].set_xlabel(f"v ({data.v.attrs['units']})")
    plt.tight_layout()
    return f, ax


def contour_plot(
    data: xr.DataArray,
    threshold: float=None,
    contourLevels: List[float]=[],
    colorbar: bool=False,
    logscale: bool=False,
    aspectratio: str="equal",
    units: List[str] = [],
):
    """ contourf ajusted for the xarray PIV dataset, creates a
        contour map for the data['w'] property.
        Input:
            data : xarray PIV DataArray, converted automatically using
            .isel(t=0)
            threshold : a threshold value, default is None (no data clipping)
            contourLevels : number of contour levels, default is None
            colbar : None (hide), 'horizontal', or 'vertical'
            logscale : boolean (True is default) create in linear/log scale
            aspectration : string, 'equal' is the default
    """

    data = dataset_to_array(data)    

    if "w" not in data.var():
        data = data.piv.vec2scal("ke")
        
    if threshold is not None:
        data["w"] = xr.where(data["w"] > threshold, threshold, data["w"])


    f, ax = plt.subplots()
    # data.plot.contourf(x='x',y='y',row='y',col='x', ax=ax)

    if len(contourLevels) == 0:
        levels = np.linspace(
            np.min(data["w"].values), 
            np.max(data["w"].values), 
            10,
        )
    else:
        levels = contourLevels  # vector of levels to set

    if logscale:
        data["w"] = np.abs(data["w"])

        c = data["w"].plot.contourf(
            x='x',
            y='y',
            levels=levels,
            cmap=plt.get_cmap("RdYlBu"),
            norm=plt.colors.LogNorm(),
            ax=ax,
        )
    else:
        c = data["w"].plot.contourf(
            x='x',
            y='y',
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


def showf(data, property="ke", **kwargs):
    """
    showf(data, var, units)
    Arguments:
        data : xarray.DataSet that contains dimensions of t,x,y
               and variables u,v and maybe w (scalar)
    """
    fig, ax = showscal(data, property=property, **kwargs)
    fig, ax = quiver(data,**kwargs)


def showscal(data, property="ke", **kwargs):
    """
    showf(data, var, units)
    Arguments:
        data : xarray.DataSet that contains dimensions of t,x,y
               and a variable w (scalar)
    """
    data = data.piv.vec2scal(property=property)
    fig, ax = contour_plot(data)
    return fig, ax


def animate(data, arrowscale=1, savepath=None):
    """ animates the quiver plot for the dataset (multiple frames)
    Input:
        data : xarray PIV type of DataSet
        arrowscale : [optional] integer, default is 1
        savepath : [optional] path to save the MP4 animation, default is None

    Output:
        if savepath is None, then only an image display of the animation
        if savepath is an existing path, a file named im.mp4 is saved

    """
    X, Y = np.meshgrid(data.x, data.y)
    X = X.T
    Y = Y.T
    U, V = data.u[:, :, 0], data.v[:, :, 0]  # first frame
    fig, ax = plt.subplots(1, 1)
    M = np.sqrt(U ** 2 + V ** 2)

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

    units = data.attrs["units"]

    cb.ax.set_ylabel("velocity (" + units[2] + ")")

    text = ax.text(
        0.2,
        1.05,
        "1/" + str(len(data.t)),
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    def update_quiver(num, Q, data, text):
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


def dataset_to_array(data:xr.Dataset, t:int=0):
    """ converts xarray Dataset to array """
    if "t" in data.dims:
        warnings.warn("Warning: function for a single frame, using the first \
               frame, supply data.isel(t=N)")
        return data.isel(t=t)
    else:
        return data