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
from pivpy.io import POS_UNITS, VEL_UNITS
from typing import List
import warnings


def quiver(
    data: xr.DataArray,
    arrScale: float = 25.0,
    threshold: float = None,
    nthArr: int = 1,
    aspectratio: str = "equal",
    colorbar: bool = False,
    colorbar_orient: str = "vertical",
    units: List = [],
    streamlines: bool = False,
):
    """creates quiver of xr.Dataset

    Args:
        data (xr.DataArray): _description_
        arrScale (float, optional): _description_. Defaults to 25.0.
        threshold (float, optional): _description_. Defaults to None.
        nthArr (int, optional): _description_. Defaults to 1.
        aspectratio (str, optional): _description_. Defaults to "equal".
        colorbar (bool, optional): _description_. Defaults to False.
        colorbar_orient (str, optional): _description_. Defaults to "vertical".
        units (List, optional): _description_. Defaults to [].
        streamlines (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    data = dataset_to_array(data)

    pos_units = data.x.attrs["units"] if len(units) == 0 else units[0]
    vel_units = data.u.attrs["units"] if len(units) == 0 else units[2]

    # clip data to the threshold
    if threshold is not None:
        data["u"] = xr.where(data["u"] > threshold, threshold, data["u"])
        data["v"] = xr.where(data["v"] > threshold, threshold, data["v"])

    data["s"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)

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
        scale=np.max(data["s"].values * arrScale),
        headwidth=2,
        ax=ax,
    )

    if colorbar is False:
        # cbar = fig.colorbar(Q, shrink=0.9, orientation=colbar_orient)
        cb = Q.colorbar
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


def histogram(data, normed=False):
    """creates two histograms of two velocity components

    Args:
        data (_type_): _description_
        normed (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
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
    contourLevels: List[float] = None,
    colorbar: bool = False,
    logscale: bool = False,
    aspectratio: str = "equal",
    units: List[str] = [],
):
    """creates contour plot of xr.DataArray

    Args:
        data (xr.DataArray): _description_
        threshold (float, optional): _description_. Defaults to None.
        contourLevels (List[float], optional): _description_. Defaults to None.
        colorbar (bool, optional): _description_. Defaults to False.
        logscale (bool, optional): _description_. Defaults to False.
        aspectratio (str, optional): _description_. Defaults to "equal".
        units (List[str], optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    data = dataset_to_array(data)

    if "w" not in data.var():
        data = data.piv.vec2scal("ke")

    if threshold is not None:
        data["w"] = xr.where(data["w"] > threshold, threshold, data["w"])

    f, ax = plt.subplots()
    # data.plot.contourf(x='x',y='y',row='y',col='x', ax=ax)

    if contourLevels is None:
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
    """shows data as quiver over a scalar background

    Args:
        data (_type_): _description_
        flow_property (str, optional): _description_. Defaults to "ke".
    """
    fig, ax = showscal(data, flow_property=flow_property, **kwargs)
    fig, ax = quiver(data, **kwargs)


def showscal(data, flow_property="ke", **kwargs):
    """creates contour plot of some scalar field of a flow property

    Args:
        data (_type_): _description_
        flow_property (str, optional): _description_. Defaults to "ke".

    Returns:
        _type_: _description_
    """
    data = data.piv.vec2scal(flow_property=flow_property)
    fig, ax = contour_plot(data, **kwargs)
    return fig, ax


def animate(data: xr.Dataset, arrowscale: int = 1, savepath: str = None):
    """animates flow fields in the data and saves to MP4 format

    Args:
        data (xr.Dataset): _description_
        arrowscale (int, optional): _description_. Defaults to 1.
        savepath (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
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
            "Warning: function for a single frame, using the first \
               frame, supply data.isel(t=N)"
        )
        return data.isel(t=t_index)
    
    
    return data
