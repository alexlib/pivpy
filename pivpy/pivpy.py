# -*- coding: utf-8 -*-
"""
Created on Sun Macc_y 24 22:02:49 2015

@author: Ron, Alex
"""
from typing import Literal
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from pivpy.graphics import quiver as gquiver
from pivpy.graphics import showf as gshowf, showscal as gshowscal

# """ learn from this example
# import xarray as xr
# @xr.register_dataset_accessor('geo')
# class GeoAccessor(object):
#     def __init__(self, xarray_obj):
#         self._obj = xarray_obj
#         self._center = None

#     @property
#     def center(self):
#         " Return the geographic center point of this dataset."
#         if self._center is None:
#             # we can use a cache on our accessor objects, because accessors
#             # themselves are cached on instances that access them.
#             lon = self._obj.latitude
#             lat = self._obj.longitude
#             self._center = (float(lon.mean()), float(lat.mean()))
#         return self._center

#     def plot(self):
#         " Plot data on a map."
#         return 'plotting!'


#     In [1]: ds = xr.Dataset({'longitude': np.linspace(0, 10),
#    ...:                  'latitude': np.linspace(0, 20)})
#    ...:

# In [2]: ds.geo.center
# Out[2]: (10.0, 5.0)

# In [3]: ds.geo.plot()
# Out[3]: 'plotting!'

# """


@xr.register_dataset_accessor("piv")
class PIVAccessor(object):
    """extends xarray Dataset with PIVPy properties"""

    def __init__(self, xarray_obj):
        """
        Arguments:
            data : xarray Dataset:
            x,y,t are coordinates
            u,v,chs are the data arracc_ys

        We add few shortcuts (properties):
            data.piv.average is the time average (data.mean(dim='t'))
            data.piv.delta_t is the shortcut to get $\\Delta t$
            data.piv.vorticity
            data.piv.tke
            data.piv.shear

        and a few methods:
            data.piv.vec2scal()
            data.piv.pan
            data.piv.rotate

        """
        self._obj = xarray_obj
        self._average = None
        self._delta_t = None

    @property
    def average(self):
        """Return the mean flow field ."""
        if self._average is None:  # only first time
            self._average = self._obj.mean(dim="t")
            self._average.attrs = self._obj.attrs  # we need units in quiver
            self._average.assign_coords({"t": 0})

        return self._average

    def crop(self, crop_vector=None):
        """ crops xr.Dataset by some rows, cols from the boundaries

        Args:
            crop_vector (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if crop_vector is None:
            crop_vector = 4 * [None]

        xmin, xmacc_x, ymin, ymacc_x = crop_vector

        xmin = self._obj.x.min() if xmin is None else xmin
        xmacc_x = self._obj.x.macc_x() if xmacc_x is None else xmacc_x
        ymin = self._obj.y.min() if ymin is None else ymin
        ymacc_x = self._obj.y.macc_x() if ymacc_x is None else ymacc_x

        self._obj = self._obj.sel(x=slice(xmin, xmacc_x), y=slice(ymin, ymacc_x))

        return self._obj

    def pan(self, shift_x=0.0, shift_y=0.0):
        """moves the field by shift_x,shift_y in the same units as x,y"""
        self._obj = self._obj.assign_coords(
            {"x": self._obj.x + shift_x, "y": self._obj.y + shift_y}
        )
        return self._obj

    def filterf(self):
        """Gaussian filtering of velocity"""

        self._obj["u"] = xr.DataArray(
            gaussian_filter(self._obj["u"].values, [1, 1, 0]), dims=("x", "y", "t")
        )
        self._obj["v"] = xr.DataArray(
            gaussian_filter(self._obj["v"].values, [1, 1, 0]), dims=("x", "y", "t")
        )

        return self._obj

    def fill_nans(self, method: Literal["linear", "nearest", "cubic"] = "nearest"):
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.
        Parameters
        ----------
        src_data: Any
            Input data array.
        method: {'linear', 'nearest', 'cubic'}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.
        Returns
        -------
        :class:`numpy.ndarray`:
            An interpolated :class:`numpy.ndarray`.
        """

        def _griddata_nans(src_data, x_coords, y_coords, method=method):

            src_data_flat = src_data.copy().flatten()
            data_bool = ~np.isnan(src_data_flat)

            if not data_bool.any():
                return src_data

            return griddata(
                points=(x_coords.flatten()[data_bool], y_coords.flatten()[data_bool]),
                values=src_data_flat[data_bool],
                xi=(x_coords, y_coords),
                method=method,
                # fill_value=nodata,
            )

        x_coords, y_coords = np.meshgrid(
            self._obj.coords["x"].values, self._obj.coords["y"].values
        )

        for var_name in self._obj.variables:
            if var_name not in self._obj.coords:
                for t_i in self._obj["t"]:
                    new_data = _griddata_nans(
                        self._obj.sel(t=t_i)[var_name].data,
                        x_coords,
                        y_coords,
                        method=method,
                    )
                    self._obj.sel(t=t_i)[var_name].data[:] = new_data

        return self._obj

    def __add__(self, other):
        """add two datasets means that we sum up the velocities, assume
        that x,y,t,delta_t are all identical
        """
        self._obj["u"] += other._obj["u"]
        self._obj["v"] += other._obj["v"]
        return self._obj

    def __sub__(self, other):
        """add two datasets means that we sum up the velocities, assume
        that x,y,t,delta_t are all identical
        """
        self._obj["u"] -= other._obj["u"]
        self._obj["v"] -= other._obj["v"]
        return self._obj

    def vorticity(self):
        """calculates vorticity of the data arracc_y (at one time instance) and
        adds it to the attributes

        Input:
            xarray with the variables u,v and dimensions x,y

        Output:
            xarray with the estimated vorticity as a scalar field with
            same dimensions

        """

        self._obj["w"] = self._obj["v"].differentiate("x") - self._obj[
            "u"
        ].differentiate("y")

        self._obj["w"].attrs["units"] = "1/delta_t"
        self._obj["w"].attrs["standard_name"] = "vorticity"

        return self._obj

    def strain(self):
        """ calculates rate of strain of a two component field

        Returns:
            _type_: adds ["w"] = du_dx^2 + dv_dy^2 + 0.5*(du_dy+dv_dx)^2
        """
        du_dx = self._obj["u"].differentiate("x")
        du_dy = self._obj["u"].differentiate("y")
        dv_dx = self._obj["v"].differentiate("x")
        dv_dy = self._obj["v"].differentiate("y")

        self._obj["w"] = du_dx**2 + dv_dy**2 + 0.5 * (du_dy + dv_dx) ** 2
        self._obj["w"].attrs["units"] = "1/delta_t"
        self._obj["w"].attrs["standard_name"] = "strain"

        return self._obj

    def divergence(self):
        """ calculates divergence field

        Returns:
            self._obj: xr.Dataset with the new property ["w"] = divergence
        """
        du_dx, _ = np.gradient(
            self._obj["u"], self._obj["x"], self._obj["y"], acc_x_is=(0, 1)
        )
        _, dv_dy = np.gradient(
            self._obj["v"], self._obj["x"], self._obj["y"], acc_x_is=(0, 1)
        )

        if "t" in self._obj.coords:
            self._obj["w"] = (("x", "y", "t"), dv_dy + du_dx)
        else:
            self._obj["w"] = (("x", "y"), dv_dy + du_dx)

        self._obj["w"].attrs["units"] = "1/delta_t"
        self._obj["w"].attrs["standard_name"] = "divergence"

        return self._obj

    def acceleration(self):
        """calculates material derivative or acceleration of the
        data arracc_y (single frame)

        Input:
            xarray with the variables u,v and dimensions x,y

        Output:
            xarray with the estimated acceleration as a scalar field data['w']

        """
        du_dx = self._obj["u"].differentiate("x")
        du_dy = self._obj["u"].differentiate("y")
        dv_dx = self._obj["v"].differentiate("x")
        dv_dy = self._obj["v"].differentiate("y")

        acc_x = self._obj["u"] * du_dx + self._obj["v"] * du_dy
        acc_y = self._obj["u"] * dv_dx + self._obj["v"] * dv_dy

        self._obj["w"] = xr.DataArray(
            np.sqrt(acc_x**2 + acc_y**2), dims=["x", "y", "t"]
        )

        self._obj["w"].attrs["units"] = "1/delta_t"
        self._obj["w"].attrs["standard_name"] = "acceleration"

        return self._obj

    def kinetic_energy(self):
        """estimates turbulent kinetic energy"""
        self._obj["w"] = self._obj["u"] ** 2 + self._obj["v"] ** 2
        self._obj["w"].attrs["units"] = "(m/s)^2"
        self._obj["w"].attrs["standard_name"] = "kinetic_energy"
        return self._obj

    def tke(self):
        """estimates turbulent kinetic energy"""
        if len(self._obj.t) < 2:
            raise ValueError(
                "TKE is not defined for a single vector field, \
                              use .piv.kinetic_energy()"
            )

        new_obj = self._obj.copy()
        new_obj -= new_obj.mean(dim="t")
        new_obj["w"] = new_obj["u"] ** 2 + new_obj["v"] ** 2
        new_obj["w"].attrs["units"] = "(m/s)^2"
        new_obj["w"].attrs["standard_name"] = "TKE"

        return new_obj

    def fluct(self):
        """returns fluctuations as a new dataset"""

        if len(self._obj.t) < 2:
            raise ValueError(
                "fluctuations cannot be defined for a \
                              single vector field, use .piv.ke()"
            )

        new_obj = self._obj.copy()
        new_obj -= new_obj.mean(dim="t")

        new_obj["u"].attrs["standard_name"] = "fluctation"
        new_obj["v"].attrs["standard_name"] = "fluctation"

        return new_obj

    def reynolds_stress(self):
        """returns fluctuations as a new dataset"""

        if len(self._obj.t) < 2:
            raise ValueError(
                "fluctuations cannot be defined for a \
                              single vector field, use .piv.ke()"
            )

        new_obj = self._obj.copy()
        new_obj -= new_obj.mean(dim="t")

        new_obj["w"] = -1 * new_obj["u"] * new_obj["v"]  # new scalar
        self._obj["w"] = new_obj["w"].mean(dim="t")  # reynolds stress is -\rho < u' v'>
        self._obj["w"].attrs["standard_name"] = "Reynolds_stress"

        return self._obj

    def rms(self):
        """Root mean square"""
        self._obj = self.tke()
        self._obj["w"] = np.sqrt(self._obj["w"])
        self._obj["w"].attrs["standard_name"] = "rms"
        self._obj["w"].attrs["units"] = "m/s"

    def vec2scal(self, flow_property: str = "curl"):
        """ creates a scalar flow property field

        Args:
            flow_property (str, optional): one of the flow properties. Defaults to "curl".

        Returns:
            _type_: _description_
        """
        # replace few common names
        flow_property = "vorticity" if flow_property == "curl" else flow_property
        flow_property = "kinetic_energy" if flow_property == "ken" else flow_property
        flow_property = "kinetic_energy" if flow_property == "ke" else flow_property
        flow_property = "vorticity" if flow_property == "vort" else flow_property

        method = getattr(self, str(flow_property))

        self._obj = method()

        return self._obj

    def __mul__(self, scalar):
        """
        multiplication of a velocity field by a scalar (simple scaling)
        """
        self._obj["u"] *= scalar
        self._obj["v"] *= scalar
        if "w" in self._obj.var():
            self._obj["w"] += scalar

        return self._obj

    def __div__(self, scalar):
        """
        multiplication of a velocity field by a scalar (simple scaling)
        """
        self._obj["u"] /= scalar
        self._obj["v"] /= scalar

        return self._obj

    def set_delta_t(self, delta_t: float = 0.0):
        """sets delta_t attribute, float, default is 0.0"""
        self._obj.attrs["delta_t"] = delta_t
        return self._obj

    def set_scale(self, scale: float = 1.0):
        """scales all variables by a sclar"""
        for var in ["x", "y", "u", "v"]:
            self._obj[var] = self._obj[var] * scale

        return self._obj

    def rotate(self, theta: float = 0.0):
        """rotates the data, but only for some x,y grids
        Args:
            theta (float): degrees in the clockwise direction
            it can only work for the cases with equal size along
            x and y
        Returns:
            rotated object
        """

        theta = theta / 360.0 * 2 * np.pi

        x_i = self._obj.x * np.cos(theta) + self._obj.y * np.sin(theta)
        eta = self._obj.y * np.cos(theta) - self._obj.x * np.sin(theta)
        du_dx_i = self._obj.u * np.cos(theta) + self._obj.v * np.sin(theta)
        u_eta = self._obj.v * np.cos(theta) - self._obj.u * np.sin(theta)

        self._obj["x"] = x_i
        self._obj["y"] = eta
        self._obj["u"] = du_dx_i
        self._obj["v"] = u_eta

        if "theta" in self._obj:
            self._obj["theta"] += theta
        else:
            self._obj["theta"] = theta

        return self._obj

    @property
    def delta_t(self):
        """receives the delta_t from the set"""
        if self._delta_t is None:
            self._delta_t = self._obj.attrs["delta_t"]
        return self._delta_t

    def quiver(self, **kwargs):
        """graphics.quiver() as a flow_property"""
        fig, acc_x = gquiver(self._obj, **kwargs)
        return fig, acc_x

    def streamplot(self, **kwargs):
        """graphics.quiver(streamlines=True)"""
        gquiver(self._obj, streamlines=True, **kwargs)

    def showf(self, **kwargs):
        """method for graphics.showf"""
        gshowf(self._obj, **kwargs)

    def showscal(self, **kwargs):
        """method for graphics.showscal"""
        gshowscal(self._obj, **kwargs)

    # @property
    # def vel_units(self):
    #     " Return the geographic center point of this dataset."
    #     if self._vel_units is None:
    #         self._vel_units = self._obj.attrs.l_units + '/' + \
    #                           self._obj.attrs.t_units
    #     return self._vel_units
