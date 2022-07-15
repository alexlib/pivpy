# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:02:49 2015

@author: Ron, Alex
"""
import numpy as np

# from scipy.stats import norm
# from scipy.ndimage.filters import gaussian_filter
# from scipy.ndimage.filters import median_filter
import xarray as xr
from pivpy import graphics
from scipy.ndimage import gaussian_filter as gf

""" learn from this example


import xarray as xr


@xr.register_dataset_accessor('geo')
class GeoAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None

    @property
    def center(self):
        " Return the geographic center point of this dataset."
        if self._center is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            lon = self._obj.latitude
            lat = self._obj.longitude
            self._center = (float(lon.mean()), float(lat.mean()))
        return self._center

    def plot(self):
        " Plot data on a map."
        return 'plotting!'


    In [1]: ds = xr.Dataset({'longitude': np.linspace(0, 10),
   ...:                  'latitude': np.linspace(0, 20)})
   ...:

In [2]: ds.geo.center
Out[2]: (10.0, 5.0)

In [3]: ds.geo.plot()
Out[3]: 'plotting!'

"""


@xr.register_dataset_accessor("piv")
class PIVAccessor(object):
    def __init__(self, xarray_obj):
        """
        Arguments:
            data : xarray Dataset:
            x,y,t are coordinates
            u,v,chs are the data arrays

        We add few shortcuts (properties):
            data.piv.average is the time average (data.mean(dim='t'))
            data.piv.dt is the shortcut to get $\\Delta t$
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
        self._dt = None

    @property
    def average(self):
        """ Return the mean flow field ."""
        if self._average is None:  # only first time
            self._average = self._obj.mean(dim="t")
            self._average.attrs = self._obj.attrs  # we need units in quiver
            self._average.assign_coords({"t": 0})

        return self._average

    def crop(self, crop_vector=[None, None, None, None]):
        """ crop number of rows, cols from either side of the vector fields
        Input:
            self : xarray Dataset
            crop_vector : [xmin,xmax,ymin,ymax] is a list of values crop
                            the data, defaults are None
        Return:
            same object as the input
        """
        xmin, xmax, ymin, ymax = crop_vector

        xmin = self._obj.x.min() if xmin is None else xmin
        xmax = self._obj.x.max() if xmax is None else xmax
        ymin = self._obj.y.min() if ymin is None else ymin
        ymax = self._obj.y.max() if ymax is None else ymax

        self._obj = self._obj.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        return self._obj

    def pan(self, dx=0.0, dy=0.0):
        """ moves the field by dx,dy in the same units as x,y """
        self._obj = self._obj.assign_coords(
            {"x": self._obj.x + dx, "y": self._obj.y + dy}
        )
        # self._obj['x'] += dx
        # self._obj['y'] += dy
        return self._obj

    def filterf(self):
        """Gaussian filtering of velocity """

        self._obj["u"] = xr.DataArray(
            gf(self._obj["u"].values, [1, 1, 0]), dims=("x", "y", "t")
        )
        self._obj["v"] = xr.DataArray(
            gf(self._obj["v"].values, [1, 1, 0]), dims=("x", "y", "t")
        )

        return self._obj

    def __add__(self, other):
        """ add two datasets means that we sum up the velocities, assume
        that x,y,t,dt are all identical
        """
        self._obj["u"] += other._obj["u"]
        self._obj["v"] += other._obj["v"]
        return self._obj

    def __sub__(self, other):
        """ add two datasets means that we sum up the velocities, assume
        that x,y,t,dt are all identical
        """
        self._obj["u"] -= other._obj["u"]
        self._obj["v"] -= other._obj["v"]
        return self._obj

    def vorticity(self):
        """ calculates vorticity of the data array (at one time instance) and
        adds it to the attributes

        Input:
            xarray with the variables u,v and dimensions x,y

        Output:
            xarray with the estimated vorticity as a scalar field with
            same dimensions

        """

        # ux, uy = np.gradient(self._obj['u'], self._obj['x'],
        #                    self._obj['y'], axis=(0, 1))
        # vx, vy = np.gradient(self._obj['v'], self._obj['x'],
        #                    self._obj['y'], axis=(0, 1))
        # self._obj['w'] = xr.DataArray(vy - ux, dims=['x', 'y'])
        # _w = xr.DataArray(vy - ux, dims=['x', 'y', 't'])
        # if 't' in self._obj.coords:
        self._obj["w"] = (
            self._obj.differentiate("x")["v"]
            - self._obj.differentiate("y")["u"]
        )
        # else:
        #     self._obj["w"] = (("x", "y"), vx - uy)
        # self._obj = self._obj.assign(w=_w)
        # self._obj.assign(w=vy-ux)

        if len(self._obj.attrs["units"]) < 5:
            # vel_units = self._obj.attrs['units'][-1]
            self._obj.attrs["units"].append("1/dt")
            self._obj.attrs['variables'].append('vorticity')
        else:
            # vel_units = self._obj.attrs['units'][-2]
            self._obj.attrs["units"][-1] = "1/dt"
            self._obj.attrs['variables'][-1] = 'vorticity'

        return self._obj

    def strain(self):
        """ calculates shear of the data array (single frame)
        Input:
            xarray with the variables u,v and dimensions x,y
        Output:
            xarray with the estimated shear as a scalar field data['w']
        """
        ux = self._obj.differentiate("x")["u"]
        uy = self._obj.differentiate("y")["u"]
        vx = self._obj.differentiate("x")["v"]
        vy = self._obj.differentiate("y")["v"]

        self._obj["w"] = ux ** 2 + vy ** 2 + 0.5 * (uy + vx) ** 2

        if len(self._obj.attrs["units"]) < 5:
            # vel_units = self._obj.attrs['units'][-1]
            self._obj.attrs["units"].append("1/dt")
            self._obj.attrs['variables'].append('strain')
        else:
            # vel_units = self._obj.attrs['units'][-2]
            self._obj.attrs["units"][-1] = "1/dt"
            self._obj.attrs['variables'][1] = 'strain'

        return self._obj

    def divergence(self):
        """ calculates shear of the data array (single frame)
        Input:
            xarray with the variables u,v and dimensions x,y
        Output:
            xarray with the estimated shear as a scalar field data['w']
        """
        ux, _ = np.gradient(
            self._obj["u"], self._obj["x"], self._obj["y"], axis=(0, 1)
        )
        _, vy = np.gradient(
            self._obj["v"], self._obj["x"], self._obj["y"], axis=(0, 1)
        )

        if "t" in self._obj.coords:
            self._obj["w"] = (("x", "y", "t"), vy + ux)
        else:
            self._obj["w"] = (("x", "y"), vy + ux)

        if len(self._obj.attrs["units"]) == 4:
            # vel_units = self._obj.attrs['units'][-1]
            self._obj.attrs["units"].append("1/dt")
            self._obj.attrs['variables'].append('divergence')
        else:
            # vel_units = self._obj.attrs['units'][-2]
            self._obj.attrs["units"][-1] = "1/dt"
            self._obj.attrs['variables'][-1] = 'divergence'

        return self._obj

    def acceleration(self):
        """ calculates material derivative or acceleration of the
        data array (single frame)

        Input:
            xarray with the variables u,v and dimensions x,y

        Output:
            xarray with the estimated acceleration as a scalar field data['w']

        """
        ux = self._obj.differentiate("x")["u"]
        uy = self._obj.differentiate("y")["u"]
        vx = self._obj.differentiate("x")["v"]
        vy = self._obj.differentiate("y")["v"]

        ax = self._obj["u"] * ux + self._obj["v"] * uy
        ay = self._obj["u"] * vx + self._obj["v"] * vy

        self._obj["w"] = xr.DataArray(
            np.sqrt(ax ** 2 + ay ** 2), dims=["x", "y", "t"]
        )

        if len(self._obj.attrs["units"]) == 4:
            vel_units = self._obj.attrs["units"][-1]
            self._obj.attrs["units"].append(f"{vel_units}^2")
            self._obj.attrs['variables'].append('acceleration')
        else:
            vel_units = self._obj.attrs["units"][-2]
            self._obj.attrs["units"][-1] = f"{vel_units}^2"
            self._obj.attrs['variables'][-1] = 'acceleration'

        return self._obj

    def ke(self):
        """ estimates turbulent kinetic energy """
        self._obj["w"] = self._obj["u"] ** 2 + self._obj["v"] ** 2

        if len(self._obj.attrs["units"]) == 4:
            vel_units = self._obj.attrs["units"][-1]
            self._obj.attrs["units"].append(f"({vel_units})^2")
            self._obj.attrs['variables'].append('ke')
        else:
            vel_units = self._obj.attrs["units"][-2]
            self._obj.attrs["units"][-1] = f"({vel_units})^2"
            self._obj.attrs['variables'][-1] = 'ke'
        return self._obj

    def tke(self):
        """ estimates turbulent kinetic energy """
        if len(self._obj.t) < 2:
            raise ValueError(
                "TKE is not defined for a single vector field, \
                              use .piv.ke()"
            )

        self._obj["w"] = (
            self._obj["u"] - self._obj["u"].mean(dim="t")
        ) ** 2 + (self._obj["v"] - self._obj["v"].mean(dim="t")) ** 2
        vel_units = self._obj.attrs["units"][-1]
        if len(self._obj.attrs['units']) < 5:
            self._obj.attrs["units"].append(f"({vel_units})^2")
            self._obj.attrs["variables"].append("tke")
        else:
            self._obj.attrs["units"][-1]  = (f"({vel_units})^2")
            self._obj.attrs["variables"][-1] = "tke"
        return self._obj

    def fluct(self):
        """ returns fluctuations as a new dataset """

        if len(self._obj.t) < 2:
            raise ValueError(
                "fluctuations cannot be defined for a \
                              single vector field, use .piv.ke()"
            )

        new_obj = self._obj.copy()
        new_obj -= new_obj.mean(dim="t")
        return new_obj

    def reynolds_stress(self):
        """ returns fluctuations as a new dataset """

        if len(self._obj.t) < 2:
            raise ValueError(
                "fluctuations cannot be defined for a \
                              single vector field, use .piv.ke()"
            )

        new_obj = self._obj.copy()
        tmp = new_obj.mean(dim="t")
        new_obj -= tmp # fluctuations
        new_obj["w"] = new_obj["u"] * new_obj["v"] # new scalar
        new_obj = new_obj.mean(dim="t") # reynolds stress is -\rho < u' v'>

        return new_obj

    def vec2scal(self, property="curl"):
        """ creates a dataset of scalar values on the same
        dimensions and coordinates as the vector dataset
        Agruments:
            data : xarray.DataSet with u,v on t,x,y grid
        Returns:
            scalar_data : xarray.Dataset w on t,x,y grid
            'w' represents one of the following properties:
                - 'curl' or 'rot' - vorticity

        """
        # replace few common names
        property = "vorticity" if property == "curl" else property
        property = "tke" if property == "ken" else property
        property = "vorticity" if property == "vort" else property

        method_name = str(property)
        method = getattr(self, method_name, lambda: "nothing")

        # if len(self._obj.attrs["variables"]) < 5:  # only x,y,u,v
        #     self._obj.attrs["variables"].append(property)
        # else:
        #     self._obj.attrs["variables"][-1] = property

        return self._obj, method()

    def __mul__(self, scalar):
        """
        multiplication of a velocity field by a scalar (simple scaling)
        """
        self._obj["u"] *= scalar
        self._obj["v"] *= scalar
        if 'w' in self._obj.var():
            self._obj['w'] += scalar

        return self._obj

    def __div__(self, scalar):
        """
        multiplication of a velocity field by a scalar (simple scaling)
        """
        self._obj["u"] /= scalar
        self._obj["v"] /= scalar

        return self._obj

    def set_dt(self, dt):
        self._obj.attrs["dt"] = dt
        return self._obj

    def set_scale(self, scale=1.0):
        for var in ['x','y','u','v']:
            self._obj[var] = self._obj[var]*scale

        return self._obj

    def set_tUnits(self, tUnits):
        self._obj.attrs["tUnits"] = tUnits

    def rotate(self, theta=0.0):
        """
        use this method in order to rotate the data
        by theta degrees in the clockwise direction
        in the present form of using xarray with coordinates
        it can only work for the cases with equal size along
        x and y
        """

        theta = theta / 360.0 * 2 * np.pi

        xi = self._obj.x * np.cos(theta) + self._obj.y * np.sin(theta)
        eta = self._obj.y * np.cos(theta) - self._obj.x * np.sin(theta)
        Uxi = self._obj.u * np.cos(theta) + self._obj.v * np.sin(theta)
        Ueta = self._obj.v * np.cos(theta) - self._obj.u * np.sin(theta)

        self._obj["x"] = xi
        self._obj["y"] = eta
        self._obj["u"] = Uxi
        self._obj["v"] = Ueta

        if "theta" in self._obj:
            self._obj["theta"] += theta
        else:
            self._obj["theta"] = theta

        return self._obj

    @property
    def dt(self):
        """ receives the dt from the set """
        if self._dt is None:
            self._dt = self._obj.attrs["dt"]
        return self._dt

    def quiver(self, **kwargs):
        """ graphics.quiver() as a property """
        fig, ax = graphics.quiver(self._obj, **kwargs)
        return fig, ax

    def streamplot(self, **kwargs):
        """ graphics.quiver(streamlines=True) """
        graphics.quiver(self._obj, streamlines=True, **kwargs)

    def showf(self, **kwargs):
        """ method for graphics.showf """
        graphics.showf(self._obj, **kwargs)

    def showscal(self, **kwargs):
        """ method for graphics.showscal """
        graphics.showscal(self._obj, **kwargs)

    # @property
    # def vel_units(self):
    #     " Return the geographic center point of this dataset."
    #     if self._vel_units is None:
    #         self._vel_units = self._obj.attrs.l_units + '/' + \
    #                           self._obj.attrs.t_units
    #     return self._vel_units
