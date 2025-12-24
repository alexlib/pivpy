# -*- coding: utf-8 -*-
"""
This script extends the functionality of xarray.Dataset by adding a new accessor called piv. The accessor adds several properties and methods that are useful for working with particle image velocimetry (PIV) data. The properties include average, which returns the mean flow field, and delta_t, which returns the time step used in the PIV measurement. The methods include crop, which allows the user to crop the data by a given number of rows and columns from the boundaries, vec2scal, which converts vector data to scalar data, pan, which pans the data by a given number of pixels, and rotate, which rotates the data by a given angle.


@author: Ron, Alex
"""
try:
    from typing_extensions import Literal
except ImportError:
    from typing import Literal
from typing import List

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from pivpy.graphics import quiver as gquiver
from pivpy.graphics import showf as gshowf
from pivpy.graphics import showscal as gshowscal
from pivpy.compute_funcs import Γ1_moving_window_function, Γ2_moving_window_function

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
            u,v,chc are the data arrays

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
        """Crops xarray Dataset to specified spatial boundaries
        
        Args:
            crop_vector (list, optional): List of [xmin, xmax, ymin, ymax] values 
                to define cropping boundaries. Use None for any value to keep 
                the original boundary. Defaults to None (no cropping).
                
        Returns:
            xr.Dataset: Cropped dataset
            
        Example:
            >>> data = data.piv.crop([5, 15, -5, -15])  # Crop to x:[5,15], y:[-5,-15]
            >>> data = data.piv.crop([None, 20, None, None])  # Crop only xmax to 20
        """
        if crop_vector is None:
            crop_vector = 4 * [None]

        xmin, xmax, ymin, ymax = crop_vector

        xmin = self._obj.x.min() if xmin is None else xmin
        xmax = self._obj.x.max() if xmax is None else xmax
        ymin = self._obj.y.min() if ymin is None else ymin
        ymax = self._obj.y.max() if ymax is None else ymax

        self._obj = self._obj.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        return self._obj

    def pan(self, shift_x=0.0, shift_y=0.0):
        """Shifts the coordinate system by specified amounts
        
        Args:
            shift_x (float, optional): Amount to shift in x direction. Defaults to 0.0.
            shift_y (float, optional): Amount to shift in y direction. Defaults to 0.0.
            
        Returns:
            xr.Dataset: Dataset with shifted coordinates
            
        Example:
            >>> data = data.piv.pan(10.0, -5.0)  # Shift x by +10, y by -5
        """
        self._obj = self._obj.assign_coords(
            {"x": self._obj.x + shift_x, "y": self._obj.y + shift_y}
        )
        return self._obj

    def filterf(self, sigma: List[float]=[1.,1.,0], **kwargs):
        """Applies Gaussian filtering to velocity fields
        
        Args:
            sigma (List[float], optional): Standard deviation for Gaussian kernel 
                in [y, x, t] dimensions. Defaults to [1., 1., 0] (spatial filtering only).
            **kwargs: Additional keyword arguments passed to scipy.ndimage.gaussian_filter
            
        Returns:
            xr.Dataset: Filtered dataset with smoothed velocity fields
            
        Example:
            >>> data = data.piv.filterf(sigma=[2., 2., 0])  # Smooth with sigma=2 in space
        """

        self._obj["u"] = xr.DataArray(
            gaussian_filter(
                self._obj["u"].values, sigma, **kwargs),
                dims=("y", "x", "t"),
                attrs = self._obj["u"].attrs,
        )
        self._obj["v"] = xr.DataArray(
            gaussian_filter(
                self._obj["v"].values, sigma, **kwargs),
                dims=("y", "x", "t"),
                attrs = self._obj["v"].attrs,
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
        """calculates vorticity of the data array (at one time instance) and
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
            self._obj["u"], self._obj["x"], self._obj["y"], axis=(0, 1)
        )
        _, dv_dy = np.gradient(
            self._obj["v"], self._obj["x"], self._obj["y"], axis=(0, 1)
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
        data array (single frame)

        Input:
            xarray with the variables u,v and dimensions x,y

        Output:
            xarray with the estimated acceleration as a scalar field data['w']

        """
        du_dx = self._obj["u"].differentiate("x")
        du_dy = self._obj["u"].differentiate("y")
        dv_dx = self._obj["v"].differentiate("x")
        dv_dy = self._obj["v"].differentiate("y")

        accel_x = self._obj["u"] * du_dx + self._obj["v"] * du_dy
        accel_y = self._obj["u"] * dv_dx + self._obj["v"] * dv_dy

        self._obj["w"] = xr.DataArray(
            np.sqrt(accel_x**2 + accel_y**2), dims=["x", "y", "t"]
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
        return self._obj

    def Γ1(self, n, convCoords = True):
        """Makes use of Dask (kind of) to run Γ1_moving_window_function via Γ1_pad.
           It takes an Xarray dataset, applies rolling window to it, groups rolling windows
           and applyies custom Γ1-calculating function to it in a parallel manner.

        Args:
            self._obj (xr.Dataset) - must contain, at least, u, v, x, y and t
            n (int) - (2*n+1) gives the rolling window size
            convCoords (bool) - either True or False, convCoords = convert coordinates,
                                if True - create two new data arrays within self._obj with
                                the names "xCoordiantes" and "yCoordiantes" that store x and y
                                coordinates as data arrays; always keep it "True" unless you
                                have already created "xCoordiantes" and "yCoordiantes" somehow
                                (say, by running Γ1 or Γ2 functions before)

        Returns:
            self._obj (xr.Dataset) - the argument with the Γ1 data array
        """
        # Xarray rolling window (below) doesn't roll over the coordinates. We're going to convert
        # them to data arrays. Xarray does't make the conversion procedure easy. So, instead of
        # Xarray, we are going to adhere to numpy for the conversion.
        if convCoords:
            PMX, PMY = np.meshgrid(self._obj.coords['x'].to_numpy(), self._obj.coords['y'].to_numpy())
            tTimes = self._obj.coords['t'].to_numpy().size
            XYshape = PMX.T.shape + (tTimes,)
            self._obj['xCoordinates'] = xr.DataArray(np.broadcast_to(PMX.T[:,:,np.newaxis], XYshape), dims=['x','y','t'])
            self._obj['yCoordinates'] = xr.DataArray(np.broadcast_to(PMY.T[:,:,np.newaxis], XYshape), dims=['x','y','t'])

        # Create the object of class rolling:
        rollingW = self._obj.rolling({"x":(2*n+1), "y":(2*n+1), "t":1}, center=True)
        # Construct the dataset containing a new dimension corresponding to the rolling window
        fieldRoll = rollingW.construct(x='rollWx', y='rollWy', t='rollWt')
        # Xarray requires stacked array in case of a multidimensional rolling window
        fieldStacked = fieldRoll.stack(gridcell=['x','y','t'])

        # map_blocks is an automated Dask-parallel mapping function. It requires a 
        # special implementation. Thus, I have to create a separate function - Γ1_pad - 
        # which performs groupping of the stacked dataset fieldStacked. Then map_blocks
        # automaticly Dask-chunks Γpad returns. Every Dask-chunk can contain several groups.
        # The chunks are computed in parallel. See here for map_blocks() function:
        # https://tutorial.xarray.dev/advanced/map_blocks/simple_map_blocks.html
        def Γ1_pad(ds, n):
            dsGroup = ds.groupby("gridcell")
            return dsGroup.map(Γ1_moving_window_function, args=[n])
        
        newArr = fieldStacked.map_blocks(Γ1_pad, args=[n]).compute()   
        # Now, the result must be unstacked to return to the original x, y, t coordinates.
        self._obj['Γ1'] = newArr.unstack("gridcell")

        self._obj['Γ1'].attrs["standard_name"] = "Gamma 1"
        self._obj['Γ1'].attrs["units"] = "dimensionless"

        return self._obj
    
    def Γ2(self, n, convCoords = True):
        """Makes use of Dask (kind of) to run Γ2_moving_window_function via Γ2_pad.
           It takes an Xarray dataset, applies rolling window to it, groups rolling windows
           and applyies custom Γ2-calculating function to it in a parallel manner.

        Args:
            self._obj (xr.Dataset) - must contain, at least, u, v, x, y and t
            n (int) - (2*n+1) gives the rolling window size
            convCoords (bool) - either True or False, convCoords = convert coordinates,
                                if True - create two new data arrays within self._obj with
                                the names "xCoordiantes" and "yCoordiantes" that store x and y
                                coordinates as data arrays; always keep it "True" unless you
                                have already created "xCoordiantes" and "yCoordiantes" somehow
                                (say, by running Γ1 or Γ2 functions before)

        Returns:
            self._obj (xr.Dataset) - the argument with the Γ2 data array
        """
        # Xarray rolling window (below) doesn't roll over the coordinates. We're going to convert
        # them to data arrays. Xarray does't make the conversion procedure easy. So, instead of
        # Xarray, we are going to adhere to numpy for the conversion.
        if convCoords:
            PMX, PMY = np.meshgrid(self._obj.coords['x'].to_numpy(), self._obj.coords['y'].to_numpy())
            tTimes = self._obj.coords['t'].to_numpy().size
            XYshape = PMX.T.shape + (tTimes,)
            self._obj['xCoordinates'] = xr.DataArray(np.broadcast_to(PMX.T[:,:,np.newaxis], XYshape), dims=['x','y','t'])
            self._obj['yCoordinates'] = xr.DataArray(np.broadcast_to(PMY.T[:,:,np.newaxis], XYshape), dims=['x','y','t'])

        # Create the object of class rolling:
        rollingW = self._obj.rolling({"x":(2*n+1), "y":(2*n+1), "t":1}, center=True)
        # Construct the dataset containing a new dimension corresponding to the rolling window
        fieldRoll = rollingW.construct(x='rollWx', y='rollWy', t='rollWt')
        # Xarray requires stacked array in case of a multidimensional rolling window
        fieldStacked = fieldRoll.stack(gridcell=['x','y','t'])

        # map_blocks is an automated Dask-parallel mapping function. It requires a 
        # special implementation. Thus, I have to create a separate function - Γ2_pad - 
        # which performs groupping of the stacked dataset fieldStacked. Then map_blocks
        # automaticly Dask-chunks Γpad returns. Every Dask-chunk can contain several groups.
        # The chunks are computed in parallel. See here for map_blocks() function:
        # https://tutorial.xarray.dev/advanced/map_blocks/simple_map_blocks.html
        def Γ2_pad(ds, n):
            dsGroup = ds.groupby("gridcell")
            return dsGroup.map(Γ2_moving_window_function, args=[n])
        
        newArr = fieldStacked.map_blocks(Γ2_pad, args=[n]).compute()   
        # Now, the result must be unstacked to return to the original x, y, t coordinates.
        self._obj['Γ2'] = newArr.unstack("gridcell")

        self._obj['Γ2'].attrs["standard_name"] = "Gamma 2"
        self._obj['Γ2'].attrs["units"] = "dimensionless"

        return self._obj

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
        fig, ax = gquiver(self._obj, **kwargs)
        return fig, ax

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
