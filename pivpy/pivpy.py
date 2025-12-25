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
from pivpy.graphics import streamplot as gstreamplot
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
            
        Raises:
            ValueError: If crop_vector has wrong length or invalid bounds
            
        Example:
            >>> data = data.piv.crop([5, 15, -5, -15])  # Crop to x:[5,15], y:[-5,-15]
            >>> data = data.piv.crop([None, 20, None, None])  # Crop only xmax to 20
        """
        if crop_vector is None:
            crop_vector = 4 * [None]
        
        if len(crop_vector) != 4:
            raise ValueError(
                f"crop_vector must have 4 elements [xmin, xmax, ymin, ymax], "
                f"got {len(crop_vector)} elements"
            )

        xmin, xmax, ymin, ymax = crop_vector

        xmin = self._obj.x.min() if xmin is None else xmin
        xmax = self._obj.x.max() if xmax is None else xmax
        ymin = self._obj.y.min() if ymin is None else ymin
        ymax = self._obj.y.max() if ymax is None else ymax
        
        # Note: We don't validate xmin < xmax or ymin < ymax because coordinates
        # might be in reverse order (e.g., negative y-axis pointing down)

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

    def clip(
        self,
        min=None,
        max=None,
        *,
        by: str = None,
        keep_attrs: bool = True,
    ):
        """Clips values in the dataset based on specified thresholds
        
        This method limits values in the dataset to fall within [min, max] range.
        It can clip the entire dataset or filter based on specific variables (U, V, 
        or scalar properties like magnitude).
        
        Args:
            min (float or None, optional): Minimum value threshold. Values below this 
                will be masked/removed. If None, no lower clipping is performed. 
                Defaults to None.
            max (float or None, optional): Maximum value threshold. Values above this 
                will be masked/removed. If None, no upper clipping is performed. 
                Defaults to None.
            by (str or None, optional): Variable name to use for clipping criterion.
                Common values include 'u', 'v', or 'magnitude', but any scalar property 
                name in the dataset is valid (e.g., 'w' for vorticity, 'tke', etc.).
                If None, clips all variables independently. If 'magnitude', computes
                velocity magnitude and uses it for filtering. Defaults to None.
            keep_attrs (bool, optional): If True, attributes will be preserved. 
                Defaults to True.
                
        Returns:
            xr.Dataset: Dataset with clipped values. If 'by' is specified, returns
                dataset with locations that don't meet the criteria set to NaN.
                
        Raises:
            ValueError: If neither min nor max is provided
            ValueError: If 'by' variable doesn't exist in the dataset and isn't 'magnitude'
            
        Examples:
            >>> # Clip all variables to [-10, 10] range
            >>> data = data.piv.clip(min=-10, max=10)
            
            >>> # Filter based on U velocity component
            >>> data = data.piv.clip(min=-5, max=5, by='u')
            
            >>> # Filter based on velocity magnitude
            >>> data = data.piv.clip(max=10, by='magnitude')
            
            >>> # Filter based on vorticity (after computing it)
            >>> data = data.piv.vorticity(name='w')
            >>> data = data.piv.clip(min=-100, max=100, by='w')
        
        See Also:
            xarray.Dataset.clip : Similar method in xarray
            numpy.clip : Equivalent function in NumPy
        """
        if min is None and max is None:
            raise ValueError("At least one of 'min' or 'max' must be provided")
        
        if by is None:
            # Clip all variables independently using xarray's built-in clip
            return self._obj.clip(min=min, max=max, keep_attrs=keep_attrs)
        
        # Clip based on a specific variable
        if by == "magnitude":
            # Compute magnitude if not already in dataset
            criterion = np.sqrt(self._obj["u"] ** 2 + self._obj["v"] ** 2)
        else:
            # Use existing variable
            if by not in self._obj:
                raise ValueError(
                    f"Variable '{by}' not found in dataset. "
                    f"Available variables: {list(self._obj.data_vars)}"
                )
            criterion = self._obj[by]
        
        # Create mask based on criterion
        mask = xr.ones_like(criterion, dtype=bool)
        if min is not None:
            mask = mask & (criterion >= min)
        if max is not None:
            mask = mask & (criterion <= max)
        
        # Apply mask to all data variables (set non-matching locations to NaN)
        result = self._obj.copy()
        for var in result.data_vars:
            result[var] = result[var].where(mask)
        
        if not keep_attrs:
            result.attrs = {}
            for var in result.data_vars:
                result[var].attrs = {}
        
        return result

    def filterf(self, sigma: List[float]=[1.,1.,0], **kwargs):
        """Applies Gaussian filtering to velocity fields
        
        Args:
            sigma (List[float], optional): Standard deviation for Gaussian kernel 
                in [y, x, t] dimensions. Defaults to [1., 1., 0] (spatial filtering only).
            **kwargs: Additional keyword arguments passed to scipy.ndimage.gaussian_filter
            
        Returns:
            xr.Dataset: Filtered dataset with smoothed velocity fields
            
        Raises:
            ValueError: If sigma has wrong length or contains invalid values
            
        Example:
            >>> data = data.piv.filterf(sigma=[2., 2., 0])  # Smooth with sigma=2 in space
        """
        if len(sigma) != 3:
            raise ValueError(
                f"sigma must have 3 elements [sigma_y, sigma_x, sigma_t], "
                f"got {len(sigma)} elements"
            )
        
        if any(s < 0 for s in sigma):
            raise ValueError(f"All sigma values must be non-negative, got {sigma}")

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

    def vorticity(self, name: str = "w"):
        """Calculates vorticity of the data array (at one time instance) and
        adds it to the dataset
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.

        Input:
            xarray with the variables u,v and dimensions x,y

        Output:
            xarray with the estimated vorticity as a scalar field with
            same dimensions
            
        Example:
            >>> data.piv.vorticity()  # Creates data["w"] with vorticity
            >>> data.piv.vorticity(name="vort")  # Creates data["vort"] with vorticity

        """

        self._obj[name] = self._obj["v"].differentiate("x") - self._obj[
            "u"
        ].differentiate("y")

        self._obj[name].attrs["units"] = "1/delta_t"
        self._obj[name].attrs["standard_name"] = "vorticity"

        return self._obj

    def strain(self, name: str = "w"):
        """Calculates rate of strain of a two component field
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.

        Returns:
            xr.Dataset: Dataset with added scalar field = du_dx^2 + dv_dy^2 + 0.5*(du_dy+dv_dx)^2
            
        Example:
            >>> data.piv.strain()  # Creates data["w"] with strain
            >>> data.piv.strain(name="strain_rate")  # Creates data["strain_rate"]
        """
        du_dx = self._obj["u"].differentiate("x")
        du_dy = self._obj["u"].differentiate("y")
        dv_dx = self._obj["v"].differentiate("x")
        dv_dy = self._obj["v"].differentiate("y")

        self._obj[name] = du_dx**2 + dv_dy**2 + 0.5 * (du_dy + dv_dx) ** 2
        self._obj[name].attrs["units"] = "1/delta_t"
        self._obj[name].attrs["standard_name"] = "strain"

        return self._obj

    def divergence(self, name: str = "w"):
        """Calculates divergence field
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.

        Returns:
            xr.Dataset: Dataset with the new property [name] = divergence
            
        Example:
            >>> data.piv.divergence()  # Creates data["w"] with divergence
            >>> data.piv.divergence(name="div")  # Creates data["div"] with divergence
        """
        du_dx, _ = np.gradient(
            self._obj["u"], self._obj["x"], self._obj["y"], axis=(0, 1)
        )
        _, dv_dy = np.gradient(
            self._obj["v"], self._obj["x"], self._obj["y"], axis=(0, 1)
        )

        if "t" in self._obj.coords:
            self._obj[name] = (("x", "y", "t"), dv_dy + du_dx)
        else:
            self._obj[name] = (("x", "y"), dv_dy + du_dx)

        self._obj[name].attrs["units"] = "1/delta_t"
        self._obj[name].attrs["standard_name"] = "divergence"

        return self._obj

    def acceleration(self, name: str = "w"):
        """Calculates material derivative or acceleration of the
        data array (single frame)
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.

        Input:
            xarray with the variables u,v and dimensions x,y

        Output:
            xarray with the estimated acceleration as a scalar field data[name]
            
        Example:
            >>> data.piv.acceleration()  # Creates data["w"] with acceleration
            >>> data.piv.acceleration(name="accel")  # Creates data["accel"]

        """
        du_dx = self._obj["u"].differentiate("x")
        du_dy = self._obj["u"].differentiate("y")
        dv_dx = self._obj["v"].differentiate("x")
        dv_dy = self._obj["v"].differentiate("y")

        accel_x = self._obj["u"] * du_dx + self._obj["v"] * du_dy
        accel_y = self._obj["u"] * dv_dx + self._obj["v"] * dv_dy

        self._obj[name] = xr.DataArray(
            np.sqrt(accel_x**2 + accel_y**2), dims=["x", "y", "t"]
        )

        self._obj[name].attrs["units"] = "1/delta_t"
        self._obj[name].attrs["standard_name"] = "acceleration"

        return self._obj

    def kinetic_energy(self, name: str = "w"):
        """Estimates kinetic energy
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xr.Dataset: Dataset with kinetic energy field
            
        Example:
            >>> data.piv.kinetic_energy()  # Creates data["w"] with KE
            >>> data.piv.kinetic_energy(name="ke")  # Creates data["ke"]
        """
        self._obj[name] = self._obj["u"] ** 2 + self._obj["v"] ** 2
        self._obj[name].attrs["units"] = "(m/s)^2"
        self._obj[name].attrs["standard_name"] = "kinetic_energy"
        return self._obj

    def tke(self, name: str = "w"):
        """Estimates turbulent kinetic energy
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xr.Dataset: New dataset with TKE field (based on fluctuations from mean)
            
        Raises:
            ValueError: If dataset has less than 2 time frames
            
        Example:
            >>> data.piv.tke()  # Creates data["w"] with TKE
            >>> data.piv.tke(name="tke")  # Creates data["tke"]
        """
        if len(self._obj.t) < 2:
            raise ValueError(
                "TKE is not defined for a single vector field, \
                              use .piv.kinetic_energy()"
            )

        new_obj = self._obj.copy()
        new_obj -= new_obj.mean(dim="t")
        new_obj[name] = new_obj["u"] ** 2 + new_obj["v"] ** 2
        new_obj[name].attrs["units"] = "(m/s)^2"
        new_obj[name].attrs["standard_name"] = "TKE"

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

    def reynolds_stress(self, name: str = "w"):
        """Calculates Reynolds stress from velocity fluctuations
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xr.Dataset: Dataset with Reynolds stress field (-<u'v'>)
            
        Raises:
            ValueError: If dataset has less than 2 time frames
            
        Example:
            >>> data.piv.reynolds_stress()  # Creates data["w"] with Reynolds stress
            >>> data.piv.reynolds_stress(name="rey_stress")  # Creates data["rey_stress"]
        """

        if len(self._obj.t) < 2:
            raise ValueError(
                "fluctuations cannot be defined for a \
                              single vector field, use .piv.ke()"
            )

        new_obj = self._obj.copy()
        new_obj -= new_obj.mean(dim="t")

        new_obj[name] = -1 * new_obj["u"] * new_obj["v"]  # new scalar
        self._obj[name] = new_obj[name].mean(dim="t")  # reynolds stress is -\rho < u' v'>
        self._obj[name].attrs["standard_name"] = "Reynolds_stress"

        return self._obj

    def rms(self, name: str = "w"):
        """Root mean square of velocity fluctuations
        
        Args:
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xr.Dataset: Dataset with RMS field (sqrt of TKE)
            
        Example:
            >>> data.piv.rms()  # Creates data["w"] with RMS
            >>> data.piv.rms(name="rms")  # Creates data["rms"]
        """
        self._obj = self.tke(name=name)
        self._obj[name] = np.sqrt(self._obj[name])
        self._obj[name].attrs["standard_name"] = "rms"
        self._obj[name].attrs["units"] = "m/s"
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

    def vec2scal(self, flow_property: str = "curl", name: str = "w"):
        """Creates a scalar flow property field from velocity data
        
        Args:
            flow_property (str, optional): Name of the flow property to compute.
                Valid options: 'curl'/'vorticity'/'vort', 'ke'/'ken'/'kinetic_energy',
                'strain', 'divergence', 'acceleration', 'tke', 'reynolds_stress', 'rms'.
                Defaults to "curl".
            name (str, optional): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xr.Dataset: Dataset with computed scalar field
            
        Raises:
            AttributeError: If the specified flow property method doesn't exist
            
        Example:
            >>> data = data.piv.vec2scal('vorticity')  # Compute vorticity in data["w"]
            >>> data = data.piv.vec2scal('ke', name='ke')  # Compute KE in data["ke"]
            >>> # Store multiple scalars in one dataset:
            >>> data = data.piv.vec2scal('vorticity', name='vort')
            >>> data = data.piv.vec2scal('tke', name='tke')
            >>> data = data.piv.vec2scal('reynolds_stress', name='rey_stress')
        """
        # Replace common aliases with canonical names
        flow_property = "vorticity" if flow_property in ["curl", "vort"] else flow_property
        flow_property = "kinetic_energy" if flow_property in ["ken", "ke"] else flow_property
        
        # Check if method exists
        if not hasattr(self, flow_property):
            valid_properties = [
                'vorticity', 'kinetic_energy', 'strain', 'divergence', 
                'acceleration', 'tke', 'reynolds_stress', 'rms'
            ]
            raise AttributeError(
                f"Unknown flow property '{flow_property}'. "
                f"Valid options are: {', '.join(valid_properties)}"
            )

        method = getattr(self, flow_property)
        self._obj = method(name=name)

        return self._obj

    def __mul__(self, scalar):
        """Multiplies velocity field by a scalar (simple scaling)
        
        Args:
            scalar (float): Scaling factor
            
        Returns:
            xr.Dataset: Scaled dataset
            
        Example:
            >>> scaled_data = data.piv * 2.0  # Double all velocities
        """
        self._obj["u"] *= scalar
        self._obj["v"] *= scalar
        if "w" in self._obj.var():
            self._obj["w"] *= scalar  # Fixed: should be multiply, not add

        return self._obj

    def __div__(self, scalar):
        """Divides velocity field by a scalar
        
        Args:
            scalar (float): Division factor
            
        Returns:
            xr.Dataset: Scaled dataset
            
        Raises:
            ValueError: If scalar is zero
            
        Example:
            >>> normalized_data = data.piv / 100.0  # Normalize velocities
        """
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
            
        self._obj["u"] /= scalar
        self._obj["v"] /= scalar

        return self._obj

    def set_delta_t(self, delta_t: float = 0.0):
        """Sets the time interval attribute for PIV measurements
        
        Args:
            delta_t (float, optional): Time interval between frame A and B. Defaults to 0.0.
            
        Returns:
            xr.Dataset: Dataset with updated delta_t attribute
            
        Raises:
            ValueError: If delta_t is negative
            
        Example:
            >>> data = data.piv.set_delta_t(0.001)  # Set dt to 1 millisecond
        """
        if delta_t < 0:
            raise ValueError(f"delta_t must be non-negative, got {delta_t}")
            
        self._obj.attrs["delta_t"] = delta_t
        return self._obj

    def set_scale(self, scale: float = 1.0):
        """Scales all spatial coordinates and velocities by a factor
        
        Args:
            scale (float, optional): Scaling factor. Defaults to 1.0.
            
        Returns:
            xr.Dataset: Dataset with scaled coordinates and velocities
            
        Raises:
            ValueError: If scale is zero or negative
            
        Example:
            >>> data = data.piv.set_scale(0.001)  # Convert from pixels to mm if 1 pix = 0.001 mm
        """
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
            
        for var in ["x", "y", "u", "v"]:
            self._obj[var] = self._obj[var] * scale

        return self._obj

    def rotate(self, theta: float = 0.0):
        """Rotates the coordinate system and velocity field
        
        Args:
            theta (float, optional): Rotation angle in degrees (clockwise). Defaults to 0.0.
            
        Returns:
            xr.Dataset: Rotated dataset
            
        Note:
            This method works best for cases with equal grid spacing in x and y directions.
            The rotation is performed in-place on coordinates and velocity components.
            
        Example:
            >>> data = data.piv.rotate(45.0)  # Rotate by 45 degrees clockwise
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
        """graphics.streamplot() as a flow_property"""
        fig, ax = gstreamplot(self._obj, **kwargs)
        return fig, ax

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
