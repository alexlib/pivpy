# -*- coding: utf-8 -*-
"""
This script extends the functionality of xarray.Dataset by adding a new accessor called piv. The accessor adds several properties and methods that are useful for working with particle image velocimetry (PIV) data. The properties include average, which returns the mean flow field, and delta_t, which returns the time step used in the PIV measurement. The methods include crop, which allows the user to crop the data by a given number of rows and columns from the boundaries, vec2scal, which converts vector data to scalar data, pan, which pans the data by a given number of pixels, and rotate, which rotates the data by a given angle.


@author: Ron, Alex
"""
try:
    from typing_extensions import Literal
except ImportError:
    from typing import Literal
from typing import List, Optional

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from pivpy.graphics import quiver as gquiver
from pivpy.graphics import showf as gshowf
from pivpy.graphics import showscal as gshowscal
from pivpy.graphics import streamplot as gstreamplot
from pivpy.graphics import autocorrelation_plot as gautocorrelation_plot
from pivpy.graphics import histscal_disp as ghistscal_disp
from pivpy.graphics import histvec_disp as ghistvec_disp
from pivpy.graphics import to_movie as gto_movie
from pivpy.graphics import jpdfscal_disp as gjpdfscal_disp
from pivpy.compute_funcs import (
    Γ1_moving_window_function,
    Γ2_moving_window_function,
    bwfilter2d,
    corrf,
    corrm,
    gradientf,
    histf,
    filter2d,
    filter2d_kernel,
    interpolat_zeros_2d,
    interpf as cinterpf,
    jpdfscal as cjpdfscal,
    probef as cprobef,
    probeaverf as cprobeaverf,
    spatiotempf as cspatiotempf,
    tempcorrf as ctempcorrf,
)

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
            crop_vector (list): List of [xmin, xmax, ymin, ymax] values 
                to define cropping boundaries. Use None for any value to keep 
                the original boundary. Defaults to None (no cropping).
                
        Returns:
            xarray.Dataset: Cropped dataset
            
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

    def extractf(
        self,
        rect,
        opt: str = "phys",
        *,
        return_rect: bool = False,
    ):
        """Extract a rectangular area from the dataset (PIVMAT-inspired).

        Parameters
        ----------
        rect:
            Rectangle as ``[x1, y1, x2, y2]``.

            If ``opt='phys'`` (default), coordinates are in physical units and the
            selection is expanded to the nearest grid points (start behaves like a
            floor, end behaves like a ceil) before clamping.

            If ``opt='mesh'``, coordinates are mesh indices (1-based, inclusive,
            MATLAB-like) before clamping.
        opt:
            'phys' (default) or 'mesh'.
        return_rect:
            If True, also returns the effective rectangle in mesh indices
            ``[ix1, iy1, ix2, iy2]`` (1-based, inclusive) after clamping.

        Returns
        -------
        xarray.Dataset
            Extracted dataset.

        tuple
            If ``return_rect=True``, returns ``(dataset, rect_mesh)`` where
            ``rect_mesh`` is ``[ix1, iy1, ix2, iy2]`` (1-based, inclusive).

        Notes
        -----
        Interactive rectangle selection (PIVMAT's 'draw') is not supported.
        """

        ds = self._obj
        if rect is None:
            raise ValueError("extractf requires rect=[x1, y1, x2, y2]")
        if isinstance(rect, str) and rect.lower().startswith("draw"):
            raise NotImplementedError("Interactive rectangle selection is not supported; pass rect explicitly.")

        if not hasattr(rect, "__len__") or len(rect) != 4:
            raise ValueError("rect must be a sequence of 4 values: [x1, y1, x2, y2]")

        x1, y1, x2, y2 = rect

        if "x" not in ds.dims or "y" not in ds.dims:
            raise ValueError("extractf requires dataset dims 'x' and 'y'")

        def _bounds_from_phys(coord_vals: np.ndarray, a: float, b: float) -> tuple[int, int]:
            vals = np.asarray(coord_vals, dtype=float)
            n = int(vals.shape[0])
            if n == 0:
                return 0, -1
            if n == 1:
                return 0, 0

            lo = float(min(a, b))
            hi = float(max(a, b))

            reversed_axis = bool(vals[1] < vals[0])
            sorted_vals = vals[::-1] if reversed_axis else vals

            i1 = int(np.searchsorted(sorted_vals, lo, side="right") - 1)
            i2 = int(np.searchsorted(sorted_vals, hi, side="left"))

            if i1 < 0:
                i1 = 0
            if i2 < 0:
                i2 = 0
            if i1 > n - 1:
                i1 = n - 1
            if i2 > n - 1:
                i2 = n - 1

            if reversed_axis:
                start = (n - 1) - i2
                stop = (n - 1) - i1
            else:
                start = i1
                stop = i2

            if start > stop:
                # Degenerate selection: choose nearest index.
                target = 0.5 * (lo + hi)
                nearest = int(np.argmin(np.abs(vals - target)))
                return nearest, nearest

            return int(start), int(stop)

        def _bounds_from_mesh(n: int, a: float, b: float) -> tuple[int, int]:
            if n <= 0:
                return 0, -1
            lo = int(np.floor(min(a, b))) - 1
            hi = int(np.ceil(max(a, b))) - 1
            lo = max(lo, 0)
            hi = min(hi, n - 1)
            if lo > hi:
                lo = hi
            return lo, hi

        opt_l = str(opt).lower()
        if opt_l.startswith("phys"):
            xs, xe = _bounds_from_phys(ds["x"].values, float(x1), float(x2))
            ys, ye = _bounds_from_phys(ds["y"].values, float(y1), float(y2))
        elif opt_l.startswith("mesh"):
            xs, xe = _bounds_from_mesh(int(ds.sizes["x"]), float(x1), float(x2))
            ys, ye = _bounds_from_mesh(int(ds.sizes["y"]), float(y1), float(y2))
        else:
            raise ValueError("opt must be 'phys' or 'mesh'")

        out = ds.isel(x=slice(xs, xe + 1), y=slice(ys, ye + 1))
        out.attrs = dict(ds.attrs)
        self._obj = out

        mesh_rect = [xs + 1, ys + 1, xe + 1, ye + 1]
        if return_rect:
            return out, mesh_rect
        return out

    def pan(self, shift_x=0.0, shift_y=0.0):
        """Shifts the coordinate system by specified amounts
        
        Args:
            shift_x (float): Amount to shift in x direction. Defaults to 0.0.
            shift_y (float): Amount to shift in y direction. Defaults to 0.0.
            
        Returns:
            xarray.Dataset: Dataset with shifted coordinates
            
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
            min (float or None): Minimum value threshold. Values below this 
                will be masked/removed. If None, no lower clipping is performed. 
                Defaults to None.
            max (float or None): Maximum value threshold. Values above this 
                will be masked/removed. If None, no upper clipping is performed. 
                Defaults to None.
            by (str or None): Variable name to use for clipping criterion.
                Common values include 'u', 'v', or 'magnitude', but any scalar property 
                name in the dataset is valid (e.g., 'w' for vorticity, 'tke', etc.).
                If None, clips all variables independently. If 'magnitude', computes
                velocity magnitude and uses it for filtering. Defaults to None.
            keep_attrs (bool): If True, attributes will be preserved. 
                Defaults to True.
                
        Returns:
            xarray.Dataset: Dataset with clipped values. If 'by' is specified, returns
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

    def filterf(self, sigma: List[float] | float = [1.0, 1.0, 0.0], method: str = "gauss", *opts: str, **kwargs):
        """Apply a spatial filter to a vector/scalar field (PIVMAT-inspired).

        This method supports two calling conventions:

        1) Legacy PIVPy Gaussian smoothing (kept for backward compatibility)::

             ds = ds.piv.filterf([sigma_y, sigma_x, sigma_t], **gaussian_kwargs)

                     This uses SciPy's ``gaussian_filter`` on ``u`` and ``v``.

        2) PIVMAT-style normalized 2D convolution (NaN-aware)::

             ds = ds.piv.filterf(filtsize, method, 'same')
             ds = ds.piv.filterf(filtsize, method)         # default is 'valid'

           where ``method`` is one of: ``'gauss'`` (default), ``'flat'``, ``'igauss'``.
           The option ``'same'`` keeps the original shape; otherwise the result is
           smaller (Matlab ``conv2(...,'valid')`` behavior) and the x/y coordinates are
           truncated accordingly.
        """

        # --- Legacy path: sigma is a 3-vector
        if isinstance(sigma, (list, tuple, np.ndarray)):
            sigma_list = list(sigma)
            if len(sigma_list) != 3:
                raise ValueError(
                    f"sigma must have 3 elements [sigma_y, sigma_x, sigma_t], got {len(sigma_list)} elements"
                )
            if any(float(s) < 0 for s in sigma_list):
                raise ValueError(f"All sigma values must be non-negative, got {sigma_list}")

            self._obj["u"] = xr.DataArray(
                gaussian_filter(self._obj["u"].values, sigma_list, **kwargs),
                dims=("y", "x", "t"),
                attrs=self._obj["u"].attrs,
            )
            self._obj["v"] = xr.DataArray(
                gaussian_filter(self._obj["v"].values, sigma_list, **kwargs),
                dims=("y", "x", "t"),
                attrs=self._obj["v"].attrs,
            )
            return self._obj

        # --- PIVMAT-style path: sigma is actually filtsize (float)
        if kwargs:
            raise TypeError(
                "PIVMAT-style filterf(filtsize, ...) does not accept **kwargs. "
                "Pass a 3-element sigma list for gaussian_filter kwargs."
            )

        ds = self._obj
        if "x" not in ds.dims or "y" not in ds.dims:
            raise ValueError("filterf requires spatial dims 'y' and 'x'")

        fs = float(sigma)
        if fs == 0.0:
            return ds

        mode = "valid"
        for opt in opts:
            o = str(opt).lower()
            if o.startswith("same"):
                mode = "same"
            elif o.startswith("valid"):
                mode = "valid"
            elif o == "":
                continue
            else:
                raise ValueError(f"Unknown filterf option: {opt!r}")

        k = filter2d_kernel(fs, method)
        ky, kx = (int(k.shape[0]), int(k.shape[1]))
        ny = int(ds.sizes["y"])
        nx = int(ds.sizes["x"])
        if mode == "same":
            ny_out, nx_out = ny, nx
            base = ds
        else:
            ny_out = ny - ky + 1
            nx_out = nx - kx + 1
            if ny_out <= 0 or nx_out <= 0:
                raise ValueError("filter kernel larger than input")

            ly = (ky - 1) // 2
            ry = (ky - 1) - ly
            lx = (kx - 1) // 2
            rx = (kx - 1) - lx
            base = ds.isel(y=slice(ly, ny - ry), x=slice(lx, nx - rx))

        def _core(a2: np.ndarray) -> np.ndarray:
            return filter2d(a2, fs, method, mode=mode)

        out = base.copy(deep=True)
        for name in ("u", "v"):
            if name not in out.data_vars:
                continue
            da_in = ds[name]
            da_out_template = out[name]
            if "y" not in da_in.dims or "x" not in da_in.dims:
                continue
            filtered = xr.apply_ufunc(
                _core,
                da_in,
                input_core_dims=[["y", "x"]],
                output_core_dims=[["y", "x"]],
                exclude_dims={"y", "x"},
                vectorize=True,
                dask="parallelized",
                dask_gufunc_kwargs={"output_sizes": {"y": ny_out, "x": nx_out}},
                output_dtypes=[float],
            )
            # Preserve original dim order (typically ('y','x','t')).
            filtered = filtered.transpose(*da_in.dims)
            # Attach the truncated coordinates from the base dataset.
            filtered = filtered.assign_coords({"y": base["y"], "x": base["x"]})
            out[name] = filtered
            out[name].attrs = dict(ds[name].attrs)

        out.attrs = dict(ds.attrs)
        self._obj = out
        return out

    def bwfilterf(
        self,
        filtsize: float = 3.0,
        order: float = 8.0,
        *,
        mode: Literal["low", "high"] = "low",
        trunc: bool = False,
        var: Optional[str] = None,
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """Butterworth spatial filter for vector/scalar fields (PIVMAT-inspired).

        Applies a low-pass (default) or high-pass Butterworth filter in Fourier space
        along the spatial dimensions (y, x). Implemented via fast NumPy FFT inside
        `xarray.apply_ufunc`, vectorized over any remaining dimensions (e.g. t).

        Parameters
        ----------
        filtsize:
            Cutoff size in grid units. If 0, returns the dataset unchanged.
        order:
            Filter order (typical range 2..10). Larger means sharper cutoff.
        mode:
            'low' or 'high'. High-pass is implemented by flipping the sign of order.
        trunc:
            If True, truncates borders of width floor(filtsize) after filtering.
        var:
            Scalar variable name to filter. If None, defaults to vector mode (u, v)
            unless `variables` is provided.
        variables:
            Explicit list of variables to filter.
        """

        fs = float(filtsize)
        if fs == 0.0:
            return self._obj

        ds = self._obj
        if "x" not in ds.dims or "y" not in ds.dims:
            raise ValueError("bwfilterf requires spatial dims 'y' and 'x'")

        ord_eff = float(order)
        if str(mode).lower().startswith("high"):
            ord_eff = -abs(ord_eff)
        else:
            ord_eff = abs(ord_eff)

        # PIVMAT behavior: enforce even spatial sizes by dropping last row/col.
        out = ds
        if int(out.sizes["x"]) % 2 == 1:
            out = out.isel(x=slice(0, -1))
        if int(out.sizes["y"]) % 2 == 1:
            out = out.isel(y=slice(0, -1))

        if variables is None:
            if var is not None:
                variables = [var]
            else:
                variables = [v for v in ("u", "v") if v in out.data_vars]
                if not variables:
                    raise ValueError("bwfilterf: no variables to filter (expected 'u'/'v' or var=...)")
        else:
            variables = list(variables)
            for v in variables:
                if v not in out.data_vars:
                    raise ValueError(f"Variable '{v}' not found in dataset")

        def _bw_core(a2: np.ndarray) -> np.ndarray:
            return bwfilter2d(a2, fs, ord_eff)

        out2 = out.copy(deep=True)
        for name in variables:
            da = out2[name]
            if "y" not in da.dims or "x" not in da.dims:
                continue
            out2[name] = xr.apply_ufunc(
                _bw_core,
                da,
                input_core_dims=[["y", "x"]],
                output_core_dims=[["y", "x"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            out2[name].attrs = dict(out[name].attrs)

        if trunc:
            ntr = int(np.floor(fs))
            if ntr > 0:
                ny = int(out2.sizes["y"])
                nx = int(out2.sizes["x"])
                if 2 * ntr >= ny or 2 * ntr >= nx:
                    raise ValueError("truncation too large for field size")
                out2 = out2.isel(y=slice(ntr, -ntr), x=slice(ntr, -ntr))

        out2.attrs = dict(out.attrs)
        self._obj = out2
        return out2

    def bwfilterf_pm(
        self,
        filtsize: float,
        order: float,
        *opts: str,
        var: Optional[str] = None,
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """PIVMAT-compatible wrapper for :meth:`bwfilterf`.

        Accepts option strings like PIVMAT:
        - 'low' (default)
        - 'high'
        - 'trunc'

        Examples
        --------
        - ``ds.piv.bwfilterf_pm(3, 8)``
        - ``ds.piv.bwfilterf_pm(3, 8, 'high')``
        - ``ds.piv.bwfilterf_pm(3, 8, 'high', 'trunc')``
        """

        mode: Literal["low", "high"] = "low"
        trunc = False
        for opt in opts:
            o = str(opt).lower()
            if o.startswith("high"):
                mode = "high"
            elif o.startswith("low"):
                mode = "low"
            elif o.startswith("trunc"):
                trunc = True
            elif o == "":
                continue
            else:
                raise ValueError(f"Unknown bwfilterf option: {opt!r}")

        return self.bwfilterf(
            filtsize=float(filtsize),
            order=float(order),
            mode=mode,
            trunc=trunc,
            var=var,
            variables=variables,
        )

    def flipf(self, dir: str = "x") -> xr.Dataset:
        """Flip vector/scalar fields about vertical or horizontal axis (PIVMAT-inspired).

        Parameters
        ----------
        dir:
            Direction to flip:

            - 'x': left-right mirror (flip along x; negate ``u``)
            - 'y': top-bottom mirror (flip along y; negate ``v``)
            - 'xy' or 'yx': both flips
            - '': do nothing

        Notes
        -----
        The x/y coordinate values are left unchanged (only the data are mirrored),
        matching PIVMAT behavior.
        """

        ds = self._obj
        d = str(dir)
        dl = d.lower()
        if dl in ("", "none"):
            return ds

        if dl not in ("x", "y", "xy", "yx"):
            raise ValueError("dir must be one of: 'x', 'y', 'xy', 'yx', ''")

        flip_x = "x" in dl
        flip_y = "y" in dl

        if (flip_x and "x" not in ds.dims) or (flip_y and "y" not in ds.dims):
            raise ValueError("flipf requires spatial dims 'x' and 'y'")

        out = ds.copy(deep=True)
        if flip_x:
            rev_x = np.arange(int(ds.sizes["x"]) - 1, -1, -1)
        if flip_y:
            rev_y = np.arange(int(ds.sizes["y"]) - 1, -1, -1)

        for name, da in ds.data_vars.items():
            flipped = da
            if flip_x and "x" in da.dims:
                flipped = flipped.isel(x=rev_x).assign_coords(x=ds["x"])
            if flip_y and "y" in da.dims:
                flipped = flipped.isel(y=rev_y).assign_coords(y=ds["y"])

            # Velocity sign convention: u is x-component, v is y-component.
            if flip_x and name in ("u", "vx"):
                flipped = -flipped
            if flip_y and name in ("v", "vy"):
                flipped = -flipped

            flipped.attrs = dict(da.attrs)
            out[name] = flipped

        out.attrs = dict(ds.attrs)
        self._obj = out
        return out

    def addnoisef(
        self,
        eps: float = 0.1,
        opt: Literal["add", "mul"] = "add",
        nc: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Adds normally-distributed white noise to velocity fields.

        This method is inspired by PIVMat's `addnoisef`.

        Args:
            eps: Noise level. For additive mode, noise std is `eps * std(u,v)`.
                 For multiplicative mode, velocity is multiplied by `(1 + eps * noise)`.
            opt: 'add' for additive noise or 'mul' for multiplicative noise.
            nc: Optional Gaussian smoothing length scale for the noise, in the same
                units as the dataset coordinates ("mesh units"). If 0, no smoothing.
            seed: Optional RNG seed for reproducibility.

        Returns:
            xarray.Dataset: Dataset with noisy u/v.
        """

        if eps is None or float(eps) == 0.0:
            return self._obj

        opt_l = str(opt).lower()
        if not (opt_l.startswith("add") or opt_l.startswith("mul")):
            raise ValueError("opt must be 'add' or 'mul'")

        if "u" not in self._obj or "v" not in self._obj:
            raise ValueError("Dataset must contain 'u' and 'v' variables")

        u0 = np.asarray(self._obj["u"].values)
        v0 = np.asarray(self._obj["v"].values)
        if u0.ndim != 3 or v0.ndim != 3:
            raise ValueError("Expected 'u' and 'v' to have dims ('y','x','t')")

        # Use a single reference scale for both components.
        ref_std = float(np.nanstd(np.stack([u0, v0], axis=0)))
        if not np.isfinite(ref_std) or ref_std == 0.0:
            ref_std = 1.0

        rng = np.random.default_rng(seed)

        # Convert smoothing length (in coordinate units) to gaussian sigma (in grid points).
        sigma_yx = None
        if nc is not None and float(nc) != 0.0:
            try:
                x = np.asarray(self._obj.coords["x"].values, dtype=float)
                y = np.asarray(self._obj.coords["y"].values, dtype=float)
                dx = float(np.nanmedian(np.diff(x))) if x.size >= 2 else 1.0
                dy = float(np.nanmedian(np.diff(y))) if y.size >= 2 else 1.0
                dx = abs(dx) if dx != 0 else 1.0
                dy = abs(dy) if dy != 0 else 1.0
                sigma_yx = (abs(float(nc)) / dy, abs(float(nc)) / dx)
            except Exception:
                sigma_yx = (abs(float(nc)), abs(float(nc)))

        u = u0.copy()
        v = v0.copy()

        for ti in range(u.shape[2]):
            nu = rng.standard_normal(size=u.shape[:2])
            nv = rng.standard_normal(size=v.shape[:2])

            if sigma_yx is not None:
                nu = gaussian_filter(nu, sigma=sigma_yx, mode="nearest")
                nv = gaussian_filter(nv, sigma=sigma_yx, mode="nearest")
                # Keep noise level roughly independent of smoothing.
                s_nu = float(np.nanstd(nu))
                s_nv = float(np.nanstd(nv))
                if s_nu > 0:
                    nu = nu / s_nu
                if s_nv > 0:
                    nv = nv / s_nv

            if opt_l.startswith("add"):
                u[:, :, ti] = u[:, :, ti] + nu * float(eps) * ref_std
                v[:, :, ti] = v[:, :, ti] + nv * float(eps) * ref_std
            else:
                u[:, :, ti] = u[:, :, ti] * (1.0 + nu * float(eps))
                v[:, :, ti] = v[:, :, ti] * (1.0 + nv * float(eps))

        self._obj["u"].values[...] = u
        self._obj["v"].values[...] = v
        return self._obj

    def averf(
        self,
        opt: str = "",
        *,
        return_std_rms: bool = False,
    ):
        """Average (and optionally std/rms) of vector/scalar fields over time.

        This method is inspired by PIVMat's `averf`.

        By default, zero elements are treated as invalid and are excluded from
        the computations. To include zeros, pass opt containing '0'.

        Args:
            opt: Option string. If it contains '0', zeros are included.
            return_std_rms: If True, also returns (std, rms) as scalar fields.

        Returns:
            xarray.Dataset: Averaged dataset (single-frame, t=0) if ``return_std_rms=False``.
            tuple: ``(avg, std, rms)`` if ``return_std_rms=True``.
        """

        include_zeros = "0" in str(opt)

        ds = self._obj
        if "t" not in ds.dims:
            raise ValueError("averf requires a time dimension 't'")

        def _mean_excluding_zeros(arr: np.ndarray) -> np.ndarray:
            # arr is (y, x, t)
            valid = np.isfinite(arr)
            if not include_zeros:
                valid = valid & (arr != 0)
            out = np.zeros(arr.shape[:2], dtype=float)
            if include_zeros:
                out = np.nanmean(arr, axis=2)
                out = np.nan_to_num(out, nan=0.0)
                return out
            denom = valid.sum(axis=2)
            num = np.where(valid, arr, 0.0).sum(axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = num / denom
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out

        # Vector field mode if u/v exist, else scalar mode if w exists.
        if "u" in ds and "v" in ds:
            u0 = np.asarray(ds["u"].values, dtype=float)
            v0 = np.asarray(ds["v"].values, dtype=float)
            if u0.ndim != 3 or v0.ndim != 3:
                raise ValueError("Expected 'u' and 'v' to have dims ('y','x','t')")

            u_mean = _mean_excluding_zeros(u0)
            v_mean = _mean_excluding_zeros(v0)

            avg = ds.isel(t=[0]).copy(deep=True)
            avg = avg.drop_vars([v for v in avg.data_vars if v not in {"u", "v", "chc", "mask"}], errors="ignore")
            avg["u"] = xr.DataArray(u_mean[:, :, None], dims=("y", "x", "t"), attrs=ds["u"].attrs)
            avg["v"] = xr.DataArray(v_mean[:, :, None], dims=("y", "x", "t"), attrs=ds["v"].attrs)
            avg = avg.assign_coords(t=np.asarray([0.0], dtype=float))
            avg.attrs = dict(ds.attrs)

            if not return_std_rms:
                return avg

            # STD and RMS as scalar fields (combined magnitude) like PIVMat.
            finite = np.isfinite(u0) & np.isfinite(v0)
            if include_zeros:
                valid = finite
            else:
                valid = finite & (u0 != 0) & (v0 != 0)

            denom = valid.sum(axis=2).astype(float)
            denom_safe = np.where(denom == 0, np.nan, denom)

            du = u0 - u_mean[:, :, None]
            dv = v0 - v_mean[:, :, None]

            std_sq = np.where(valid, du * du + dv * dv, 0.0).sum(axis=2) / denom_safe
            rms_sq = np.where(valid, u0 * u0 + v0 * v0, 0.0).sum(axis=2) / denom_safe
            std_field = np.sqrt(std_sq)
            rms_field = np.sqrt(rms_sq)
            std_field = np.nan_to_num(std_field, nan=0.0)
            rms_field = np.nan_to_num(rms_field, nan=0.0)

            std_ds = ds.isel(t=[0]).copy(deep=True)
            rms_ds = ds.isel(t=[0]).copy(deep=True)
            std_ds = std_ds.drop_vars(list(std_ds.data_vars), errors="ignore")
            rms_ds = rms_ds.drop_vars(list(rms_ds.data_vars), errors="ignore")

            units = ds["u"].attrs.get("units", "")
            std_ds["w"] = xr.DataArray(std_field[:, :, None], dims=("y", "x", "t"), attrs={"standard_name": "standard_deviation", "units": units})
            rms_ds["w"] = xr.DataArray(rms_field[:, :, None], dims=("y", "x", "t"), attrs={"standard_name": "root_mean_square", "units": units})
            std_ds = std_ds.assign_coords(t=np.asarray([0.0], dtype=float))
            rms_ds = rms_ds.assign_coords(t=np.asarray([0.0], dtype=float))
            std_ds.attrs = dict(ds.attrs)
            rms_ds.attrs = dict(ds.attrs)
            return avg, std_ds, rms_ds

        if "w" in ds:
            w0 = np.asarray(ds["w"].values, dtype=float)
            if w0.ndim != 3:
                raise ValueError("Expected 'w' to have dims ('y','x','t')")
            w_mean = _mean_excluding_zeros(w0)
            avg = ds.isel(t=[0]).copy(deep=True)
            avg = avg.drop_vars([v for v in avg.data_vars if v != "w"], errors="ignore")
            avg["w"] = xr.DataArray(w_mean[:, :, None], dims=("y", "x", "t"), attrs=ds["w"].attrs)
            avg = avg.assign_coords(t=np.asarray([0.0], dtype=float))
            avg.attrs = dict(ds.attrs)
            if not return_std_rms:
                return avg

            finite = np.isfinite(w0)
            valid = finite if include_zeros else (finite & (w0 != 0))
            denom = valid.sum(axis=2).astype(float)
            denom_safe = np.where(denom == 0, np.nan, denom)
            dw = w0 - w_mean[:, :, None]
            std_sq = np.where(valid, dw * dw, 0.0).sum(axis=2) / denom_safe
            rms_sq = np.where(valid, w0 * w0, 0.0).sum(axis=2) / denom_safe
            std_field = np.nan_to_num(np.sqrt(std_sq), nan=0.0)
            rms_field = np.nan_to_num(np.sqrt(rms_sq), nan=0.0)

            units = ds["w"].attrs.get("units", "")
            std_ds = ds.isel(t=[0]).copy(deep=True)
            rms_ds = ds.isel(t=[0]).copy(deep=True)
            std_ds = std_ds.drop_vars(list(std_ds.data_vars), errors="ignore")
            rms_ds = rms_ds.drop_vars(list(rms_ds.data_vars), errors="ignore")
            std_ds["w"] = xr.DataArray(std_field[:, :, None], dims=("y", "x", "t"), attrs={"standard_name": "standard_deviation", "units": units})
            rms_ds["w"] = xr.DataArray(rms_field[:, :, None], dims=("y", "x", "t"), attrs={"standard_name": "root_mean_square", "units": units})
            std_ds = std_ds.assign_coords(t=np.asarray([0.0], dtype=float))
            rms_ds = rms_ds.assign_coords(t=np.asarray([0.0], dtype=float))
            std_ds.attrs = dict(ds.attrs)
            rms_ds.attrs = dict(ds.attrs)
            return avg, std_ds, rms_ds

        raise ValueError("Dataset must contain either ('u','v') or 'w' to use averf")

    def azaverf(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        *,
        center_units: Literal["phys", "mesh"] = "phys",
        rmax: Optional[float] = None,
        keepzero: bool = False,
        return_profiles: bool = False,
        var: Optional[str] = None,
        frame: Optional[int] = None,
    ):
        """Azimuthal average of a vector/scalar field.

        Inspired by PIVMAT's ``azaverf``.

        Parameters
        ----------
        x0, y0:
            Center location.
        center_units:
            'phys' (default) for coordinate units or 'mesh' for index units
            (0-based indices in x/y arrays).
        rmax:
            Optional maximum radius.
        keepzero:
            If False (default), zero elements are treated as invalid and excluded.
        return_profiles:
            If True, return radial profiles instead of an averaged field.
        var:
            Scalar variable name to average. If None, uses vector mode (u, v).
        frame:
            Optional integer time index to process a single frame.

        Returns
        -------
        xarray.Dataset
            If ``return_profiles`` is False, returns an averaged dataset.

        tuple
            If ``return_profiles`` is True:
            - Vector mode: ``(r, ur, ut)``
            - Scalar mode: ``(r, p)``
        """

        ds = self._obj
        if "x" not in ds.coords or "y" not in ds.coords:
            raise ValueError("azaverf requires 'x' and 'y' coordinates")

        x = np.asarray(ds.coords["x"].values, dtype=float)
        y = np.asarray(ds.coords["y"].values, dtype=float)
        if x.size < 2 or y.size < 2:
            raise ValueError("azaverf requires at least 2 points in x and y")

        dx = float(np.nanmedian(np.diff(x)))
        dy = float(np.nanmedian(np.diff(y)))
        dx_abs = abs(dx) if dx != 0 else 1.0
        dy_abs = abs(dy) if dy != 0 else 1.0
        dr = dx_abs
        if not np.isfinite(dr) or dr == 0.0:
            dr = 1.0

        if center_units == "mesh":
            # x0/y0 are 0-based indices into x/y.
            x0_phys = float(x[0] + x0 * dx)
            y0_phys = float(y[0] + y0 * dy)
            rmax_phys = None if rmax is None else float(rmax) * dr
        elif center_units == "phys":
            x0_phys = float(x0)
            y0_phys = float(y0)
            rmax_phys = None if rmax is None else float(rmax)
        else:
            raise ValueError("center_units must be 'phys' or 'mesh'")

        X, Y = np.meshgrid(x, y)
        dX = X - x0_phys
        dY = Y - y0_phys
        R = np.sqrt(dX * dX + dY * dY)
        if rmax_phys is not None:
            Rmask = R <= rmax_phys
        else:
            Rmask = np.ones_like(R, dtype=bool)

        # Bin index: 0 corresponds to r in [0, dr)
        bin_idx = np.floor(R / dr).astype(int)
        # Exclude center where projections are undefined.
        not_center = R > 0

        if "t" in ds.dims:
            t_indices = list(range(int(ds.sizes["t"])))
        else:
            # Treat as single frame.
            t_indices = [0]

        if frame is not None:
            frame_i = int(frame)
            t_indices = [frame_i]

        # Decide scalar vs vector mode.
        scalar_mode = var is not None
        if scalar_mode:
            if var not in ds:
                raise ValueError(f"Scalar variable '{var}' not found in dataset")
        else:
            if "u" not in ds or "v" not in ds:
                raise ValueError("Vector mode azaverf requires 'u' and 'v'")

        maxbin = int(np.nanmax(bin_idx[Rmask])) if np.any(Rmask) else 0
        # Preallocate profiles across time.
        if scalar_mode:
            prof = np.full((maxbin + 1, len(t_indices)), np.nan, dtype=float)
            counts_any = np.zeros((maxbin + 1,), dtype=int)
        else:
            prof_ur = np.full((maxbin + 1, len(t_indices)), np.nan, dtype=float)
            prof_ut = np.full((maxbin + 1, len(t_indices)), np.nan, dtype=float)
            counts_any = np.zeros((maxbin + 1,), dtype=int)

        # Flatten helpers.
        bin_flat = bin_idx.ravel()
        R_flat = R.ravel()
        dX_flat = dX.ravel()
        dY_flat = dY.ravel()
        base_mask_flat = (Rmask & not_center).ravel()

        for k, ti in enumerate(t_indices):
            if "t" in ds.dims:
                if scalar_mode:
                    A = np.asarray(ds[var].isel(t=ti).values, dtype=float)
                else:
                    U = np.asarray(ds["u"].isel(t=ti).values, dtype=float)
                    V = np.asarray(ds["v"].isel(t=ti).values, dtype=float)
            else:
                if scalar_mode:
                    A = np.asarray(ds[var].values, dtype=float)
                else:
                    U = np.asarray(ds["u"].values, dtype=float)
                    V = np.asarray(ds["v"].values, dtype=float)

            if scalar_mode:
                A_flat = A.ravel()
                finite = np.isfinite(A_flat)
                valid = base_mask_flat & finite
                if not keepzero:
                    valid = valid & (A_flat != 0)
                if not np.any(valid):
                    continue
                cnt = np.bincount(bin_flat[valid], minlength=maxbin + 1)
                s = np.bincount(bin_flat[valid], weights=A_flat[valid], minlength=maxbin + 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    p = s / cnt
                prof[:, k] = p
                counts_any = np.maximum(counts_any, cnt)
            else:
                U_flat = U.ravel()
                V_flat = V.ravel()
                finite = np.isfinite(U_flat) & np.isfinite(V_flat)
                valid = base_mask_flat & finite
                if not keepzero:
                    valid = valid & (U_flat != 0) & (V_flat != 0)
                if not np.any(valid):
                    continue

                with np.errstate(divide="ignore", invalid="ignore"):
                    ur_pt = (U_flat * dX_flat + V_flat * dY_flat) / R_flat
                    ut_pt = (-U_flat * dY_flat + V_flat * dX_flat) / R_flat

                cnt = np.bincount(bin_flat[valid], minlength=maxbin + 1)
                sur = np.bincount(bin_flat[valid], weights=ur_pt[valid], minlength=maxbin + 1)
                sut = np.bincount(bin_flat[valid], weights=ut_pt[valid], minlength=maxbin + 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ur = sur / cnt
                    ut = sut / cnt
                prof_ur[:, k] = ur
                prof_ut[:, k] = ut
                counts_any = np.maximum(counts_any, cnt)

        nonzero_bins = np.nonzero(counts_any)[0]
        if nonzero_bins.size == 0:
            if return_profiles:
                if scalar_mode:
                    return np.asarray([]), np.asarray([[]])
                return np.asarray([]), np.asarray([[]]), np.asarray([[]])
            # Return zeros field of same shape.
            out = ds.copy(deep=True)
            if scalar_mode:
                out[var] = out[var] * 0
            else:
                out["u"] = out["u"] * 0
                out["v"] = out["v"] * 0
            return out

        first_bin = int(nonzero_bins[0])
        last_bin = int(nonzero_bins[-1])
        r_vec = (np.arange(first_bin, last_bin + 1, dtype=float) * dr)

        if return_profiles:
            if scalar_mode:
                return r_vec, prof[first_bin : last_bin + 1, :]
            return r_vec, prof_ur[first_bin : last_bin + 1, :], prof_ut[first_bin : last_bin + 1, :]

        # Build azimuthally averaged field.
        out = ds.copy(deep=True)

        # Reconstruct per-frame fields from profiles.
        bin2 = bin_idx.copy()
        in_range = (bin2 >= first_bin) & (bin2 <= last_bin) & not_center & Rmask
        # Prepare output arrays.
        if "t" in ds.dims:
            t_full = int(ds.sizes["t"])
            if frame is None:
                t_out_indices = list(range(t_full))
                prof_cols = {ti: idx for idx, ti in enumerate(t_indices)}
            else:
                t_out_indices = [int(frame)]
                prof_cols = {int(frame): 0}
        else:
            t_out_indices = [0]
            prof_cols = {0: 0}

        if scalar_mode:
            # Set values by bin, outside range -> 0.
            if "t" in ds.dims:
                for ti in t_out_indices:
                    col = prof_cols.get(ti, None)
                    if col is None:
                        out[var].isel(t=ti).values[...] = 0.0
                        continue
                    p_full = np.zeros((maxbin + 1,), dtype=float)
                    p_slice = prof[:, col]
                    p_full[:] = np.nan_to_num(p_slice, nan=0.0)
                    w_new = np.zeros_like(R)
                    w_new[in_range] = p_full[bin2[in_range]]
                    out[var].isel(t=ti).values[...] = w_new
            else:
                p_full = np.zeros((maxbin + 1,), dtype=float)
                p_full[:] = np.nan_to_num(prof[:, 0], nan=0.0)
                w_new = np.zeros_like(R)
                w_new[in_range] = p_full[bin2[in_range]]
                out[var].values[...] = w_new
            return out

        # Vector mode.
        if "t" in ds.dims:
            for ti in t_out_indices:
                col = prof_cols.get(ti, None)
                if col is None:
                    out["u"].isel(t=ti).values[...] = 0.0
                    out["v"].isel(t=ti).values[...] = 0.0
                    continue
                ur_full = np.zeros((maxbin + 1,), dtype=float)
                ut_full = np.zeros((maxbin + 1,), dtype=float)
                ur_full[:] = np.nan_to_num(prof_ur[:, col], nan=0.0)
                ut_full[:] = np.nan_to_num(prof_ut[:, col], nan=0.0)
                ur_grid = np.zeros_like(R)
                ut_grid = np.zeros_like(R)
                ur_grid[in_range] = ur_full[bin2[in_range]]
                ut_grid[in_range] = ut_full[bin2[in_range]]

                u_new = np.zeros_like(R)
                v_new = np.zeros_like(R)
                # u = ur * cos(theta) - ut * sin(theta) where cos=dx/r, sin=dy/r
                u_new[in_range] = ur_grid[in_range] * (dX[in_range] / R[in_range]) - ut_grid[in_range] * (dY[in_range] / R[in_range])
                v_new[in_range] = ur_grid[in_range] * (dY[in_range] / R[in_range]) + ut_grid[in_range] * (dX[in_range] / R[in_range])
                out["u"].isel(t=ti).values[...] = u_new
                out["v"].isel(t=ti).values[...] = v_new
        else:
            ur_full = np.zeros((maxbin + 1,), dtype=float)
            ut_full = np.zeros((maxbin + 1,), dtype=float)
            ur_full[:] = np.nan_to_num(prof_ur[:, 0], nan=0.0)
            ut_full[:] = np.nan_to_num(prof_ut[:, 0], nan=0.0)
            ur_grid = np.zeros_like(R)
            ut_grid = np.zeros_like(R)
            ur_grid[in_range] = ur_full[bin2[in_range]]
            ut_grid[in_range] = ut_full[bin2[in_range]]
            u_new = np.zeros_like(R)
            v_new = np.zeros_like(R)
            u_new[in_range] = ur_grid[in_range] * (dX[in_range] / R[in_range]) - ut_grid[in_range] * (dY[in_range] / R[in_range])
            v_new[in_range] = ur_grid[in_range] * (dY[in_range] / R[in_range]) + ut_grid[in_range] * (dX[in_range] / R[in_range])
            out["u"].values[...] = u_new
            out["v"].values[...] = v_new

        return out

    def azprofile(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        r: float = 1.0,
        na: int | None = None,
        *,
        var: str | None = None,
        frame: int | None = None,
        angle_dim: str = "angle",
    ):
        """Azimuthal profile sampled along a circle (PIVMAT-style).

        This is a port of PIVMAT's ``azprofile``:
        samples a scalar or vector field along the circle
        ``(x, y) = (x0 + r*cos(a), y0 + r*sin(a))``.

        Parameters
        ----------
        x0, y0:
            Circle center in the same units as coordinates ``x`` and ``y``.
        r:
            Circle radius.
        na:
            Number of angular samples. If None, uses PIVMAT's default heuristic
            ``round(4*r/abs(dx))`` where ``dx`` is the x-grid spacing.
        var:
            Scalar variable to sample. If None, samples vector components ``u`` and ``v``
            and returns (angle, ur, ut).
        frame:
            Optional time index. If provided, samples only that frame.
        angle_dim:
            Name of the angular dimension.

        Returns
        -------
        tuple
            Scalar mode: ``(angle, p)``.

        tuple
            Vector mode: ``(angle, ur, ut)``.

        Notes
        -----
        Returned arrays are NumPy arrays. If the dataset has a time dimension and
        ``frame`` is None, the profiles have shape ``(na, nt)``.
        """

        ds = self._obj
        if "x" not in ds.coords or "y" not in ds.coords:
            raise ValueError("azprofile requires 'x' and 'y' coordinates")

        x = np.asarray(ds.coords["x"].values, dtype=float)
        if x.size < 2:
            raise ValueError("azprofile requires at least 2 x points")
        dx = float(np.nanmedian(np.diff(x)))
        dx_abs = abs(dx) if np.isfinite(dx) and dx != 0 else 1.0
        if na is None:
            na = int(round(4.0 * float(r) / dx_abs))
        na = int(na)
        if na <= 0:
            raise ValueError("na must be positive")

        angle = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
        x_s = x0 + float(r) * np.cos(angle)
        y_s = y0 + float(r) * np.sin(angle)

        a_da = xr.DataArray(angle, dims=(angle_dim,), coords={angle_dim: angle})
        x_da = xr.DataArray(x_s, dims=(angle_dim,), coords={angle_dim: angle})
        y_da = xr.DataArray(y_s, dims=(angle_dim,), coords={angle_dim: angle})
        cos_da = xr.DataArray(np.cos(angle), dims=(angle_dim,), coords={angle_dim: angle})
        sin_da = xr.DataArray(np.sin(angle), dims=(angle_dim,), coords={angle_dim: angle})

        # Optional frame selection.
        if frame is not None:
            if "t" not in ds.dims:
                raise ValueError("frame was provided but dataset has no 't' dimension")
            ds = ds.isel(t=int(frame))

        scalar_mode = var is not None
        if scalar_mode:
            if var not in ds:
                raise ValueError(f"Scalar variable '{var}' not found in dataset")
            p = ds[var].interp(x=x_da, y=y_da)
            # Ensure angle-first for numpy return.
            if angle_dim in p.dims:
                p = p.transpose(angle_dim, ...)
            return angle, np.asarray(p.values)

        # Vector mode
        if "u" not in ds or "v" not in ds:
            raise ValueError("Vector mode azprofile requires 'u' and 'v'")

        u_samp = ds["u"].interp(x=x_da, y=y_da)
        v_samp = ds["v"].interp(x=x_da, y=y_da)
        ur = u_samp * cos_da + v_samp * sin_da
        ut = -u_samp * sin_da + v_samp * cos_da

        ur = ur.transpose(angle_dim, ...)
        ut = ut.transpose(angle_dim, ...)
        return angle, np.asarray(ur.values), np.asarray(ut.values)

    def phaseaverf(
        self,
        period,
        *,
        opt: str = "",
        method: Literal["linear", "nearest"] = "linear",
    ):
        """Phase-average a vector/scalar dataset over a period.

        Inspired by PIVMAT's ``phaseaverf``.

        Parameters
        ----------
        period:
            If integer P: returns P phase-averaged fields, where phase i is the
            average of frames i, i+P, i+2P, ...
            If non-integer float: resamples linearly in time and averages. The
            result has length floor(period).
            If sequence: performs loop averaging with step=period[-1].
        opt:
            Passed to ``averf``. By default, zeros are excluded; pass '0' to include.
        method:
            Interpolation method used for non-integer periods.

        Returns
        -------
        xarray.Dataset
            Phase-averaged dataset with dim 't' == n_phases.
        """

        ds = self._obj
        if "t" not in ds.dims:
            raise ValueError("phaseaverf requires a time dimension 't'")

        n_frames = int(ds.sizes.get("t", 0) or 0)
        if n_frames <= 0:
            raise ValueError("Empty time dimension")

        # Work in index space (0..n_frames-1) to make non-integer periods well-defined.
        tini = np.arange(n_frames, dtype=float)

        def _avg_for_points(points: np.ndarray) -> xr.Dataset:
            points = np.asarray(points, dtype=float)
            points = points[(points >= 0) & (points <= n_frames - 1)]
            if points.size == 0:
                # Return a 0-field with the same layout (single frame)
                zero = ds.isel(t=[0]).copy(deep=True)
                for v in list(zero.data_vars):
                    zero[v].values[...] = 0.0
                return zero.assign_coords(t=np.asarray([0.0], dtype=float))
            sub = ds.piv.resamplef(tini=tini, tfin=points, method=method)
            return sub.piv.averf(opt)

        phases: list[xr.Dataset] = []

        # Determine period type.
        if np.isscalar(period):
            p = float(period)
            if abs(p - float(int(round(p)))) < 1e-10:
                P = int(np.floor(p))
                if P <= 0:
                    raise ValueError("period must be positive")
                for i in range(P):
                    # Fast path: integer stride selection, no interpolation needed.
                    sub = ds.isel(t=slice(i, None, P))
                    phases.append(sub.piv.averf(opt))
            else:
                P = int(np.floor(p))
                if P <= 0:
                    raise ValueError("period must be >= 1")
                for i in range(P):
                    points = np.arange(float(i), float(n_frames), p)
                    phases.append(_avg_for_points(points))
        else:
            tvec = np.asarray(period, dtype=float).ravel()
            if tvec.size == 0:
                raise ValueError("period sequence must be non-empty")
            step = float(tvec[-1])
            if step <= 0:
                raise ValueError("period step (last element) must be positive")
            for start in tvec:
                points = np.arange(float(start), float(n_frames), step)
                phases.append(_avg_for_points(points))

        out = xr.concat(phases, dim="t")
        out = out.assign_coords(t=np.arange(out.sizes["t"], dtype=float))
        out.attrs = dict(ds.attrs)
        return out

    def probef(
        self,
        x0,
        y0,
        *,
        variables: Optional[list[str]] = None,
        method: str = "linear",
    ) -> xr.Dataset:
        """Record the time evolution of probe point(s) (PIVMAT-inspired).

        This samples variable(s) at point(s) ``(x0, y0)`` using spatial
        interpolation.

        Parameters
        ----------
        x0, y0:
            Probe location(s) in physical units. Scalars or 1D arrays.
        variables:
            Variables to sample. If None, defaults to ``['u','v']`` when present,
            otherwise ``['w']``.
        method:
            Interpolation method ('linear' or 'nearest' are typical).

        Returns
        -------
        xarray.Dataset
            Sampled time series. For multiple probe points, includes a ``probe`` dim
            and coordinates ``x_probe``/``y_probe``.
        """

        return cprobef(self._obj, x0, y0, variables=variables, method=method)

    def probeaverf(
        self,
        rect,
        *,
        variables: Optional[list[str]] = None,
        skipna: bool = True,
    ) -> xr.Dataset:
        """Time series averaged over a rectangular area (PIVMAT-inspired).

        Parameters
        ----------
        rect:
            Rectangle ``[x1, y1, x2, y2]`` in physical units.
        variables:
            Variables to average. If None, defaults to ``['u','v']`` when present,
            otherwise ``['w']``.
        skipna:
            If True (default), NaNs are ignored.

        Returns
        -------
        xarray.Dataset
            Spatially averaged time series.
        """

        return cprobeaverf(self._obj, rect, variables=variables, skipna=skipna)

    def spatiotempf(
        self,
        X,
        Y,
        *,
        var: str = "w",
        n: Optional[int] = None,
        method: str = "linear",
    ) -> xr.Dataset:
        """Spatio-temporal diagram along line segment(s) (PIVMAT-inspired).

        Parameters
        ----------
        X, Y:
            Endpoints in physical units. Single line: ``X=[x0,x1]``, ``Y=[y0,y1]``.
            Multiple lines: ``X=[[x0,x1],[...]]``, same for ``Y``.
        var:
            Scalar variable name to sample.
        n:
            Number of sample points along each line (None -> heuristic).
        method:
            Interpolation method ('linear' or 'nearest' are typical).

        Returns
        -------
        xarray.Dataset
            Dataset containing variable ``st``.
        """

        return cspatiotempf(self._obj, X, Y, var=var, n=n, method=method)

    def tempcorrf(
        self,
        *,
        variables: Optional[list[str]] = None,
        opt: str = "",
        normalize: bool = False,
    ) -> xr.Dataset:
        """Temporal correlation function (PIVMAT-inspired).

        Parameters
        ----------
        variables:
            Variables to include. Default is ``['u','v']`` if present, otherwise ``['w']``.
        opt:
            Include zeros if opt contains ``'0'`` (default excludes zeros).
        normalize:
            If True, normalizes so that ``f(t=0)=1``.
        """

        return ctempcorrf(self._obj, variables=variables, opt=opt, normalize=normalize)

    def resamplef(
        self,
        tini,
        tfin,
        *,
        method: Literal["linear", "nearest"] = "linear",
    ):
        """(Temporal) re-sampling of vector/scalar fields.

        This method is inspired by PIVMat's `resamplef`.

        The dataset is re-sampled from initial times `tini` to new times `tfin`
        using interpolation along the time dimension.

        Requirements (as in PIVMat):
        - len(tini) == len(ds.t)
        - tini is strictly increasing
        - all tfin values are within [tini[0], tini[-1]]

        Args:
            tini: 1D sequence of initial times (length == number of frames).
            tfin: 1D sequence of target times.
            method: Interpolation method.

        Returns:
            xarray.Dataset: resampled dataset with dim 't' == len(tfin) and coords 't' == tfin.
        """

        ds = self._obj
        if "t" not in ds.dims:
            raise ValueError("resamplef requires a time dimension 't'")

        tini_arr = np.asarray(tini, dtype=float).ravel()
        tfin_arr = np.asarray(tfin, dtype=float).ravel()

        n_frames = int(ds.sizes.get("t", 0) or 0)
        if tini_arr.size != n_frames:
            raise ValueError("Size of tini must coincide with the dataset time dimension")

        if tini_arr.size < 2:
            raise ValueError("tini must contain at least 2 points")

        if np.any(np.diff(tini_arr) <= 0):
            raise ValueError("tini must be strictly increasing")

        if tfin_arr.size == 0:
            raise ValueError("tfin must be non-empty")

        if float(np.min(tfin_arr)) < float(tini_arr[0]) or float(np.max(tfin_arr)) > float(tini_arr[-1]):
            raise ValueError("Some values of tfin fall outside the bounds of tini")

        # Interpolate in a dedicated coordinate to avoid assumptions about existing ds.t.
        ds_time = ds.assign_coords(_resample_time=("t", tini_arr)).swap_dims({"t": "_resample_time"})
        out = ds_time.interp(_resample_time=tfin_arr, method=method)
        out = out.swap_dims({"_resample_time": "t"}).assign_coords(t=tfin_arr)
        out = out.drop_vars("_resample_time")
        out.attrs = dict(ds.attrs)
        return out

    def spaverf(
        self,
        opt: str = "xy",
        *,
        var: Optional[str] = None,
    ):
        """Spatial average over X and/or Y of a vector/scalar field.

        This method is inspired by PIVMat's `spaverf`.

        Args:
            opt: 'x', 'y', or 'xy' (default). If opt contains '0', zeros are
                included in the mean; otherwise zeros are excluded (treated as invalid).
                Examples: 'xy', 'x0', 'y0', 'xy0'.
            var: Scalar variable name to average (e.g. 'w'). If None, averages
                vector components 'u' and 'v'.

        Returns:
            xarray.Dataset: Dataset with spatially-averaged variable(s), broadcast back
            to the original shape.
        """

        ds = self._obj
        opt_l = str(opt).lower() if opt is not None else "xy"
        include_zeros = "0" in opt_l
        axis = opt_l.replace("0", "") or "xy"

        if axis not in {"x", "y", "xy"}:
            raise ValueError("Invalid axis; expected 'x', 'y', or 'xy' (optionally with '0')")

        def _mean_broadcast(da: xr.DataArray, reduce_dims: list[str]) -> xr.DataArray:
            if include_zeros:
                mean = da.mean(dim=reduce_dims, skipna=True)
            else:
                mean = da.where(da != 0).mean(dim=reduce_dims, skipna=True)
                mean = mean.fillna(0.0)
            # Broadcast back to original y/x/t shape.
            return mean.broadcast_like(da)

        if var is None:
            if "u" not in ds or "v" not in ds:
                raise ValueError("Vector mode spaverf requires 'u' and 'v'")
            vars_to_process = ["u", "v"]
        else:
            if var not in ds:
                raise ValueError(f"Scalar variable '{var}' not found in dataset")
            vars_to_process = [var]

        reduce_dims: list[str]
        if axis == "x":
            reduce_dims = ["x"]
        elif axis == "y":
            reduce_dims = ["y"]
        else:
            reduce_dims = ["y", "x"]

        out = ds.copy(deep=True)
        for name in vars_to_process:
            da = out[name]
            # Ensure y/x exist; allow missing t (single frame).
            if "y" not in da.dims or "x" not in da.dims:
                raise ValueError(f"Variable '{name}' must have spatial dims 'y' and 'x'")
            out[name] = _mean_broadcast(da, reduce_dims)
            out[name].attrs = dict(ds[name].attrs)

        out.attrs = dict(ds.attrs)
        return out

    def subaverf(
        self,
        opt: str = "e",
        *,
        var: Optional[str] = None,
    ):
        """Subtract an ensemble (temporal) or spatial average from a field.

        This method is inspired by PIVMat's `subaverf`.

        - If `opt` contains 'e' (default): subtract the ensemble/temporal mean
          computed by `averf`.
        - Otherwise: subtract a spatial mean computed by `spaverf` using `opt`
          as the axis selector ('x', 'y', 'xy', optionally with '0').

        By default, the subtraction preserves invalid zeros: locations that are
        exactly zero in the original data remain zero after subtraction.

        Args:
            opt: Option string. Default 'e'.
            var: Scalar variable name (e.g. 'w'). If None, operates on 'u' and 'v'.

        Returns:
            xarray.Dataset: Dataset with mean-subtracted variable(s).
        """

        ds = self._obj
        opt_l = str(opt).lower() if opt is not None else "e"
        ensemble = "e" in opt_l

        if var is None:
            if "u" not in ds or "v" not in ds:
                raise ValueError("Vector mode subaverf requires 'u' and 'v'")
            vars_to_process = ["u", "v"]
        else:
            if var not in ds:
                raise ValueError(f"Scalar variable '{var}' not found in dataset")
            vars_to_process = [var]

        out = ds.copy(deep=True)

        if ensemble:
            # Allow '0' to be passed through to averf if user included it.
            opt_for_averf = opt_l.replace("e", "")
            mean_ds = ds.piv.averf(opt_for_averf)

            for name in vars_to_process:
                da = ds[name]
                mean_da = mean_ds[name]

                # Broadcast mean to all time steps.
                if "t" in da.dims and "t" in mean_da.dims and mean_da.sizes.get("t", 1) == 1:
                    mean_b = mean_da.isel(t=0).broadcast_like(da)
                else:
                    mean_b = mean_da.broadcast_like(da)

                new = da - mean_b

                # Preserve invalid zeros (PIVMat multiplies by logical(original)).
                if "t" in da.dims:
                    new = new.where(da != 0, 0.0)
                else:
                    new = new.where(da != 0, 0.0)

                out[name] = new
                out[name].attrs = dict(ds[name].attrs)

            out.attrs = dict(ds.attrs)
            return out

        # Spatial subtraction mode.
        spatial_mean = ds.piv.spaverf(opt_l, var=var)
        for name in vars_to_process:
            da = ds[name]
            mean_da = spatial_mean[name]
            new = da - mean_da
            new = new.where(da != 0, 0.0)
            out[name] = new
            out[name].attrs = dict(ds[name].attrs)

        out.attrs = dict(ds.attrs)
        return out

    def fill_nans(self, method: Literal["linear", "nearest", "cubic"] = "nearest"):
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.
        Parameters
        ----------
        src_data: Any
            Input data array.
        method: {'linear', 'nearest', 'cubic'}
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

    def fill_zeros(
        self,
        *,
        fill: bool = False,
        max_iter: int | None = None,
        variables: list[str] | None = None,
    ) -> xr.Dataset:
        """Fill zero-valued holes using 4-neighbor interpolation.

        This is a PIVMAT ``interpolat``-style helper, useful when invalid vectors
        are encoded as zeros.

        Parameters
        ----------
        fill:
            If True, iterate until no zeros remain (or until max_iter).
        max_iter:
            Optional iteration cap.
        variables:
            Variables to process. Default: ['u', 'v'] if present; otherwise all data_vars.
        """

        ds = self._obj
        if variables is None:
            variables = [v for v in ("u", "v") if v in ds.data_vars] or list(ds.data_vars)

        out = ds.copy(deep=True)
        for name in variables:
            da = out[name]
            if da.ndim < 2:
                continue
            out[name] = interpolat_zeros_2d(da, fill=fill, max_iter=max_iter)
            out[name].attrs = dict(ds[name].attrs)

        out.attrs = dict(ds.attrs)
        return out

    def interpf(
        self,
        method: int = 0,
        *,
        variables: list[str] | None = None,
        missing: str = "0nan",
    ) -> xr.Dataset:
        """Interpolate missing data (PIVMAT-style ``interpf``).

        Missing values are defined as 0 and/or NaN (see ``missing``). The
        interpolation is applied frame-by-frame along ``t`` if present.

        Parameters
        ----------
        method:
            Interpolation method selector:
            ``0`` Laplacian inpainting (sparse solve),
            ``1`` nearest-neighbor fill,
            ``2`` linear interpolation with nearest fallback.
        variables:
            Variables to process. Default: ['u','v'] if present; otherwise ['w']
            if present; otherwise all data variables.
        missing:
            Missing-value definition: ``'0nan'`` (default), ``'nan'``, or ``'0'``.

        Returns
        -------
        xarray.Dataset
            Filled dataset.
        """

        return cinterpf(self._obj, method=int(method), variables=variables, missing=missing)

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
            name (str): Name for the output scalar field. Defaults to "w".
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
            name (str): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.

        Returns:
            xarray.Dataset: Dataset with added scalar field = du_dx^2 + dv_dy^2 + 0.5*(du_dy+dv_dx)^2
            
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
            name (str): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.

        Returns:
            xarray.Dataset: Dataset with the new property [name] = divergence
            
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
            name (str): Name for the output scalar field. Defaults to "w".
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
            name (str): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xarray.Dataset: Dataset with kinetic energy field
            
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
            name (str): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xarray.Dataset: New dataset with TKE field (based on fluctuations from mean)
            
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
            name (str): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xarray.Dataset: Dataset with Reynolds stress field (-<u'v'>)
            
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
            name (str): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xarray.Dataset: Dataset with RMS field (sqrt of TKE)
            
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
            self._obj (xarray.Dataset): Must contain at least ``u``, ``v``, ``x``, ``y`` and ``t``.
            n (int): Rolling window radius. Window size is ``(2*n+1) x (2*n+1)``.
            convCoords (bool): Convert coordinates.
                                if True - create two new data arrays within self._obj with
                                the names "xCoordiantes" and "yCoordiantes" that store x and y
                                coordinates as data arrays; always keep it "True" unless you
                                have already created "xCoordiantes" and "yCoordiantes" somehow
                                (say, by running Γ1 or Γ2 functions before)

        Returns:
            self._obj (xarray.Dataset) - the argument with the Γ1 data array
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
            self._obj (xarray.Dataset): Must contain at least ``u``, ``v``, ``x``, ``y`` and ``t``.
            n (int): Rolling window radius. Window size is ``(2*n+1) x (2*n+1)``.
            convCoords (bool): Convert coordinates.
                                if True - create two new data arrays within self._obj with
                                the names "xCoordiantes" and "yCoordiantes" that store x and y
                                coordinates as data arrays; always keep it "True" unless you
                                have already created "xCoordiantes" and "yCoordiantes" somehow
                                (say, by running Γ1 or Γ2 functions before)

        Returns:
            self._obj (xarray.Dataset) - the argument with the Γ2 data array
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
            flow_property (str): Name of the flow property to compute.
                Valid options: 'curl'/'vorticity'/'vort', 'ke'/'ken'/'kinetic_energy',
                'strain', 'divergence', 'acceleration', 'tke', 'reynolds_stress', 'rms'.
                Defaults to "curl".
            name (str): Name for the output scalar field. Defaults to "w".
                Use different names to store multiple scalar fields in one dataset.
                
        Returns:
            xarray.Dataset: Dataset with computed scalar field
            
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
            xarray.Dataset: Scaled dataset
            
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
            xarray.Dataset: Scaled dataset
            
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
            delta_t (float): Time interval between frame A and B. Defaults to 0.0.
            
        Returns:
            xarray.Dataset: Dataset with updated delta_t attribute
            
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
            scale (float): Scaling factor. Defaults to 1.0.
            
        Returns:
            xarray.Dataset: Dataset with scaled coordinates and velocities
            
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
            theta (float): Rotation angle in degrees (clockwise). Defaults to 0.0.
            
        Returns:
            xarray.Dataset: Rotated dataset
            
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
        fig, ax = gshowf(self._obj, **kwargs)
        return fig, ax

    def showscal(self, **kwargs):
        """method for graphics.showscal"""
        gshowscal(self._obj, **kwargs)

    def to_movie(self, output, **kwargs):
        """Save the Dataset as a movie (fast artist-updating renderer).

        This is a convenience wrapper around :func:`pivpy.graphics.to_movie`.

        Parameters
        ----------
        output:
            Output path (e.g. ``'movie.mp4'`` / ``'movie.gif'``). If ``None`` and
            ``return_frames=True`` is passed, returns a list of RGBA frames.
        **kwargs:
            Passed through to :func:`pivpy.graphics.to_movie`.
        """

        return gto_movie(self._obj, output, **kwargs)

    def jpdfscal(self, var1: str, var2: str, nbin: int = 101) -> xr.Dataset:
        """Joint PDF (2D histogram) of two scalar variables (PIVMAT-style).

        Parameters
        ----------
        var1, var2:
            Names of scalar variables in the Dataset.
        nbin:
            Number of bins per axis (odd integer, default 101).
        """

        if var1 not in self._obj:
            raise KeyError(f"Variable {var1} not found in dataset")
        if var2 not in self._obj:
            raise KeyError(f"Variable {var2} not found in dataset")
        return cjpdfscal(self._obj[var1], self._obj[var2], nbin=int(nbin))

    def jpdfscal_disp(self, var1: str, var2: str, nbin: int = 101, **kwargs):
        """Compute and display the joint PDF of two scalar variables."""

        jpdf = self.jpdfscal(var1, var2, nbin=nbin)
        return gjpdfscal_disp(jpdf, **kwargs)

    def histscal_disp(self, *args, **kwargs):
        """method for graphics.histscal_disp"""
        return ghistscal_disp(self._obj, *args, **kwargs)

    def histvec_disp(self, *args, **kwargs):
        """method for graphics.histvec_disp"""
        return ghistvec_disp(self._obj, *args, **kwargs)

    def autocorrelation_plot(self, variable: str = "u", spatial_average: bool = True, **kwargs):
        """Creates autocorrelation plot of a specified variable
        
        Args:
            variable (str): Variable name to plot autocorrelation for 
                (e.g., 'u', 'v', 'w', 'c', or any other data variable). Defaults to "u".
            spatial_average (bool): If True and time dimension exists, compute 
                spatial average before temporal autocorrelation. If False, flatten all 
                dimensions. Defaults to True for proper temporal analysis.
            **kwargs: Additional keyword arguments passed to graphics.autocorrelation_plot
            
        Returns:
            matplotlib.axes.Axes: The axes object containing the autocorrelation plot
            
        Example:
            >>> data.piv.autocorrelation_plot(variable='u')
            >>> data.piv.autocorrelation_plot(variable='v', spatial_average=False)
        """
        return gautocorrelation_plot(self._obj, variable=variable, 
                                     spatial_average=spatial_average, **kwargs)

    def corrm(
        self,
        variable: str = "u",
        dim: int | str = "x",
        *,
        half: bool = False,
        nan_as_zero: bool = True,
        lag_dim: str = "lag",
    ) -> xr.DataArray:
        """PIVMAT-style matrix correlation for a variable.

        Parameters
        ----------
        variable:
            Name of the DataArray variable in the Dataset.
        dim:
            Dimension name (recommended) or 1/2 like MATLAB for 2D arrays.
        half:
            If True, return only non-negative lags (including zero-lag).
        nan_as_zero:
            If True, treat NaNs as missing data and replace by 0 before correlating.
        lag_dim:
            Name of the lag dimension in the returned DataArray.
        """

        if variable not in self._obj:
            raise KeyError(f"Variable {variable} not in dataset")

        return corrm(
            self._obj[variable],
            dim=dim,
            half=half,
            nan_as_zero=nan_as_zero,
            lag_dim=lag_dim,
        )

    def corrf(
        self,
        variable: str = "u",
        dim: int | str = "x",
        *,
        normalize: bool = False,
        nan_as_zero: bool = True,
        nowarning: bool = False,
    ) -> xr.Dataset:
        """PIVMAT-style spatial correlation and integral scales for a scalar variable.

        This wraps :func:`pivpy.compute_funcs.corrf` and returns a Dataset with
        coordinate ``r`` and variable ``f`` plus scalar outputs (``isinf``, ``r5``, ...).
        """

        if variable not in self._obj:
            raise KeyError(f"Variable {variable} not in dataset")

        return corrf(
            self._obj[variable],
            dim=dim,
            normalize=normalize,
            nan_as_zero=nan_as_zero,
            nowarning=nowarning,
        )

    def gradientf(self, variable: str = "w") -> xr.Dataset:
        """PIVMAT-style gradient of a scalar variable.

        This wraps :func:`pivpy.compute_funcs.gradientf` and returns a new
        Dataset containing gradient components as variables ``u`` and ``v``.

        Parameters
        ----------
        variable:
            Name of the scalar variable in the Dataset (default: ``'w'``).

        Returns
        -------
        xarray.Dataset
            Dataset with variables ``u`` and ``v``.
        """

        if variable not in self._obj:
            raise KeyError(f"Variable {variable} not in dataset")

        return gradientf(self._obj[variable])

    def histf(
        self,
        variable: str | None = None,
        bin=None,
        opt: str = "",
    ) -> xr.Dataset:
        """PIVMAT-style histogram of a vector/scalar field.

        - Scalar mode: pass ``variable='w'`` (or any scalar var name) to get a
          Dataset with coordinate ``bin`` and variable ``h``.
        - Vector mode: pass ``variable=None`` (default) to compute histograms
          for both components (``u``/``v`` or ``vx``/``vy``), returning variables
          ``hx`` and ``hy``.

        By default, zero values are treated as invalid and excluded. Pass
        ``opt`` containing ``'0'`` to include zeros.
        """

        ds = self._obj
        include_zeros = "0" in str(opt)

        if variable is not None:
            if variable not in ds:
                raise KeyError(f"Variable {variable} not in dataset")
            return histf(ds[variable], bin=bin, opt="0" if include_zeros else "")

        # Vector mode
        if "u" in ds and "v" in ds:
            xname, yname = "u", "v"
        elif "vx" in ds and "vy" in ds:
            xname, yname = "vx", "vy"
        else:
            raise ValueError("histf vector mode requires ('u','v') or ('vx','vy')")

        hx_ds = histf(ds[xname], bin=bin, opt="0" if include_zeros else "")
        centers = hx_ds["bin"].values
        hy_ds = histf(ds[yname], bin=centers, opt="0" if include_zeros else "")

        out = xr.Dataset(
            {
                "hx": ("bin", np.asarray(hx_ds["h"].values, dtype=int)),
                "hy": ("bin", np.asarray(hy_ds["h"].values, dtype=int)),
            },
            coords={"bin": centers},
        )
        out["hx"].attrs["long_name"] = f"histogram({xname})"
        out["hy"].attrs["long_name"] = f"histogram({yname})"
        return out

    # @property
    # def vel_units(self):
    #     " Return the geographic center point of this dataset."
    #     if self._vel_units is None:
    #         self._vel_units = self._obj.attrs.l_units + '/' + \
    #                           self._obj.attrs.t_units
    #     return self._vel_units
