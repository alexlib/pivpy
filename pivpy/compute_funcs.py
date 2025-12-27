import numpy as np
import xarray as xr

try:
    from scipy.ndimage import convolve as _nd_convolve
except Exception:  # pragma: no cover
    _nd_convolve = None


def corrx(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    half: bool = False,
    nan_as_zero: bool = True,
) -> np.ndarray:
    """Vector correlation (PIVMAT-compatible).

    This ports the behavior of PIVMAT's ``corrx.m``:
    - Zero-padding is used outside the signal support.
    - Each lag is normalized by the number of *non-zero* products
      (so missing data encoded as zeros does not bias the result).

    Parameters
    ----------
    x, y:
        1D vectors. If ``y`` is None, autocorrelation is computed.
    half:
        If True, return only non-negative lags (including zero-lag).
    nan_as_zero:
        If True, NaNs are treated as missing data and replaced by 0.

    Returns
    -------
    np.ndarray
        Correlation vector of length ``2*N-1`` (or ``N`` if ``half=True``).
    """

    x_arr = np.asarray(x)
    if x_arr.ndim != 1:
        x_arr = x_arr.reshape(-1)

    y_arr = x_arr if y is None else np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Vectors lengths must agree.")

    x_arr = x_arr.astype(float, copy=False)
    y_arr = y_arr.astype(float, copy=False)
    if nan_as_zero:
        x_arr = np.nan_to_num(x_arr, nan=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0)

    n = int(x_arr.shape[0])
    if n == 0:
        return np.array([], dtype=float)

    y_pad = np.concatenate([np.zeros(n - 1, dtype=float), y_arr, np.zeros(n - 1, dtype=float)])

    c = np.zeros(2 * n - 1, dtype=float)
    # MATLAB: for i=(-n+1):(n-1)
    # Python: i in [-(n-1), ..., (n-1)]
    for i in range(-(n - 1), n):
        start = (n - i - 1)
        stop = (2 * n - i - 1)
        segment = y_pad[start:stop]
        prod = x_arr * segment
        weight = int(np.count_nonzero(prod))
        if weight == 0:
            weight = 1
        c[(n - 1) + i] = float(np.sum(prod) / weight)

    if half:
        c = c[(n - 1) :]

    return c


def corrm(
    x: xr.DataArray | np.ndarray,
    dim: int | str = 1,
    *,
    half: bool = False,
    nan_as_zero: bool = True,
    lag_dim: str = "lag",
) -> xr.DataArray | np.ndarray:
    """Matrix correlation along one dimension (PIVMAT-compatible).

    For a 2D matrix shaped (M, N):
    - ``dim=1`` correlates each column vector -> output shape (2*M-1, N)
    - ``dim=2`` correlates each row vector    -> output shape (M, 2*N-1)

    For xarray objects, ``dim`` may be a dimension name and the function
    generalizes to N-D by correlating along that dimension.
    """

    if isinstance(x, xr.DataArray):
        if isinstance(dim, int):
            if dim not in (1, 2):
                raise ValueError("For xarray inputs, dim as int must be 1 or 2.")
            if x.ndim < dim:
                raise ValueError(f"Input has only {x.ndim} dims; cannot use dim={dim}.")
            dim_name = x.dims[dim - 1]
        else:
            dim_name = dim
            if dim_name not in x.dims:
                raise ValueError(f"Dimension '{dim_name}' not found in DataArray dims {x.dims}.")

        n = int(x.sizes[dim_name])
        if n == 0:
            out = xr.full_like(x.isel({dim_name: slice(0, 0)}), np.nan)
            return out

        def _corrx_1d(vec: np.ndarray) -> np.ndarray:
            return corrx(vec, half=False, nan_as_zero=nan_as_zero)

        out = xr.apply_ufunc(
            _corrx_1d,
            x,
            input_core_dims=[[dim_name]],
            output_core_dims=[[lag_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Put the lag dimension where the original dim was.
        dim_index = list(x.dims).index(dim_name)
        desired_order = list(x.dims)
        desired_order[dim_index] = lag_dim
        out = out.transpose(*desired_order)

        lag = np.arange(-(n - 1), n, dtype=int)
        out = out.assign_coords({lag_dim: lag})

        if half:
            out = out.sel({lag_dim: slice(0, None)})

        return out

    # NumPy path (expects 2D for PIVMAT-like behavior)
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError("NumPy corrm expects a 2D array; for N-D use xarray DataArray.")

    m, n = arr.shape
    if dim == 2:
        c = np.zeros((m, 2 * n - 1), dtype=float)
        for i in range(m):
            c[i, :] = corrx(arr[i, :], half=False, nan_as_zero=nan_as_zero)
        if half:
            c = c[:, (n - 1) :]
        return c
    if dim == 1:
        c = np.zeros((2 * m - 1, n), dtype=float)
        for j in range(n):
            c[:, j] = corrx(arr[:, j], half=False, nan_as_zero=nan_as_zero)
        if half:
            c = c[(m - 1) :, :]
        return c

    raise ValueError("dim must be 1 or 2 (NumPy inputs) or a valid xarray dim name.")


def meannz(
    x: xr.DataArray | np.ndarray,
    dim: int | str | None = None,
    *,
    keep_attrs: bool = True,
) -> xr.DataArray | np.ndarray:
    """Mean of nonzero elements (PIVMAT-compatible).

    Matches the intent of PIVMAT's ``meannz.m``: normalize by the number of
    nonzero samples instead of the total number of samples.

    Notes
    -----
    - For xarray inputs, zeros are excluded but NaNs are skipped.
    - Where a reduction slice has no nonzero samples, the result is 0.
    """

    if isinstance(x, xr.DataArray):
        if dim is None:
            # First non-singleton dimension (Matlab-like).
            dim_name = next((d for d in x.dims if x.sizes[d] != 1), x.dims[0])
        elif isinstance(dim, int):
            dim_name = x.dims[int(dim)]
        else:
            dim_name = dim

        nz = x.where(x != 0)
        summed = nz.sum(dim=dim_name, skipna=True)
        count = (x != 0).sum(dim=dim_name)
        out = summed / count.where(count != 0)
        out = out.fillna(0.0)
        if keep_attrs:
            out.attrs = dict(x.attrs)
        return out

    arr = np.asarray(x)
    if dim is None:
        dim = next((i for i, s in enumerate(arr.shape) if s != 1), 0)
    if not isinstance(dim, int):
        raise ValueError("For NumPy inputs, dim must be an int or None.")

    arr_f = arr.astype(float, copy=False)
    mask = arr_f != 0
    summed = np.sum(np.where(mask, arr_f, 0.0), axis=dim)
    count = np.sum(mask, axis=dim)
    out = np.divide(summed, count, out=np.zeros_like(summed, dtype=float), where=(count != 0))
    out = np.nan_to_num(out, nan=0.0)
    return out


def interpolat_zeros_2d(
    m: xr.DataArray | np.ndarray,
    *,
    fill: bool = False,
    max_iter: int | None = None,
    nan_as_zero: bool = True,
) -> xr.DataArray | np.ndarray:
    """Interpolate zeros in a 2D field using 4-neighbor averaging.

    This mirrors the behavior of PIVMAT's legacy ``interpolat.m``:
    - Replaces zero entries by the mean of their nonzero 4-neighbors.
    - If ``fill=True``, repeats until no zeros remain (or ``max_iter``).

    Parameters
    ----------
    m:
        2D array (or xarray DataArray with at least two dims).
    fill:
        Iterate until no zeros remain.
    max_iter:
        Optional hard stop for iterations (recommended when fill=True).
    nan_as_zero:
        If True, NaNs are treated as 0 (missing/invalid).
    """

    def _interp_pass(a2: np.ndarray) -> np.ndarray:
        if _nd_convolve is None:
            raise ImportError("scipy is required for interpolat_zeros_2d")

        a2 = a2.astype(float, copy=True)
        if nan_as_zero:
            a2 = np.nan_to_num(a2, nan=0.0)

        zero_mask = a2 == 0
        if not np.any(zero_mask):
            return a2

        kernel = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        nonzero = (~zero_mask).astype(float)
        neigh_sum = _nd_convolve(a2, kernel, mode="nearest")
        neigh_cnt = _nd_convolve(nonzero, kernel, mode="nearest")

        # Only update zeros where at least one nonzero neighbor exists.
        update = zero_mask & (neigh_cnt > 0)
        a2[update] = neigh_sum[update] / neigh_cnt[update]
        return a2

    if isinstance(m, xr.DataArray):
        if m.ndim < 2:
            raise ValueError("interpolat_zeros_2d requires at least 2D input")
        y_dim, x_dim = m.dims[0], m.dims[1]

        def _core(arr2: np.ndarray) -> np.ndarray:
            out = arr2
            it = 0
            while True:
                new = _interp_pass(out)
                it += 1
                if not fill:
                    return new
                if not np.any(new == 0):
                    return new
                if max_iter is not None and it >= int(max_iter):
                    return new
                if np.array_equal(new, out):
                    return new
                out = new

        out = xr.apply_ufunc(
            _core,
            m,
            input_core_dims=[[y_dim, x_dim]],
            output_core_dims=[[y_dim, x_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        out = out.assign_coords({y_dim: m[y_dim], x_dim: m[x_dim]})
        out.attrs = dict(m.attrs)
        return out

    arr = np.asarray(m)
    if arr.ndim != 2:
        raise ValueError("interpolat_zeros_2d expects a 2D NumPy array")
    out = arr.astype(float, copy=True)
    it = 0
    while True:
        new = _interp_pass(out)
        it += 1
        if not fill:
            return new
        if not np.any(new == 0):
            return new
        if max_iter is not None and it >= int(max_iter):
            return new
        if np.array_equal(new, out):
            return new
        out = new


def Γ1_moving_window_function(
        fWin: xr.Dataset,
        n: int,
) -> xr.DataArray:
    """
    This is the implementation of Γ1 function given by equation 9 in L.Graftieaux, M. Michard,
    N. Grosjean, "Combining PIV, POD and vortex identification algorithms for the study of
    unsteady turbulent swirling flows", Meas.Sci.Technol., 12(2001), p.1422-1429.
    Γ1 function is used to identify the locations of the centers of the vortices (which are
    given by the Γ1 peak values within the velocity field).
    IMPORTANT NOTICE: even though this function, theoretically, can be used on its own,  
    it is not supposed to. It is designed to be used with Dask for big
    PIV datasets. The recomendation is not to use this function on its own, but rather use
    Γ1 attribute of piv class of PIVPY package (example of usage in this case would be: ds.piv.Γ1())
    This function accepts a (2*n+1)x(2*n+1) neighborhood of one velocity vector from the
    entire velocity field in the form of Xarray dataset. And for this neighborhood only, 
    it calculates the value of Γ1.
    Also, note this function is designed in a way that assumes point P (see the referenced
    article) to coincide with the center for fWin.
    This function works only for 2D velocity field.

    Args:
        fWin (xarray.Dataset) - a moving window of the dataset (fWin = field rolling window)
        n (int) - the rolling window size (n=1 means a 3x3 rolling window)

    Returns:
        xr.DataArray(Γ1) (xr.DataArray) - an xarray DataArray object with Γ1 caclculated for
                                          for the given rolling window
    """
    # We must convert fWin to numpy, because when this function was originally implemented 
    # with fWin being an xr.Dataset, it was unbelievably slow! Conversion of fWin to numpy 
    # proved to give an incredible boost in speed.
    # To speed up the things even more I put everything in one line, which is unreadable.
    # Thus, to understand, what is going one, I'm giving a break up of the line
    # (the names of the variables are taken from the referenced article):
    # PMx = fWin['xCoordinates'].to_numpy() - float(fWin['xCoordinates'][n,n])
    # PMy = fWin['yCoordinates'].to_numpy() - float(fWin['yCoordinates'][n,n])
    # PM = np.sqrt(np.add(np.square(PMx), np.square(PMy)))
    # u = fWin['u'].to_numpy()
    # v = fWin['v'].to_numpy()
    # U = (u**2 + v**2)**(0.5)
    # The external tensor product (PM ^ U) * z (see the referenced article) can be simplified as a
    # cross product for the case of the 2D velocity field: (PM x U). According to the rules of
    # cross product, for the 2D velocity field, we have (PM x U) = PM_x * v - PM_y * u. In other
    # words, we don't have to compute the sin given in the referenced article. But I am, still,
    # going to use sin down below as the variable to be consistent with the expression given
    # in the referenced article.
    # sinΘM_Γ1 = (PMx*v - PMy*u) / PM / U 
    # Γ1 = np.nansum(sinΘM_Γ1) / (((2*n+1)**2))
    # And now here goes my one-liner. Note, that I didn't put PMx, PMy, u and v calculations
    # into my line. That's because I figured out emperically that would slow down the calculations.
    # n always points to the central interrogation window (just think of it). It gives me point P.
    # Robustly extract the central-point coordinates.
    # Newer xarray versions may reorder dimensions; avoid assuming positional [n, n]
    # returns a scalar.
    xcoords = fWin['xCoordinates']
    ycoords = fWin['yCoordinates']

    def _center_scalar(arr: xr.DataArray) -> float:
        if 'rollWx' in arr.dims and 'rollWy' in arr.dims:
            sel = arr.isel(rollWx=n, rollWy=n)
        elif len(arr.dims) >= 2:
            sel = arr.isel({arr.dims[0]: n, arr.dims[1]: n})
        else:
            sel = arr
        # Reduce any remaining dims (e.g. rollWt) to a scalar.
        for d in list(sel.dims):
            sel = sel.isel({d: 0})
        return float(np.asarray(sel.values).reshape(-1)[0])

    cx = _center_scalar(xcoords)
    cy = _center_scalar(ycoords)

    PMx = np.subtract(xcoords.to_numpy(), cx)
    PMy = np.subtract(ycoords.to_numpy(), cy)    
    u = fWin['u'].to_numpy()
    v = fWin['v'].to_numpy()  
    # Since for the case when point M coincides with point P we have a 0/0 situation, we'll
    # recive a warning. To temporarily suspend that warning do the following (credit goes to
    # https://stackoverflow.com/a/29950752/10073233):
    with np.errstate(divide='ignore', invalid='ignore'):
        Γ1 = np.nanmean(np.divide(np.subtract(np.multiply(PMx,v), np.multiply(PMy,u)), np.multiply(np.sqrt(np.add(np.square(PMx), np.square(PMy))), np.sqrt(np.add(np.square(u), np.square(v))))))

    return xr.DataArray(Γ1).fillna(0.0) # fillna(0) is necessary for plotting


def Γ2_moving_window_function(
        fWin: xr.Dataset,
        n: int,
) -> xr.DataArray:
    """
    This is the implementation of Γ2 function given by equation 11 in L.Graftieaux, M. Michard,
    N. Grosjean, "Combining PIV, POD and vortex identification algorithms for the study of
    unsteady turbulent swirling flows", Meas.Sci.Technol., 12(2001), p.1422-1429.
    Γ2 function is used to identify the boundaries of the vortices in a velocity field.
    IMPORTANT NOTICE: even though this function, theoretically, can be used on its own, 
    it is not supposed to. It is designed to be used with Dask for big
    PIV datasets. The recomendation is not to use this function on its own, but rather use
    Γ2 attribute of piv class of PIVPY package (example of usage in this case would be: ds.piv.Γ2())
    This function accepts a (2*n+1)x(2*n+1) neighborhood of one velocity vector from the
    entire velocity field in the form of Xarray dataset. And for this neighborhood only, 
    it calculates the value of Γ2.
    Also, note this function is designed in a way that assumes point P (see the referenced
    article) to coincide with the center for fWin.
    And finally, the choice of convective velocity (see the referenced article) is made in the
    article: it is the average velocity within fWin.
    This function works only for 2D velocity field.

    Args:
        fWin (xarray.Dataset) - a moving window of the dataset (fWin = field rolling window)
        n (int) - the rolling window size (n=1 means a 3x3 rolling window)

    Returns:
        xr.DataArray(Γ2) (xr.DataArray) - an xarray DataArray object with Γ2 caclculated for
                                          for the given rolling window
    """
    # We must convert fWin to numpy, because when this function was originally implemented 
    # with fWin being an xr.Dataset, it was unbelievably slow! Conversion of fWin to numpy 
    # proved to give an incredible boost in speed.
    # To speed up the things even more I put everything in one line, which is unreadable.
    # Thus, to understand, what is going one, I'm giving a break up of the line
    # (the names of the variables are taken from the referenced article):
    # PMx = fWin['xCoordinates'].to_numpy() - float(fWin['xCoordinates'][n,n])
    # PMy = fWin['yCoordinates'].to_numpy() - float(fWin['yCoordinates'][n,n])
    # PM = np.sqrt(np.add(np.square(PMx), np.square(PMy)))
    # u = fWin['u'].to_numpy()
    # v = fWin['v'].to_numpy()
    # We are going to include point P into the calculations of velocity UP_tilde
    # uP_tilde = np.nanmean(u)
    # vP_tilde = np.nanmean(v)
    # uDif = u - uP_tilde
    # vDif = v - vP_tilde
    # UDif = (uDif**2 + vDif**2)**(0.5)
    # The external tensor product (PM ^ UDif) * z (see the referenced article) can be simplified as a
    # cross product for the case of the 2D velocity field: (PM x UDif). According to the rules of
    # cross product, for the 2D velocity field, we have (PM x UDif) = PM_x * vDif - PM_y * uDif. 
    # I am going to use sin down below as the variable to be consistent with the expression given
    # for Γ1 function.
    # sinΘM_Γ2 = (PMx*vDif - PMy*uDif) / PM / UDif 
    # Γ2 = np.nansum(sinΘM_Γ2) / (((2*n+1)**2))
    # And now here goes my one-liner. Note, that I didn't put PMx, PMy, u and v calculations
    # into my line. That's because I figured out emperically that would slow down the calculations.
    # n always points to the central interrogation window (just think of it). It gives me point P.
    xcoords = fWin['xCoordinates']
    ycoords = fWin['yCoordinates']

    def _center_scalar(arr: xr.DataArray) -> float:
        if 'rollWx' in arr.dims and 'rollWy' in arr.dims:
            sel = arr.isel(rollWx=n, rollWy=n)
        elif len(arr.dims) >= 2:
            sel = arr.isel({arr.dims[0]: n, arr.dims[1]: n})
        else:
            sel = arr
        for d in list(sel.dims):
            sel = sel.isel({d: 0})
        return float(np.asarray(sel.values).reshape(-1)[0])

    cx = _center_scalar(xcoords)
    cy = _center_scalar(ycoords)

    PMx = np.subtract(xcoords.to_numpy(), cx)
    PMy = np.subtract(ycoords.to_numpy(), cy)    
    u = fWin['u'].to_numpy()
    v = fWin['v'].to_numpy()  
    uDif = u - np.nanmean(u)
    vDif = v - np.nanmean(v)
    # Since for the case when point M coincides with point P, we have a 0/0 situation and we'll
    # recive a warning. To temporarily suspend that warning do the following (credit goes to
    # https://stackoverflow.com/a/29950752/10073233):
    with np.errstate(divide='ignore', invalid='ignore'):
        Γ2 = np.nanmean(np.divide(np.subtract(np.multiply(PMx,vDif), np.multiply(PMy,uDif)), np.multiply(np.sqrt(np.add(np.square(PMx), np.square(PMy))), np.sqrt(np.add(np.square(uDif), np.square(vDif))))))

    return xr.DataArray(Γ2).fillna(0.0) # fillna(0) is necessary for plotting