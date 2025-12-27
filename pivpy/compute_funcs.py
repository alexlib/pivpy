import warnings

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

try:
    from scipy.ndimage import map_coordinates as _nd_map_coordinates
except Exception:  # pragma: no cover
    _nd_map_coordinates = None

try:
    from scipy.interpolate import griddata as _sp_griddata
except Exception:  # pragma: no cover
    _sp_griddata = None

try:
    from scipy.ndimage import convolve as _nd_convolve
except Exception:  # pragma: no cover
    _nd_convolve = None

try:
    from scipy.signal import convolve2d as _signal_convolve2d
except Exception:  # pragma: no cover
    _signal_convolve2d = None


def _kernel_flat(filtsize: float) -> np.ndarray:
    n = int(np.ceil(float(filtsize)))
    if n <= 0:
        return np.ones((1, 1), dtype=float)
    k = np.ones((n, n), dtype=float)
    k /= float(np.sum(k))
    return k


def _kernel_gauss(filtsize: float, *, integrated: bool = False) -> np.ndarray:
    fs = float(filtsize)
    if fs <= 0.0 or not np.isfinite(fs):
        return np.ones((1, 1), dtype=float)

    half_width = int(np.ceil(3.5 * fs))
    coords = np.arange(-half_width, half_width + 1, dtype=float)

    if integrated:
        # Integrated Gaussian over pixel area [-0.5, 0.5], to reduce discretization
        # effects for small filtsize.
        try:
            from scipy.special import erf  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("filterf(method='igauss') requires SciPy (scipy.special.erf)") from exc

        s = np.sqrt(2.0) * fs
        left = (coords - 0.5) / s
        right = (coords + 0.5) / s
        g1 = 0.5 * (erf(right) - erf(left))
        k = np.outer(g1, g1)
    else:
        X, Y = np.meshgrid(coords, coords)
        k = np.exp(-(X * X + Y * Y) / (2.0 * fs * fs))

    k = np.asarray(k, dtype=float)
    s = float(np.sum(k))
    if not np.isfinite(s) or s == 0.0:
        return np.ones((1, 1), dtype=float)
    k /= s
    return k


def filter2d_kernel(filtsize: float = 1.0, method: str = "gauss") -> np.ndarray:
    """Return the normalized 2D kernel used by :func:`filter2d`.

    Parameters
    ----------
    filtsize:
        Filter size in mesh units.
    method:
        'gauss', 'flat', or 'igauss'.
    """

    m = str(method).lower()
    fs = float(filtsize)
    if m.startswith("flat"):
        return _kernel_flat(fs)
    if m.startswith("igauss"):
        return _kernel_gauss(fs, integrated=True)
    return _kernel_gauss(fs, integrated=False)


def filter2d(
    arr2: ArrayLike,
    filtsize: float = 1.0,
    method: str = "gauss",
    *,
    mode: str = "valid",
) -> np.ndarray:
    """2D spatial filter by normalized convolution (PIVMAT-inspired).

    Parameters
    ----------
    arr2:
        2D field to filter.
    filtsize:
        Filter size in mesh units (Gaussian sigma-like scale).
    method:
        'gauss', 'flat', or 'igauss' (integrated Gaussian).
    mode:
        'valid' (default) or 'same'. Matches Matlab conv2 modes.

    Notes
    -----
    Missing values are treated as NaN and excluded from the convolution by
    using a weight-mask normalization.
    """

    a = np.asarray(arr2, dtype=float)
    if a.ndim != 2:
        raise ValueError("filter2d expects a 2D array")

    fs = float(filtsize)
    if fs == 0.0:
        return a

    m = str(method).lower()
    if m.startswith("flat"):
        k = _kernel_flat(fs)
    elif m.startswith("igauss"):
        k = _kernel_gauss(fs, integrated=True)
    else:
        k = _kernel_gauss(fs, integrated=False)

    if _signal_convolve2d is None:  # pragma: no cover
        raise ImportError("filter2d requires scipy.signal.convolve2d")

    finite = np.isfinite(a)
    a0 = np.where(finite, a, 0.0)
    w = finite.astype(float)

    mode_l = str(mode).lower()
    if mode_l not in ("valid", "same"):
        raise ValueError("mode must be 'valid' or 'same'")

    num = _signal_convolve2d(a0, k, mode=mode_l, boundary="fill", fillvalue=0.0)
    den = _signal_convolve2d(w, k, mode=mode_l, boundary="fill", fillvalue=0.0)

    out = np.full_like(num, np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def bwfilter2d(arr2: np.ndarray, filtsize: float, order: float) -> np.ndarray:
    """2D Butterworth filter in Fourier space (PIVMAT-inspired).

    Parameters
    ----------
    arr2:
        2D array.
    filtsize:
        Cutoff size in grid units. If 0, returns input.
    order:
        Butterworth order. Positive -> low-pass. Negative -> high-pass.

    Notes
    -----
    This follows the PIVMAT convention:
    - k is measured in index space on the FFT-shifted grid.
    - kc = n / filtsize, where n = min(nx, ny).
    - Transfer: T(k)=1/(1+(k/kc)^(order/2)).
    """

    a = np.asarray(arr2, dtype=float)
    if a.ndim != 2:
        raise ValueError("bwfilter2d expects a 2D array")

    fs = float(filtsize)
    if fs == 0.0:
        return a

    ny, nx = a.shape
    # PIVMAT behavior: if odd, discard last row/col.
    if nx % 2 == 1:
        a = a[:, :-1]
        nx -= 1
    if ny % 2 == 1:
        a = a[:-1, :]
        ny -= 1

    n = float(min(nx, ny))
    kc = n / fs
    if not np.isfinite(kc) or kc == 0.0:
        return a

    # Integer-like wavenumbers on the shifted FFT grid: [-N/2 .. N/2-1]
    kx = np.fft.fftshift(np.fft.fftfreq(nx) * nx)
    ky = np.fft.fftshift(np.fft.fftfreq(ny) * ny)
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX * KX + KY * KY)

    p = float(order) / 2.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        powterm = np.power(k / kc, p)
        T = 1.0 / (1.0 + powterm)

    # Fix the zero-mode explicitly for numerical stability.
    if order < 0:
        T[k == 0] = 0.0
    else:
        T[k == 0] = 1.0

    sp = np.fft.fftshift(np.fft.fft2(a))
    out = np.fft.ifft2(np.fft.ifftshift(sp * T)).real
    return out


def corrx(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    half: bool = False,
    nan_as_zero: bool = True,
) -> np.ndarray:
    """Vector correlation (PIVMAT-compatible).

    This ports the behavior of PIVMAT's ``corrx.m``. Zero-padding is used
    outside the signal support, and each lag is normalized by the number of
    non-zero products (so missing data encoded as zeros does not bias the
    result).

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
    numpy.ndarray
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


def _sel_coord_range(ds: xr.Dataset, coord: str, lo: float, hi: float) -> xr.Dataset:
    if coord not in ds.coords:
        raise ValueError(f"Dataset is missing coordinate '{coord}'")

    vals = np.asarray(ds[coord].values, dtype=float)
    if vals.size == 0:
        return ds
    if vals.size == 1:
        return ds

    increasing = bool(vals[1] >= vals[0])
    a = float(lo)
    b = float(hi)
    if increasing:
        return ds.sel({coord: slice(min(a, b), max(a, b))})
    return ds.sel({coord: slice(max(a, b), min(a, b))})


def probef(
    ds: xr.Dataset,
    x0,
    y0,
    *,
    variables: list[str] | None = None,
    method: str = "linear",
) -> xr.Dataset:
    """Record the time evolution of probe point(s) in a dataset (PIVMAT-inspired).

    This samples one or more variables at one or more probe points (x0, y0)
    using xarray's interpolation along the spatial coordinates.

    Parameters
    ----------
    ds:
        Input dataset with spatial coordinates ``x`` and ``y``.
    x0, y0:
        Probe coordinates in physical units. Scalars or 1D arrays of equal length.
    variables:
        Variables to sample. If None, defaults to ``['u','v']`` when present,
        otherwise tries ``['w']``.
    method:
        Interpolation method, passed to ``DataArray.interp`` (e.g. 'linear', 'nearest').

    Returns
    -------
    xarray.Dataset
        Dataset of sampled time series. If multiple probe points are given,
        the result has a ``probe`` dimension.
    """

    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("probef requires coordinates 'x' and 'y'")

    if variables is None:
        if "u" in ds and "v" in ds:
            variables = ["u", "v"]
        elif "w" in ds:
            variables = ["w"]
        else:
            # Fall back to first data_var if any.
            if not ds.data_vars:
                raise ValueError("probef: dataset has no data variables")
            variables = [next(iter(ds.data_vars))]

    variables = list(variables)
    for name in variables:
        if name not in ds:
            raise KeyError(f"Variable '{name}' not found in dataset")

    x_arr = np.asarray(x0, dtype=float)
    y_arr = np.asarray(y0, dtype=float)
    if x_arr.ndim == 0 and y_arr.ndim == 0:
        # Single probe point
        x_da: xr.DataArray | float = float(x_arr)
        y_da: xr.DataArray | float = float(y_arr)
        probe_coord = None
    else:
        x_arr = np.atleast_1d(x_arr).astype(float)
        y_arr = np.atleast_1d(y_arr).astype(float)
        if x_arr.shape != y_arr.shape:
            raise ValueError("x0 and y0 must have the same shape")
        probe = np.arange(int(x_arr.size), dtype=int)
        x_da = xr.DataArray(x_arr.reshape(-1), dims=("probe",), coords={"probe": probe})
        y_da = xr.DataArray(y_arr.reshape(-1), dims=("probe",), coords={"probe": probe})
        probe_coord = probe

    out_vars: dict[str, xr.DataArray] = {}
    for name in variables:
        da = ds[name]
        if "x" not in da.dims or "y" not in da.dims:
            raise ValueError(f"Variable '{name}' must have dims including 'x' and 'y'")
        sampled = da.interp(x=x_da, y=y_da, method=method)
        out_vars[name] = sampled

    out = xr.Dataset(out_vars)
    if probe_coord is not None:
        out = out.assign_coords(
            x_probe=("probe", np.asarray(x_arr.reshape(-1), dtype=float)),
            y_probe=("probe", np.asarray(y_arr.reshape(-1), dtype=float)),
        )
    out.attrs = dict(ds.attrs)
    return out


def probeaverf(
    ds: xr.Dataset,
    rect,
    *,
    variables: list[str] | None = None,
    skipna: bool = True,
) -> xr.Dataset:
    """Time series averaged over a rectangular area (PIVMAT-inspired).

    Parameters
    ----------
    ds:
        Input dataset with spatial coordinates ``x`` and ``y``.
    rect:
        Rectangle as ``[x1, y1, x2, y2]`` in physical units.
    variables:
        Variables to average. If None, defaults to ``['u','v']`` when present,
        otherwise tries ``['w']``.
    skipna:
        If True (default), NaNs are ignored in the mean.

    Returns
    -------
    xarray.Dataset
        Dataset containing the spatially averaged time series.
    """

    if rect is None or not hasattr(rect, "__len__") or len(rect) != 4:
        raise ValueError("rect must be [x1, y1, x2, y2]")

    if variables is None:
        if "u" in ds and "v" in ds:
            variables = ["u", "v"]
        elif "w" in ds:
            variables = ["w"]
        else:
            if not ds.data_vars:
                raise ValueError("probeaverf: dataset has no data variables")
            variables = [next(iter(ds.data_vars))]

    variables = list(variables)
    for name in variables:
        if name not in ds:
            raise KeyError(f"Variable '{name}' not found in dataset")

    x1, y1, x2, y2 = [float(v) for v in rect]
    sub = _sel_coord_range(ds, "x", x1, x2)
    sub = _sel_coord_range(sub, "y", y1, y2)

    out_vars: dict[str, xr.DataArray] = {}
    for name in variables:
        da = sub[name]
        if "x" not in da.dims or "y" not in da.dims:
            raise ValueError(f"Variable '{name}' must have dims including 'x' and 'y'")
        out_vars[name] = da.mean(dim=("y", "x"), skipna=bool(skipna))

    out = xr.Dataset(out_vars)
    out.attrs = dict(ds.attrs)
    return out


def spatiotempf(
    ds: xr.Dataset,
    X,
    Y,
    *,
    var: str = "w",
    n: int | None = None,
    method: str = "linear",
) -> xr.Dataset:
    """Spatio-temporal diagram along one (or more) line segments (PIVMAT-inspired).

    This samples a scalar field along line segment(s) defined by endpoints
    ``(X[i,0], Y[i,0]) -> (X[i,1], Y[i,1])`` and returns the sampled values
    as a function of time and curvilinear coordinate.

    Parameters
    ----------
    ds:
        Input dataset with coordinates ``x`` and ``y``.
    X, Y:
        Endpoints in physical units.

        - Single line: ``X=[x0,x1]``, ``Y=[y0,y1]``.
        - Multiple lines: ``X=[[x0,x1],[...]]``, ``Y=[[y0,y1],[...]]``.
    var:
        Scalar variable name to sample.
    n:
        Number of sample points along each line. If None, a heuristic based on
        grid spacing is used.
    method:
        Interpolation method for ``DataArray.interp``.

    Returns
    -------
    xarray.Dataset
        Dataset with variable ``st``.

        - dims: ``('t','s')`` for a single line, or ``('line','t','s')`` for multiple.
        - coords: ``s`` is the distance along the line (same units as x/y).
          ``x_line`` and ``y_line`` give the sampled coordinates along the line.
    """

    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("spatiotempf requires coordinates 'x' and 'y'")
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset")

    da = ds[var]
    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError(f"Variable '{var}' must have dims including 'x' and 'y'")
    if "t" not in da.dims:
        raise ValueError("spatiotempf requires a time dimension 't'")

    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)

    # Normalize to (nlines, 2)
    if X_arr.ndim == 1:
        if X_arr.size != 2 or Y_arr.ndim != 1 or Y_arr.size != 2:
            raise ValueError("For a single line, X and Y must be length-2 sequences")
        X_arr = X_arr.reshape(1, 2)
        Y_arr = Y_arr.reshape(1, 2)
        single_line = True
    else:
        if X_arr.ndim != 2 or Y_arr.ndim != 2 or X_arr.shape != Y_arr.shape or X_arr.shape[1] != 2:
            raise ValueError("For multiple lines, X and Y must be shaped (nlines, 2)")
        single_line = X_arr.shape[0] == 1

    # Heuristic for n based on average grid spacing
    if n is None:
        xvals = np.asarray(ds["x"].values, dtype=float)
        yvals = np.asarray(ds["y"].values, dtype=float)
        dx = float(np.nanmedian(np.abs(np.diff(xvals)))) if xvals.size >= 2 else 1.0
        dy = float(np.nanmedian(np.abs(np.diff(yvals)))) if yvals.size >= 2 else 1.0
        d = float(np.nanmin([dx, dy])) if np.isfinite(dx) and np.isfinite(dy) else 1.0
        if not np.isfinite(d) or d <= 0:
            d = 1.0
        lengths = np.sqrt((X_arr[:, 1] - X_arr[:, 0]) ** 2 + (Y_arr[:, 1] - Y_arr[:, 0]) ** 2)
        maxlen = float(np.nanmax(lengths)) if lengths.size else 0.0
        n = int(max(2, min(4096, np.ceil(maxlen / d) + 1)))
    else:
        n = int(n)
        if n < 2:
            raise ValueError("n must be >= 2")

    s = np.linspace(0.0, 1.0, n, dtype=float)

    # We keep a normalized s in [0,1] as the dimension for robust concatenation.
    # Physical distance along the line is provided as coordinate ``s_phys``.
    line_lengths = np.sqrt((X_arr[:, 1] - X_arr[:, 0]) ** 2 + (Y_arr[:, 1] - Y_arr[:, 0]) ** 2)
    line_lengths = np.asarray(line_lengths, dtype=float)

    out_list: list[xr.DataArray] = []
    for i in range(int(X_arr.shape[0])):
        x_line = X_arr[i, 0] + s * (X_arr[i, 1] - X_arr[i, 0])
        y_line = Y_arr[i, 0] + s * (Y_arr[i, 1] - Y_arr[i, 0])

        x_da = xr.DataArray(x_line, dims=("s",), coords={"s": s})
        y_da = xr.DataArray(y_line, dims=("s",), coords={"s": s})
        sampled = da.interp(x=x_da, y=y_da, method=method)
        # sampled dims: (t, s) plus any others (but var should be scalar).
        sampled = sampled.transpose("t", "s", ...)
        sampled = sampled.assign_coords(
            x_line=("s", np.asarray(x_line, dtype=float)),
            y_line=("s", np.asarray(y_line, dtype=float)),
            s_phys=("s", np.asarray(s * float(line_lengths[i]), dtype=float)),
        )
        out_list.append(sampled)

    if len(out_list) == 1:
        st = out_list[0]
        # For single-line case, promote s_phys to be the primary coordinate values
        # while keeping the dimension name 's'.
        st = st.assign_coords(s=np.asarray(st["s_phys"].values, dtype=float))
        st.name = "st"
        out = xr.Dataset({"st": st})
        out.attrs = dict(ds.attrs)
        return out

    st_all = xr.concat(out_list, dim="line")
    st_all = st_all.assign_coords(line=np.arange(st_all.sizes["line"], dtype=int))
    st_all.name = "st"
    out = xr.Dataset({"st": st_all})
    out.attrs = dict(ds.attrs)
    return out


def tempcorrf(
    ds: xr.Dataset,
    *,
    variables: list[str] | None = None,
    opt: str = "",
    normalize: bool = False,
) -> xr.Dataset:
    """Temporal correlation function of vector or scalar fields (PIVMAT-inspired).

    For a scalar field ``w(t)``, computes a time-lag correlation
    ``C(T) = < w(t) w(t+T) >`` where the average is taken over space and
    over time pairs.

    For a vector field, returns the sum of the correlations of each component.

    Parameters
    ----------
    ds:
        Input dataset with time dimension ``t``.
    variables:
        Variables to include. If None, defaults to ``['u','v']`` when both
        present, otherwise ``['w']`` if present.
    opt:
        If opt contains ``'0'``, zeros are included as valid values.
        Otherwise (default), zeros are treated as missing and ignored.
    normalize:
        If True, normalizes by the zero-lag value so that ``C(0)=1``.

    Returns
    -------
    xarray.Dataset
        Dataset with coords ``t`` (lag, integer) and variable ``f``.
    """

    if "t" not in ds.dims:
        raise ValueError("tempcorrf requires a time dimension 't'")

    if variables is None:
        if "u" in ds and "v" in ds:
            variables = ["u", "v"]
        elif "w" in ds:
            variables = ["w"]
        else:
            if not ds.data_vars:
                raise ValueError("tempcorrf: dataset has no data variables")
            variables = [next(iter(ds.data_vars))]

    variables = list(variables)
    for name in variables:
        if name not in ds:
            raise KeyError(f"Variable '{name}' not found in dataset")

    include_zeros = "0" in str(opt).lower()

    n = int(ds.sizes.get("t", 0) or 0)
    if n <= 0:
        raise ValueError("Empty time dimension")

    lags = np.arange(n, dtype=int)
    cor = np.zeros(n, dtype=float)

    # Compute correlation per lag, summing over variables.
    # IMPORTANT: use NumPy arrays to avoid xarray coordinate alignment when
    # multiplying time-shifted views (isel preserves time coordinates).
    for k in range(n):
        num_total = 0.0
        den_total = 0.0
        for name in variables:
            da = ds[name]
            a0 = np.asarray(da.isel(t=slice(0, n - k)).data, dtype=float)
            a1 = np.asarray(da.isel(t=slice(k, n)).data, dtype=float)

            finite = np.isfinite(a0) & np.isfinite(a1)
            if not include_zeros:
                finite &= (a0 != 0.0) & (a1 != 0.0)

            prod = a0 * a1
            num = float(np.nansum(np.where(finite, prod, np.nan)))
            den = float(np.sum(finite))
            num_total += num
            den_total += den

        cor[k] = (num_total / den_total) if den_total > 0 else np.nan

    if normalize:
        c0 = cor[0]
        if np.isfinite(c0) and c0 != 0.0:
            cor = cor / c0
        else:
            cor = cor * np.nan

    out = xr.Dataset(
        {
            "f": ("t", cor),
        },
        coords={"t": lags.astype(float)},
    )
    out["t"].attrs["long_name"] = "time lag"
    out.attrs = dict(ds.attrs)
    return out


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


def _corrf_scales(
    r: np.ndarray,
    f: np.ndarray,
    *,
    nowarning: bool = False,
) -> dict[str, float]:
    """Compute integral scales and crossover radii for a 1D correlation curve.

    This follows the PIVMAT `corrf.m` convention:
    - isinf integrates from r=0 to r=r_max using a simple Riemann sum.
    - r0/r1/r2/r5 are the first crossover radii for thresholds 0, 0.1, 0.2, 0.5
      (linearly interpolated between samples).
    - is0/is1/is2/is5 integrate from r=0 up to the first index where the
      threshold is crossed (inclusive), normalized by f(0).
    """

    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    if r.ndim != 1 or f.ndim != 1 or r.shape[0] != f.shape[0]:
        raise ValueError("r and f must be 1D arrays of equal length")
    if r.shape[0] == 0:
        nan = float("nan")
        return {
            "isinf": nan,
            "r0": nan,
            "is0": nan,
            "r1": nan,
            "is1": nan,
            "r2": nan,
            "is2": nan,
            "r5": nan,
            "is5": nan,
        }

    f0 = float(f[0])
    if not np.isfinite(f0) or f0 == 0.0:
        nan = float("nan")
        return {
            "isinf": nan,
            "r0": nan,
            "is0": nan,
            "r1": nan,
            "is1": nan,
            "r2": nan,
            "is2": nan,
            "r5": nan,
            "is5": nan,
        }

    if r.shape[0] >= 2:
        dr = float(r[1] - r[0])
    else:
        dr = 1.0

    def _first_crossing(target: float, *, use_normalized: bool) -> tuple[float, float]:
        if use_normalized:
            series = f / f0
            thr = target
        else:
            series = f
            thr = target

        idx = np.where(series <= thr)[0]
        if idx.size == 0:
            if not nowarning:
                warnings.warn(
                    f"Correlation function does not cross threshold {target}.",
                    UserWarning,
                    stacklevel=2,
                )
            return float("nan"), float("nan")

        k = int(idx[0])
        if k == 0:
            return float(r[0]), 0.0

        # Linear interpolation (PIVMAT-style): use samples k-1 and k.
        r0_ = float(r[k])
        r1_ = float(r[k - 1])
        f0_ = float(series[k])
        f1_ = float(series[k - 1])

        denom = (f0_ - f1_)
        if denom == 0.0:
            rt = r0_
        else:
            rt = r0_ + (thr - f0_) * ((r0_ - r1_) / denom)

        # Integral scale: sum up to index k (inclusive), normalized.
        is_t = float(np.sum((f[: (k + 1)] / f0)) * dr)
        return float(rt), float(is_t)

    isinf = float(np.sum((f / f0)) * dr)
    r0, is0 = _first_crossing(0.0, use_normalized=False)
    r1, is1 = _first_crossing(0.1, use_normalized=True)
    r2, is2 = _first_crossing(0.2, use_normalized=True)
    r5, is5 = _first_crossing(0.5, use_normalized=True)

    return {
        "isinf": isinf,
        "r0": r0,
        "is0": is0,
        "r1": r1,
        "is1": is1,
        "r2": r2,
        "is2": is2,
        "r5": r5,
        "is5": is5,
    }


def corrf(
    x: xr.DataArray,
    dim: int | str = "x",
    *,
    normalize: bool = False,
    nan_as_zero: bool = True,
    nowarning: bool = False,
    r_dim: str = "r",
) -> xr.Dataset:
    """Spatial correlation function and integral scales (PIVMAT-inspired).

    This is a Python/xarray equivalent of PIVMAT's ``corrf.m`` for a scalar field.

    The correlation along a direction is defined (conceptually) as:
    f(r) = < F(x,y) F(x+r,y) >
    where <..> denotes spatial averaging over the orthogonal
    direction and ensemble averaging over any remaining dimensions (e.g. time).

    Parameters
    ----------
    x:
        Scalar field as an ``xarray.DataArray`` (typically with dims including
        ``'x'`` and ``'y'``, and optionally ``'t'``).
    dim:
        Direction of separation: ``'x'``/``'y'`` (recommended) or MATLAB-like
        ``1``/``2``.
    normalize:
        If True, normalize the correlation so that ``f(0)=1``.
    nan_as_zero:
        If True, treat NaNs as missing data and replace by 0 before correlating.
        (Missing values encoded as 0 are handled by the PIVMAT-style weighting in
        ``corrx``/``corrm``.)
    nowarning:
        If True, suppress warnings when crossover radii are undefined.
    r_dim:
        Name of the separation-length coordinate in the returned Dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with 1D variable ``f`` over coordinate ``r`` and scalar
        variables ``isinf, r0, is0, r1, is1, r2, is2, r5, is5``.
    """

    if not isinstance(x, xr.DataArray):
        raise TypeError("corrf expects an xarray.DataArray")

    # Resolve the dimension name for coordinate spacing.
    if isinstance(dim, str):
        dim_name = dim
        if dim_name in ("x", "y") and dim_name not in x.dims:
            # Allow 'x'/'y' mapping only if present; otherwise let corrm raise.
            pass
    else:
        if dim not in (1, 2):
            raise ValueError("dim must be 'x', 'y', 1 or 2")
        if x.ndim < dim:
            raise ValueError(f"Input has only {x.ndim} dims; cannot use dim={dim}.")
        dim_name = x.dims[dim - 1]

    c = corrm(x, dim=dim, half=True, nan_as_zero=nan_as_zero, lag_dim="lag")

    # Average over all dimensions except lag.
    mean_dims = [d for d in c.dims if d != "lag"]
    f_da = c.mean(dim=mean_dims, skipna=True)

    # Separation length: lag index (0..N-1) times grid spacing.
    if dim_name in x.coords and x.sizes.get(dim_name, 0) >= 2:
        coord = x[dim_name].values
        diffs = np.diff(coord.astype(float))
        dr = float(np.abs(diffs[0])) if diffs.size else 1.0
        if diffs.size and not np.allclose(diffs, diffs[0]):
            if not nowarning:
                warnings.warn(
                    f"Non-uniform spacing detected along '{dim_name}'; using first step for dr.",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        dr = 1.0

    lag = f_da["lag"].values.astype(float)
    r = lag * dr
    f = f_da.values.astype(float)

    if normalize and f.size:
        if f[0] != 0.0:
            f = f / float(f[0])

    scales = _corrf_scales(r, f, nowarning=nowarning)

    out = xr.Dataset(coords={r_dim: r})
    out["f"] = (r_dim, f)
    for k, v in scales.items():
        out[k] = xr.DataArray(v)

    out.attrs["dim"] = str(dim)
    out.attrs["dr"] = float(dr)
    out.attrs["normalized"] = bool(normalize)
    out.attrs["variable"] = str(x.name) if x.name is not None else ""

    return out


def gradientf(scalar: xr.DataArray) -> xr.Dataset:
    """Gradient of a scalar field (PIVMAT-inspired).

    This is a Python/xarray equivalent of PIVMAT's ``gradientf.m``.

    Parameters
    ----------
    scalar:
        Scalar field as an ``xarray.DataArray`` with dims including ``x`` and ``y``.
        A time dimension (typically ``t``) is allowed and is preserved.

    Returns
    -------
    xarray.Dataset
        Dataset with variables ``u`` and ``v`` containing the partial derivatives
        ``d(scalar)/dx`` and ``d(scalar)/dy``.

    Notes
    -----
    - Coordinates are taken from the input DataArray.
    - Units are propagated if both the scalar and coordinate units are available.
    """

    if not isinstance(scalar, xr.DataArray):
        raise TypeError("gradientf expects an xarray.DataArray")

    if "x" not in scalar.dims or "y" not in scalar.dims:
        raise ValueError("gradientf requires dims 'x' and 'y'")

    gx = scalar.differentiate("x")
    gy = scalar.differentiate("y")

    out = xr.Dataset({"u": gx, "v": gy})

    # Best-effort metadata propagation.
    w_units = str(scalar.attrs.get("units", ""))
    x_units = str(getattr(scalar.coords.get("x", None), "attrs", {}).get("units", ""))
    y_units = str(getattr(scalar.coords.get("y", None), "attrs", {}).get("units", ""))

    name = scalar.name or "scalar"
    out["u"].attrs = dict(gx.attrs)
    out["v"].attrs = dict(gy.attrs)
    out["u"].attrs.setdefault("long_name", f"d/dx({name})")
    out["v"].attrs.setdefault("long_name", f"d/dy({name})")

    if w_units and x_units:
        out["u"].attrs["units"] = f"{w_units}/{x_units}"
    if w_units and y_units:
        out["v"].attrs["units"] = f"{w_units}/{y_units}"

    return out


def _hist_counts(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Histogram counts using bin centers (MATLAB hist-like behavior)."""

    c = np.asarray(centers, dtype=float).ravel()
    if c.size == 0:
        return np.asarray([], dtype=int)

    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(c.size, dtype=int)

    # Use midpoints between centers as bin edges, with infinite end caps.
    mids = 0.5 * (c[:-1] + c[1:])
    edges = np.concatenate(([-np.inf], mids, [np.inf]))
    h, _ = np.histogram(v, bins=edges)
    return np.asarray(h, dtype=int)


def histf(
    scalar: xr.DataArray,
    bin: np.ndarray | None = None,
    opt: str = "",
) -> xr.Dataset:
    """Histogram of a scalar field (PIVMAT-inspired).

    This is a Python/xarray equivalent of PIVMAT's ``histf.m`` for scalar
    fields. Values are stacked over all dimensions.

    Parameters
    ----------
    scalar:
        Scalar field as an ``xarray.DataArray``.
    bin:
        Optional 1D array of bin centers. If not provided, a default binning is
        estimated from the mean and standard deviation of the first frame
        (if a ``t`` dimension exists) or from the full field otherwise.
    opt:
        Option string. If it contains ``'0'``, zero values are included.

    Returns
    -------
    xarray.Dataset
        Dataset with coordinate ``bin`` and variable ``h``.
    """

    if not isinstance(scalar, xr.DataArray):
        raise TypeError("histf expects an xarray.DataArray")

    include_zeros = "0" in str(opt)

    da0 = scalar
    if "t" in scalar.dims:
        try:
            da0 = scalar.isel(t=0)
        except Exception:
            da0 = scalar

    ref = np.asarray(da0.values, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    if not include_zeros:
        ref = ref[ref != 0]

    if bin is None:
        if ref.size == 0:
            centers = np.linspace(-1.0, 1.0, 200)
        else:
            mean = float(np.mean(ref))
            std = float(np.std(ref))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            if mean < std:
                centers = np.linspace(-20.0 * std, 20.0 * std, 200)
            else:
                centers = np.linspace(mean - 20.0 * std, mean + 20.0 * std, 200)
    else:
        centers = np.asarray(bin, dtype=float).ravel()

    vals = np.asarray(scalar.values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if not include_zeros:
        vals = vals[vals != 0]

    h = _hist_counts(vals, centers)
    out = xr.Dataset({"h": ("bin", h)}, coords={"bin": centers})
    out["h"].attrs["long_name"] = "histogram"
    return out


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


def inpaint_missing_2d(
    a2: ArrayLike,
    *,
    method: int = 0,
    missing: str = "0nan",
) -> np.ndarray:
    """Inpaint missing values in a 2D array (PIVMAT ``interpf``-style).

    Missing values are defined as NaNs and/or zeros.

    Parameters
    ----------
    a2:
        2D array.
    method:
        Integer method selector (PIVMAT-inspired):

        - ``0``: Laplacian (harmonic) inpainting via sparse linear solve.
        - ``1``: Nearest-neighbor fill (fast, robust).
        - ``2``: Linear interpolation via ``scipy.interpolate.griddata``.
    missing:
        Missing-value definition:
        - ``"0nan"`` (default): treat both ``0`` and ``NaN`` as missing.
        - ``"nan"``: treat only NaNs as missing.
        - ``"0"``: treat only zeros as missing.

    Returns
    -------
    numpy.ndarray
        Filled array (float).
    """

    a = np.asarray(a2, dtype=float)
    if a.ndim != 2:
        raise ValueError("inpaint_missing_2d expects a 2D array")

    missing_l = str(missing).lower()
    if missing_l not in {"0nan", "nan", "0"}:
        raise ValueError("missing must be one of: '0nan', 'nan', '0'")

    mask_nan = ~np.isfinite(a)
    mask_zero = a == 0.0
    if missing_l == "0nan":
        miss = mask_nan | mask_zero
    elif missing_l == "nan":
        miss = mask_nan
    else:
        miss = mask_zero

    if not np.any(miss):
        return a

    # If everything is missing, return zeros to match common PIV conventions.
    if np.all(miss):
        return np.zeros_like(a, dtype=float)

    m = int(method)
    if m == 1:
        # Nearest-neighbor fill via distance transform.
        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("method=1 requires SciPy (scipy.ndimage.distance_transform_edt)") from exc

        valid = ~miss
        # distance_transform_edt expects False for features; compute indices of nearest valid.
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        out = a.copy()
        out[miss] = out[iy[miss], ix[miss]]
        out[~np.isfinite(out)] = 0.0
        return out

    if m == 2:
        try:
            from scipy.interpolate import griddata  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("method=2 requires SciPy (scipy.interpolate.griddata)") from exc

        ny, nx = a.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        pts = np.column_stack([yy[~miss].ravel(), xx[~miss].ravel()])
        vals = a[~miss].ravel()
        xi = (yy[miss], xx[miss])
        out = a.copy()
        filled = griddata(pts, vals, xi, method="linear")
        # griddata returns NaN outside convex hull; fall back to nearest for those.
        if np.any(~np.isfinite(filled)):
            filled2 = griddata(pts, vals, xi, method="nearest")
            filled = np.where(np.isfinite(filled), filled, filled2)
        out[miss] = filled
        out[~np.isfinite(out)] = 0.0
        return out

    if m != 0:
        raise ValueError("Unsupported method. Supported: 0, 1, 2")

    # Method 0: solve Laplace equation on missing nodes with Dirichlet boundary on known nodes.
    try:
        from scipy.sparse import lil_matrix  # type: ignore
        from scipy.sparse.linalg import spsolve  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("method=0 requires SciPy sparse (scipy.sparse, scipy.sparse.linalg)") from exc

    ny, nx = a.shape
    idx = -np.ones((ny, nx), dtype=int)
    unknown_positions = np.argwhere(miss)
    n_unknown = int(unknown_positions.shape[0])
    for k, (iy, ix) in enumerate(unknown_positions):
        idx[iy, ix] = k

    A = lil_matrix((n_unknown, n_unknown), dtype=float)
    b = np.zeros(n_unknown, dtype=float)

    # 4-neighbor Laplacian stencil; adjust at borders.
    for k, (iy, ix) in enumerate(unknown_positions):
        coeff_center = 0.0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y2 = iy + dy
            x2 = ix + dx
            if y2 < 0 or y2 >= ny or x2 < 0 or x2 >= nx:
                continue
            coeff_center += 1.0
            if miss[y2, x2]:
                A[k, idx[y2, x2]] = -1.0
            else:
                b[k] += float(a[y2, x2])
        A[k, k] = float(coeff_center)

    sol = spsolve(A.tocsr(), b)
    out = a.copy()
    out[miss] = sol
    out[~np.isfinite(out)] = 0.0
    return out


def interpf(
    data: xr.Dataset,
    *,
    method: int = 0,
    variables: list[str] | None = None,
    missing: str = "0nan",
) -> xr.Dataset:
    """Interpolate missing data in a Dataset (PIVMAT ``interpf`` port).

    Missing data are values equal to 0 and/or NaN (configurable via ``missing``).
    The interpolation is applied frame-by-frame along ``t`` if present.

    Parameters
    ----------
    data:
        Input Dataset.
    method:
        See :func:`inpaint_missing_2d`.
    variables:
        Variables to process. Default: ['u','v'] if present, else ['w'] if present,
        else all data variables.
    missing:
        See :func:`inpaint_missing_2d`.

    Returns
    -------
    xarray.Dataset
        New Dataset with missing values filled.
    """

    ds = data
    if variables is None:
        if "u" in ds.data_vars and "v" in ds.data_vars:
            variables = ["u", "v"]
        elif "w" in ds.data_vars:
            variables = ["w"]
        else:
            variables = list(ds.data_vars)

    out = ds.copy(deep=True)
    for name in variables:
        if name not in out.data_vars:
            raise KeyError(f"Variable {name} not found in dataset")
        da = out[name]
        if da.ndim < 2:
            continue

        # Determine the 2D core dims (y,x) and keep remaining dims vectorized.
        y_dim, x_dim = da.dims[0], da.dims[1]

        def _core(arr2: np.ndarray) -> np.ndarray:
            return inpaint_missing_2d(arr2, method=method, missing=missing)

        filled = xr.apply_ufunc(
            _core,
            da,
            input_core_dims=[[y_dim, x_dim]],
            output_core_dims=[[y_dim, x_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # apply_ufunc may reorder non-core dims; restore original dim order.
        try:
            filled = filled.transpose(*da.dims)
        except Exception:
            pass
        filled = filled.assign_coords({y_dim: da[y_dim], x_dim: da[x_dim]})
        filled.attrs = dict(ds[name].attrs)
        out[name] = filled

    out.attrs = dict(ds.attrs)
    return out


def jpdfscal(
    s1: xr.DataArray,
    s2: xr.DataArray,
    *,
    nbin: int = 101,
) -> xr.Dataset:
    """Joint histogram ("joint PDF" in PIVMAT terminology) of two scalar fields.

    This ports the behavior of PIVMAT's ``jpdfscal``. The output is a 2D count
    matrix over symmetric bin centers spanning ``[-max(abs(s)), +max(abs(s))]``
    for each scalar.

    Parameters
    ----------
    s1, s2:
        Scalar fields as DataArrays. They must be broadcastable to the same
        shape; non-finite pairs are ignored.
    nbin:
        Number of bin centers per axis (default: 101).

    Returns
    -------
    xarray.Dataset
        Dataset with coordinates ``bin1`` and ``bin2`` and a 2D variable ``hi``
        containing counts.
    """

    if not isinstance(s1, xr.DataArray) or not isinstance(s2, xr.DataArray):
        raise TypeError("jpdfscal expects two xarray.DataArray inputs")

    n = int(nbin)
    if n < 3 or n % 2 == 0:
        # PIVMAT uses odd default (101). Odd makes the center bin land at 0.
        raise ValueError("nbin must be an odd integer >= 3")

    a1, a2 = xr.align(s1, s2, join="exact")
    v1 = np.asarray(a1.values, dtype=float).ravel()
    v2 = np.asarray(a2.values, dtype=float).ravel()

    finite = np.isfinite(v1) & np.isfinite(v2)
    v1 = v1[finite]
    v2 = v2[finite]

    if v1.size == 0:
        max1 = 0.0
    else:
        max1 = float(np.max(np.abs(v1)))
    if v2.size == 0:
        max2 = 0.0
    else:
        max2 = float(np.max(np.abs(v2)))

    bin1 = np.linspace(-max1, max1, n, dtype=float) if max1 > 0 else np.linspace(-1.0, 1.0, n, dtype=float)
    bin2 = np.linspace(-max2, max2, n, dtype=float) if max2 > 0 else np.linspace(-1.0, 1.0, n, dtype=float)

    rg = (n - 1) / 2.0

    def _to_index(v: np.ndarray, vmax: float) -> np.ndarray:
        if vmax <= 0.0 or not np.isfinite(vmax):
            return np.full_like(v, int(rg), dtype=int)
        idx = np.rint(rg * (1.0 + (v / vmax))).astype(int)
        return np.clip(idx, 0, n - 1)

    i1 = _to_index(v1, max1)
    i2 = _to_index(v2, max2)

    hi = np.zeros((n, n), dtype=float)
    # Vectorized 2D bincount
    flat = i1 * n + i2
    bc = np.bincount(flat, minlength=n * n)
    hi[:, :] = bc.reshape((n, n))

    ds = xr.Dataset(
        data_vars={"hi": (("bin1", "bin2"), hi)},
        coords={"bin1": ("bin1", bin1), "bin2": ("bin2", bin2)},
    )

    ds["hi"].attrs["long_name"] = "joint histogram"
    # Carry basic metadata if present.
    ds.attrs["namew1"] = str(a1.attrs.get("long_name", a1.name or "s1"))
    ds.attrs["namew2"] = str(a2.attrs.get("long_name", a2.name or "s2"))
    ds.attrs["unitw1"] = str(a1.attrs.get("units", ""))
    ds.attrs["unitw2"] = str(a2.attrs.get("units", ""))
    return ds


# -----------------------------------------------------------------------------
# PIVMAT-inspired spectral / structure-function methods
# -----------------------------------------------------------------------------


def _as_frames(obj: xr.Dataset | xr.DataArray) -> list[xr.Dataset | xr.DataArray]:
    """Normalize input into a list of 2D fields (frame-wise)."""

    if isinstance(obj, xr.Dataset) and "t" in obj.dims:
        return [obj.isel(t=i) for i in range(int(obj.sizes["t"]))]
    if isinstance(obj, xr.DataArray) and "t" in obj.dims:
        return [obj.isel(t=i) for i in range(int(obj.sizes["t"]))]
    return [obj]


def _hann(n: int) -> np.ndarray:
    # MATLAB hann(n) matches numpy.hanning(n)
    return np.hanning(int(n))


def _kx_ky_from_coords(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    dx = float(np.abs(x[1] - x[0])) if x.size >= 2 else 1.0
    dy = float(np.abs(y[1] - y[0])) if y.size >= 2 else 1.0
    nkx = nx // 2
    nky = ny // 2
    kx = np.linspace(0.0, np.pi * (1.0 - 1.0 / nkx), nkx) / dx
    ky = np.linspace(0.0, np.pi * (1.0 - 1.0 / nky), nky) / dy
    return kx, ky


def _bin_centers_to_edges(bin_centers: np.ndarray) -> np.ndarray:
    b = np.asarray(bin_centers, dtype=float)
    if b.size < 2:
        raise ValueError("bin must have at least 2 entries")
    w = float(np.abs(b[1] - b[0]))
    edges = np.concatenate([b - w / 2.0, [b[-1] + w / 2.0]])
    return edges


def _azimuthal_average_square(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average for a square 2D matrix around its center.

    Returns a (k_index, avg) where k_index is in mesh-index units.
    """

    a = np.asarray(mat, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Azimuthal average requires a square 2D array")
    n = a.shape[0]
    # radius in index space relative to center
    c = (n - 1) / 2.0
    yy, xx = np.indices((n, n), dtype=float)
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    r_int = np.rint(rr).astype(int)
    r_max = int(n // 2)

    out = np.zeros(r_max + 1, dtype=float)
    cnt = np.zeros(r_max + 1, dtype=float)
    valid = np.isfinite(a)
    for r in range(r_max + 1):
        mask = (r_int == r) & valid
        if np.any(mask):
            out[r] = float(np.mean(a[mask]))
            cnt[r] = float(np.sum(mask))
    k = np.arange(r_max + 1, dtype=float)
    return k, out


def specf(f: xr.Dataset | xr.DataArray, *opts: str) -> xr.Dataset:
    """1D power spectrum of vector/scalar fields (PIVMAT-inspired).

    This follows PIVMAT's conventions:
    - Requires even x/y sizes (drops last row/col if odd)
    - Optional Hann apodization via the option ``'hann'``
    - Normalization such that \n\n
    $$\\int E(k)\\,dk \approx \\langle s^2 \\rangle$$

    Returns an :class:`xarray.Dataset` with coordinates ``kx`` and ``ky``.
    For vector fields, returns ``exvx, exvy, eyvx, eyvy``.
    For scalar fields, returns ``ex, ey``.
    If the field is square, also returns isotropic components ``k`` and ``e``.
    """

    use_hann = any(str(o).lower().startswith("hann") for o in opts)

    frames = _as_frames(f)
    first = frames[0]

    if isinstance(first, xr.Dataset):
        is_vector = ("u" in first.data_vars and "v" in first.data_vars)
        is_scalar = ("w" in first.data_vars) and not is_vector
    else:
        # DataArray treated as scalar
        is_vector = False
        is_scalar = True

    # Extract coords and sizes
    if isinstance(first, xr.Dataset):
        x = np.asarray(first["x"].values, dtype=float)
        y = np.asarray(first["y"].values, dtype=float)
        ny = int(first.sizes["y"])
        nx = int(first.sizes["x"])
    else:
        # DataArray
        if "x" not in first.dims or "y" not in first.dims:
            raise ValueError("specf expects DataArray with dims ('y','x')")
        x = np.asarray(first["x"].values, dtype=float)
        y = np.asarray(first["y"].values, dtype=float)
        ny = int(first.sizes["y"])
        nx = int(first.sizes["x"])

    # Enforce even sizes (PIVMAT behavior)
    nx2 = nx - (nx % 2)
    ny2 = ny - (ny % 2)
    if nx2 != nx or ny2 != ny:
        frames2: list[xr.Dataset | xr.DataArray] = []
        for fr in frames:
            frames2.append(fr.isel(x=slice(0, nx2), y=slice(0, ny2)))
        frames = frames2
        nx, ny = nx2, ny2
        x = x[:nx]
        y = y[:ny]

    nkx = nx // 2
    nky = ny // 2
    kx, ky = _kx_ky_from_coords(x, y, nx, ny)

    # Hann windows
    if use_hann:
        hann_x = _hann(nx)[None, :]  # along x
        hann_y = _hann(ny)[:, None]  # along y

    def _fft_power_x(a2: np.ndarray) -> np.ndarray:
        # mean(|fft(a, axis=x)|^2, over y)
        aa = a2
        if use_hann:
            aa = aa * hann_x
        fx = np.fft.fft(aa, axis=1)
        return np.mean(np.abs(fx) ** 2, axis=0)

    def _fft_power_y(a2: np.ndarray) -> np.ndarray:
        aa = a2
        if use_hann:
            aa = aa * hann_y
        fy = np.fft.fft(aa, axis=0)
        return np.mean(np.abs(fy) ** 2, axis=1)

    if is_vector:
        exvx_i = []
        exvy_i = []
        eyvx_i = []
        eyvy_i = []
        for fr in frames:
            assert isinstance(fr, xr.Dataset)
            u2 = np.asarray(fr["u"].values, dtype=float)
            v2 = np.asarray(fr["v"].values, dtype=float)
            exvx_i.append(_fft_power_x(u2))
            exvy_i.append(_fft_power_x(v2))
            eyvx_i.append(_fft_power_y(u2))
            eyvy_i.append(_fft_power_y(v2))

        exvx = np.mean(np.stack(exvx_i, axis=0), axis=0)
        exvy = np.mean(np.stack(exvy_i, axis=0), axis=0)
        eyvx = np.mean(np.stack(eyvx_i, axis=0), axis=0)
        eyvy = np.mean(np.stack(eyvy_i, axis=0), axis=0)

        # PIVMAT normalization
        dkx = float(kx[1]) if kx.size >= 2 else 1.0
        dky = float(ky[1]) if ky.size >= 2 else 1.0
        exvx = 2.0 * exvx[:nkx] / (nx * ny) / dkx
        exvy = 2.0 * exvy[:nkx] / (nx * ny) / dkx
        eyvx = 2.0 * eyvx[:nky] / (nx * ny) / dky
        eyvy = 2.0 * eyvy[:nky] / (nx * ny) / dky
        exvx[0] /= 2.0
        exvy[0] /= 2.0
        eyvx[0] /= 2.0
        eyvy[0] /= 2.0

        ds = xr.Dataset(
            data_vars={
                "exvx": ("kx", exvx),
                "exvy": ("kx", exvy),
                "eyvx": ("ky", eyvx),
                "eyvy": ("ky", eyvy),
            },
            coords={"kx": ("kx", kx), "ky": ("ky", ky)},
        )
        ds.attrs["appod"] = "Hann" if use_hann else "None"
    else:
        # scalar
        ex_i = []
        ey_i = []
        for fr in frames:
            if isinstance(fr, xr.Dataset):
                if "w" not in fr.data_vars:
                    raise ValueError("specf scalar path expects variable 'w'")
                a2 = np.asarray(fr["w"].values, dtype=float)
            else:
                a2 = np.asarray(fr.values, dtype=float)
            ex_i.append(_fft_power_x(a2))
            ey_i.append(_fft_power_y(a2))
        ex = np.mean(np.stack(ex_i, axis=0), axis=0)
        ey = np.mean(np.stack(ey_i, axis=0), axis=0)

        dkx = float(kx[1]) if kx.size >= 2 else 1.0
        dky = float(ky[1]) if ky.size >= 2 else 1.0
        ex = 2.0 * ex[:nkx] / (nx * ny) / dkx
        ey = 2.0 * ey[:nky] / (nx * ny) / dky
        ex[0] /= 2.0
        ey[0] /= 2.0

        ds = xr.Dataset(data_vars={"ex": ("kx", ex), "ey": ("ky", ey)}, coords={"kx": ("kx", kx), "ky": ("ky", ky)})
        ds.attrs["appod"] = "Hann" if use_hann else "None"

    # isotropic spectrum for square domain
    # NOTE: xarray will broadcast (kx + ky) into 2D if we add DataArrays
    # with different dimension names. Keep this explicitly 1D.
    if nx == ny:
        k = 0.5 * (np.asarray(ds["kx"].values, dtype=float) + np.asarray(ds["ky"].values, dtype=float))
        ds = ds.assign_coords({"k": ("k", k)})
        if is_vector:
            el = 0.5 * (np.asarray(ds["exvx"].values, dtype=float) + np.asarray(ds["eyvy"].values, dtype=float))
            et = 0.5 * (np.asarray(ds["exvy"].values, dtype=float) + np.asarray(ds["eyvx"].values, dtype=float))
            e = el + et
            ds["el"] = ("k", el)
            ds["et"] = ("k", et)
            ds["e"] = ("k", e)
        else:
            e = 0.5 * (np.asarray(ds["ex"].values, dtype=float) + np.asarray(ds["ey"].values, dtype=float))
            ds["e"] = ("k", e)

    return ds


def spec2f(f: xr.Dataset | xr.DataArray, *opts: str) -> xr.Dataset:
    """2D power spectrum (PIVMAT-inspired)."""

    use_hann = any(str(o).lower().startswith("hann") for o in opts)
    frames = _as_frames(f)
    first = frames[0]

    if isinstance(first, xr.Dataset):
        is_vector = ("u" in first.data_vars and "v" in first.data_vars)
        is_scalar = ("w" in first.data_vars) and not is_vector
        x = np.asarray(first["x"].values, dtype=float)
        y = np.asarray(first["y"].values, dtype=float)
        ny = int(first.sizes["y"])
        nx = int(first.sizes["x"])
    else:
        is_vector = False
        is_scalar = True
        x = np.asarray(first["x"].values, dtype=float)
        y = np.asarray(first["y"].values, dtype=float)
        ny = int(first.sizes["y"])
        nx = int(first.sizes["x"])

    nx2 = nx - (nx % 2)
    ny2 = ny - (ny % 2)
    if nx2 != nx or ny2 != ny:
        frames = [fr.isel(x=slice(0, nx2), y=slice(0, ny2)) for fr in frames]
        nx, ny = nx2, ny2
        x = x[:nx]
        y = y[:ny]

    nkx = nx // 2
    nky = ny // 2
    kx, ky = _kx_ky_from_coords(x, y, nx, ny)
    dkx = float(kx[1]) if kx.size >= 2 else 1.0
    dky = float(ky[1]) if ky.size >= 2 else 1.0

    if use_hann:
        hann_x = _hann(nx)[None, :]
        hann_y = _hann(ny)[:, None]

    def _spec2(a2: np.ndarray) -> np.ndarray:
        aa = a2
        if use_hann:
            aa = (aa * hann_x) * hann_y
        # Full shifted spectrum is (ny, nx). PIVMAT-style outputs are typically
        # reported on the positive (one-sided) wavenumber grid only.
        ft = np.fft.fftshift(np.abs(np.fft.fft2(aa)) ** 2)
        return ft[ny // 2 : ny, nx // 2 : nx]

    if is_vector:
        ex_i = []
        ey_i = []
        for fr in frames:
            assert isinstance(fr, xr.Dataset)
            ex_i.append(_spec2(np.asarray(fr["u"].values, dtype=float)))
            ey_i.append(_spec2(np.asarray(fr["v"].values, dtype=float)))
        ex = np.mean(np.stack(ex_i, axis=0), axis=0)
        ey = np.mean(np.stack(ey_i, axis=0), axis=0)

        ex = ex / (nx * ny) ** 2 / dkx
        ey = ey / (nx * ny) ** 2 / dky
        e = ex + ey
        ds = xr.Dataset(
            data_vars={"ex": (("ky", "kx"), ex), "ey": (("ky", "kx"), ey), "e": (("ky", "kx"), e)},
            coords={"kx": ("kx", kx), "ky": ("ky", ky)},
        )
        ds.attrs["appod"] = "Hann" if use_hann else "None"
    else:
        if not is_scalar:
            raise ValueError("spec2f expects a scalar DataArray or a Dataset with 'w'")
        e_i = []
        for fr in frames:
            if isinstance(fr, xr.Dataset):
                a2 = np.asarray(fr["w"].values, dtype=float)
            else:
                a2 = np.asarray(fr.values, dtype=float)
            e_i.append(_spec2(a2))
        e = np.mean(np.stack(e_i, axis=0), axis=0)
        e = e / (nx * ny) ** 2 / dkx
        ds = xr.Dataset(data_vars={"e": (("ky", "kx"), e)}, coords={"kx": ("kx", kx), "ky": ("ky", ky)})
        ds.attrs["appod"] = "Hann" if use_hann else "None"

    # Azimuthal average for square domains
    if nx == ny:
        kk, ep = _azimuthal_average_square(np.asarray(ds["e"].values))
        ds["k"] = ("k", kk * float(np.abs(kx[1] - kx[0])))
        ds["ep"] = ("k", ep)

    return ds


def tempspecf(v: xr.Dataset | xr.DataArray, freq: float = 1.0, *opts: str) -> xr.Dataset:
    """Temporal power spectrum averaged over space (PIVMAT-inspired)."""

    if isinstance(v, xr.Dataset):
        if "t" not in v.dims:
            raise ValueError("tempspecf expects a time series Dataset with dim 't'")
        is_vector = ("u" in v and "v" in v)
        is_scalar = ("w" in v) and not is_vector
    else:
        if "t" not in v.dims:
            raise ValueError("tempspecf expects a DataArray with dim 't'")
        is_vector = False
        is_scalar = True

    if int(v.sizes["t"]) < 4:
        raise ValueError("Sample size too small")

    use_hann = any(str(o).lower().startswith("hann") for o in opts)
    include_zero = any(str(o).lower().startswith("zero") for o in opts)
    doublex = any(str(o).lower().startswith("doublex") for o in opts)
    doubley = any(str(o).lower().startswith("doubley") for o in opts)

    nt = int(v.sizes["t"])
    # match PIVMAT: length floor(nt/2) (exclude DC)
    nfreq = nt // 2
    f_hz = (np.arange(1, nfreq + 1, dtype=float) * float(freq) / float(nt))
    w = 2.0 * np.pi * f_hz
    df = float(freq) / float(nt)

    win = np.ones(nt, dtype=float)
    if use_hann:
        win = _hann(nt)

    def _one_series_psd(x: np.ndarray) -> np.ndarray | None:
        x = np.asarray(x, dtype=float)
        if not include_zero:
            if np.any(x == 0.0) or np.any(~np.isfinite(x)):
                return None
        if np.any(~np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0)
        x = x - float(np.mean(x))
        x = x * win
        X = np.fft.rfft(x)
        # one-sided density (exclude DC at k=0)
        # Parseval: mean(x^2) ~= sum(S(f))*df
        S = (2.0 * (np.abs(X[1 : nfreq + 1]) ** 2)) / (nt * nt) / df
        # Nyquist term (if present) should not be doubled
        if nt % 2 == 0:
            S[-1] /= 2.0
        return S

    if isinstance(v, xr.Dataset) and is_vector:
        u = np.asarray(v["u"].values, dtype=float)
        vv = np.asarray(v["v"].values, dtype=float)
        # shapes: (y,x,t)
        # iterate all points
        acc = np.zeros(nfreq, dtype=float)
        nnz = 0
        for iy in range(u.shape[0]):
            for ix in range(u.shape[1]):
                su = _one_series_psd(u[iy, ix, :])
                sv = _one_series_psd(vv[iy, ix, :])
                if su is None or sv is None:
                    continue
                if doublex:
                    acc += 2.0 * su + sv
                elif doubley:
                    acc += su + 2.0 * sv
                else:
                    acc += su + sv
                nnz += 1
        etot = acc / float(nnz if nnz else 1)
    else:
        # scalar
        if isinstance(v, xr.Dataset):
            a = np.asarray(v["w"].values, dtype=float)
        else:
            a = np.asarray(v.values, dtype=float)
        acc = np.zeros(nfreq, dtype=float)
        nnz = 0
        for iy in range(a.shape[0]):
            for ix in range(a.shape[1]):
                s = _one_series_psd(a[iy, ix, :])
                if s is None:
                    continue
                acc += s
                nnz += 1
        etot = acc / float(nnz if nnz else 1)

    # Return density vs w (rad/s) to match the signature.
    # Convert from per-Hz to per-(rad/s): E(w) = E(f) / (2*pi)
    etot_w = etot / (2.0 * np.pi)
    return xr.Dataset(data_vars={"e": ("w", etot_w)}, coords={"w": ("w", w)})


def ssf(s: xr.Dataset, dim: int | str = 1, *opts: str) -> xr.Dataset:
    """Structure functions of a scalar field (PIVMAT-inspired)."""

    if "w" not in s:
        raise ValueError("ssf expects a scalar Dataset with variable 'w'")

    # options
    include_zero = any(str(o) == "0" for o in opts)
    maxorder = 4
    if any(str(o).lower().startswith("maxorder") for o in opts):
        # accept ('maxorder', N) style
        try:
            idx = [i for i, o in enumerate(opts) if str(o).lower().startswith("maxorder")][-1]
            maxorder = max(4, int(opts[idx + 1]))
        except Exception:
            raise ValueError("ssf: expected integer after 'maxorder'")
    if maxorder > 30:
        raise ValueError("Maximum order too large")

    if isinstance(dim, str):
        dim_l = dim.lower()
        if dim_l == "x":
            dim_i = 1
        elif dim_l == "y":
            dim_i = 2
        else:
            raise ValueError("dim must be 1,2,'x','y'")
    else:
        dim_i = int(dim)

    w0 = s["w"].isel(t=0) if "t" in s.dims else s["w"]
    rms = float(np.nanstd(w0.values))
    bin_centers = None
    if any(str(o).lower().startswith("bin") for o in opts):
        idx = [i for i, o in enumerate(opts) if str(o).lower().startswith("bin")][-1]
        try:
            bin_centers = np.asarray(opts[idx + 1], dtype=float)
        except Exception:
            raise ValueError("ssf: expected a numeric bin vector after 'bin'")
    if bin_centers is None:
        maxbin = 10.0 * rms
        bin_centers = np.linspace(-maxbin, maxbin, 1000)
    binwidth = float(np.abs(bin_centers[1] - bin_centers[0]))
    edges = _bin_centers_to_edges(bin_centers)

    r_list = None
    if any(str(o).lower() == "r" for o in opts):
        idx = [i for i, o in enumerate(opts) if str(o).lower() == "r"][-1]
        r_list = np.asarray(opts[idx + 1], dtype=int)
    if r_list is None:
        default_r = np.asarray(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 112, 128, 142, 160, 176, 192, 224, 256],
            dtype=int,
        )
        maxdr = min(int(s.sizes["x"]), int(s.sizes["y"]))
        r_list = default_r[default_r < maxdr]

    frames = _as_frames(s)
    hsi = np.zeros((r_list.size, bin_centers.size), dtype=float)

    for fr in frames:
        assert isinstance(fr, xr.Dataset)
        a = np.asarray(fr["w"].values, dtype=float)
        nx = a.shape[1]
        ny = a.shape[0]
        for ir, rr in enumerate(r_list):
            if dim_i == 1:
                if rr >= nx:
                    continue
                dsx = a[:, rr:] - a[:, : nx - rr]
                if not include_zero:
                    valid = (a[:, rr:] != 0) & (a[:, : nx - rr] != 0) & np.isfinite(dsx)
                    dsx = np.where(valid, dsx, 0.0)
                vals = dsx.ravel()
            else:
                if rr >= ny:
                    continue
                dsy = a[rr:, :] - a[: ny - rr, :]
                if not include_zero:
                    valid = (a[rr:, :] != 0) & (a[: ny - rr, :] != 0) & np.isfinite(dsy)
                    dsy = np.where(valid, dsy, 0.0)
                vals = dsy.ravel()
            vals = vals[(vals != 0.0) & np.isfinite(vals)]
            if vals.size:
                hist, _ = np.histogram(vals, bins=edges)
                hsi[ir, :] += hist

    pdfsi = np.zeros_like(hsi)
    sf = np.zeros((r_list.size, maxorder), dtype=float)
    sfabs = np.zeros((r_list.size, maxorder), dtype=float)
    skew = np.zeros(r_list.size, dtype=float)
    flat = np.zeros(r_list.size, dtype=float)
    n_used = np.zeros(r_list.size, dtype=float)

    for ir in range(r_list.size):
        nsi = float(np.sum(hsi[ir, :]))
        n_used[ir] = nsi
        if nsi > 0:
            pdfsi[ir, :] = hsi[ir, :] / (nsi * binwidth)
        for order in range(1, maxorder + 1):
            sf[ir, order - 1] = float(np.sum(pdfsi[ir, :] * (bin_centers**order)) * binwidth)
            sfabs[ir, order - 1] = float(np.sum(pdfsi[ir, :] * (np.abs(bin_centers) ** order)) * binwidth)
        if sf[ir, 1] != 0:
            skew[ir] = sf[ir, 2] / (sf[ir, 1] ** 1.5)
            flat[ir] = sf[ir, 3] / (sf[ir, 1] ** 2)

    scaler = float(np.abs(s["x"].values[1] - s["x"].values[0])) if s["x"].size >= 2 else 1.0

    return xr.Dataset(
        data_vars={
            "hsi": (("r", "bin"), hsi),
            "pdfsi": (("r", "bin"), pdfsi),
            "sf": (("r", "order"), sf),
            "sfabs": (("r", "order"), sfabs),
            "skew": ("r", skew),
            "flat": ("r", flat),
            "n": ("r", n_used),
        },
        coords={
            "r": ("r", r_list.astype(float)),
            "bin": ("bin", bin_centers),
            "order": ("order", np.arange(1, maxorder + 1, dtype=int)),
        },
        attrs={"scaler": scaler, "binwidth": binwidth},
    )


def vsf(v: xr.Dataset, *opts: str) -> xr.Dataset:
    """Structure functions of a vector field (PIVMAT-inspired)."""

    if "u" not in v or "v" not in v:
        raise ValueError("vsf expects a vector Dataset with variables 'u' and 'v'")

    include_zero = any(str(o) == "0" for o in opts)
    maxorder = 4
    if any(str(o).lower().startswith("maxorder") for o in opts):
        idx = [i for i, o in enumerate(opts) if str(o).lower().startswith("maxorder")][-1]
        try:
            maxorder = max(4, int(opts[idx + 1]))
        except Exception:
            raise ValueError("vsf: expected integer after 'maxorder'")
    if maxorder > 30:
        raise ValueError("Maximum order too large")

    # default bins based on rms of u component
    u0 = v["u"].isel(t=0) if "t" in v.dims else v["u"]
    rms = float(np.nanstd(u0.values))
    bin_centers = None
    if any(str(o).lower().startswith("bin") for o in opts):
        idx = [i for i, o in enumerate(opts) if str(o).lower().startswith("bin")][-1]
        try:
            bin_centers = np.asarray(opts[idx + 1], dtype=float)
        except Exception:
            raise ValueError("vsf: expected a numeric bin vector after 'bin'")
    if bin_centers is None:
        maxbin = 10.0 * rms
        bin_centers = np.linspace(-maxbin, maxbin, 1000)
    binwidth = float(np.abs(bin_centers[1] - bin_centers[0]))
    edges = _bin_centers_to_edges(bin_centers)

    r_list = None
    if any(str(o).lower() == "r" for o in opts):
        idx = [i for i, o in enumerate(opts) if str(o).lower() == "r"][-1]
        r_list = np.asarray(opts[idx + 1], dtype=int)
    if r_list is None:
        default_r = np.asarray(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 112, 128, 142, 160, 176, 192, 224, 256],
            dtype=int,
        )
        maxdr = min(int(v.sizes["x"]), int(v.sizes["y"]))
        r_list = default_r[default_r < maxdr]

    frames = _as_frames(v)
    hlvi = np.zeros((r_list.size, bin_centers.size), dtype=float)
    htvi = np.zeros((r_list.size, bin_centers.size), dtype=float)

    for fr in frames:
        assert isinstance(fr, xr.Dataset)
        u = np.asarray(fr["u"].values, dtype=float)
        w = np.asarray(fr["v"].values, dtype=float)
        ny, nx = u.shape
        for ir, rr in enumerate(r_list):
            # along x: longitudinal uses u, transverse uses v
            if rr < nx:
                dvlx = u[:, rr:] - u[:, : nx - rr]
                dvty = w[:, rr:] - w[:, : nx - rr]
                if not include_zero:
                    vmask = (u[:, rr:] != 0) & (u[:, : nx - rr] != 0)
                    tmask = (w[:, rr:] != 0) & (w[:, : nx - rr] != 0)
                    dvlx = np.where(vmask, dvlx, 0.0)
                    dvty = np.where(tmask, dvty, 0.0)
                vals_lx = dvlx.ravel()
                vals_ty = dvty.ravel()
                vals_lx = vals_lx[(vals_lx != 0.0) & np.isfinite(vals_lx)]
                vals_ty = vals_ty[(vals_ty != 0.0) & np.isfinite(vals_ty)]
                if vals_lx.size:
                    h, _ = np.histogram(vals_lx, bins=edges)
                    hlvi[ir, :] += h
                if vals_ty.size:
                    h, _ = np.histogram(vals_ty, bins=edges)
                    htvi[ir, :] += h

            # along y: longitudinal uses v, transverse uses -u
            if rr < ny:
                dvly = w[rr:, :] - w[: ny - rr, :]
                dvtx = -u[rr:, :] + u[: ny - rr, :]
                if not include_zero:
                    lmask = (w[rr:, :] != 0) & (w[: ny - rr, :] != 0)
                    tmask = (u[rr:, :] != 0) & (u[: ny - rr, :] != 0)
                    dvly = np.where(lmask, dvly, 0.0)
                    dvtx = np.where(tmask, dvtx, 0.0)
                vals_ly = dvly.ravel()
                vals_tx = dvtx.ravel()
                vals_ly = vals_ly[(vals_ly != 0.0) & np.isfinite(vals_ly)]
                vals_tx = vals_tx[(vals_tx != 0.0) & np.isfinite(vals_tx)]
                if vals_ly.size:
                    h, _ = np.histogram(vals_ly, bins=edges)
                    hlvi[ir, :] += h
                if vals_tx.size:
                    h, _ = np.histogram(vals_tx, bins=edges)
                    htvi[ir, :] += h

    pdflvi = np.zeros_like(hlvi)
    pdftvi = np.zeros_like(htvi)
    lsf = np.zeros((r_list.size, maxorder), dtype=float)
    tsf = np.zeros((r_list.size, maxorder), dtype=float)
    lsfabs = np.zeros((r_list.size, maxorder), dtype=float)
    tsfabs = np.zeros((r_list.size, maxorder), dtype=float)
    skew_long = np.zeros(r_list.size, dtype=float)
    skew_trans = np.zeros(r_list.size, dtype=float)
    flat_long = np.zeros(r_list.size, dtype=float)
    flat_trans = np.zeros(r_list.size, dtype=float)
    n_l = np.zeros(r_list.size, dtype=float)
    n_t = np.zeros(r_list.size, dtype=float)

    # centered structure functions (PIVMAT default)
    for ir in range(r_list.size):
        nl = float(np.sum(hlvi[ir, :]))
        nt = float(np.sum(htvi[ir, :]))
        n_l[ir] = nl
        n_t[ir] = nt
        if nl > 0:
            pdflvi[ir, :] = hlvi[ir, :] / (nl * binwidth)
        if nt > 0:
            pdftvi[ir, :] = htvi[ir, :] / (nt * binwidth)

        meanl = float(np.sum(pdflvi[ir, :] * bin_centers) * binwidth)
        meant = float(np.sum(pdftvi[ir, :] * bin_centers) * binwidth)
        for order in range(1, maxorder + 1):
            lsf[ir, order - 1] = float(np.sum(pdflvi[ir, :] * ((bin_centers - meanl) ** order)) * binwidth)
            tsf[ir, order - 1] = float(np.sum(pdftvi[ir, :] * ((bin_centers - meant) ** order)) * binwidth)
            lsfabs[ir, order - 1] = float(np.sum(pdflvi[ir, :] * (np.abs(bin_centers - meanl) ** order)) * binwidth)
            tsfabs[ir, order - 1] = float(np.sum(pdftvi[ir, :] * (np.abs(bin_centers - meant) ** order)) * binwidth)

        if lsf[ir, 1] != 0:
            skew_long[ir] = lsf[ir, 2] / (lsf[ir, 1] ** 1.5)
            flat_long[ir] = lsf[ir, 3] / (lsf[ir, 1] ** 2)
        if tsf[ir, 1] != 0:
            skew_trans[ir] = tsf[ir, 2] / (tsf[ir, 1] ** 1.5)
            flat_trans[ir] = tsf[ir, 3] / (tsf[ir, 1] ** 2)

    scaler = float(np.abs(v["x"].values[1] - v["x"].values[0])) if v["x"].size >= 2 else 1.0

    return xr.Dataset(
        data_vars={
            "hlvi": (("r", "bin"), hlvi),
            "htvi": (("r", "bin"), htvi),
            "pdflvi": (("r", "bin"), pdflvi),
            "pdftvi": (("r", "bin"), pdftvi),
            "lsf": (("r", "order"), lsf),
            "tsf": (("r", "order"), tsf),
            "lsfabs": (("r", "order"), lsfabs),
            "tsfabs": (("r", "order"), tsfabs),
            "skew_long": ("r", skew_long),
            "skew_trans": ("r", skew_trans),
            "flat_long": ("r", flat_long),
            "flat_trans": ("r", flat_trans),
            "n_long": ("r", n_l),
            "n_trans": ("r", n_t),
        },
        coords={
            "r": ("r", r_list.astype(float)),
            "bin": ("bin", bin_centers),
            "order": ("order", np.arange(1, maxorder + 1, dtype=int)),
        },
        attrs={"scaler": scaler, "binwidth": binwidth},
    )


def operf(
    op: str,
    f1: xr.Dataset | list[xr.Dataset],
    f2: xr.Dataset | list[xr.Dataset] | float | int | np.ndarray | None = None,
):
    """Perform algebraic/elementwise operations on vector/scalar fields.

    This is a pragmatic, PIVMAT-inspired helper similar to MATLAB's ``operf``.

    Parameters
    ----------
    op:
        Operation string (e.g. ``'+'``, ``'-'``, ``'.*'``, ``'./'``, comparisons like ``'>='``,
        or unary ops like ``'abs'``).
    f1:
        A vector Dataset (contains ``u`` and ``v``) or scalar Dataset (contains ``w``),
        or a list of such datasets.
    f2:
        Optional second operand: a Dataset (or list), or a scalar/array.
    """

    def _is_vector(ds: xr.Dataset) -> bool:
        return "u" in ds.data_vars and "v" in ds.data_vars

    def _is_scalar(ds: xr.Dataset) -> bool:
        return "w" in ds.data_vars and not _is_vector(ds)

    def _with_history(ds: xr.Dataset, entry: str) -> xr.Dataset:
        hist = list(ds.attrs.get("history", []))
        hist.append(entry)
        ds.attrs["history"] = hist
        return ds

    def _apply_unary(ds: xr.Dataset) -> xr.Dataset:
        op_s = str(op)
        op_l = op_s.lower()

        if op_s in ("+", "-"):
            out = ds.copy(deep=True)
            if _is_vector(out):
                sgn = 1.0 if op_s == "+" else -1.0
                out["u"] = (out["u"] * sgn).astype(float)
                out["v"] = (out["v"] * sgn).astype(float)
            elif _is_scalar(out):
                sgn = 1.0 if op_s == "+" else -1.0
                out["w"] = (out["w"] * sgn).astype(float)
            else:
                raise ValueError("operf: expected a vector (u,v) or scalar (w) Dataset")
            return _with_history(out, f"operf('{op_s}', ans)")

        unary_map: dict[str, object] = {
            "log": np.log,
            "exp": np.exp,
            "abs": np.abs,
            "real": np.real,
            "imag": np.imag,
            "conj": np.conj,
            "angle": np.angle,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
        }

        if op_l == "logabs":
            func = lambda a: np.log(np.abs(a))  # noqa: E731
        elif op_l in unary_map:
            func = unary_map[op_l]
        else:
            if not _is_vector(ds):
                raise ValueError("operf: invalid unary operation for scalar field")
            if not hasattr(ds, "piv"):
                raise ValueError("operf: xarray accessor 'piv' not available")
            out = ds.piv.vec2scal(op)  # type: ignore[attr-defined]
            if not isinstance(out, xr.Dataset):
                raise ValueError("operf: vec2scal returned unexpected type")
            return _with_history(out, f"operf('{op}', ans)")

        out = ds.copy(deep=True)
        if _is_vector(out):
            out["u"] = xr.apply_ufunc(func, out["u"])  # type: ignore[arg-type]
            out["v"] = xr.apply_ufunc(func, out["v"])  # type: ignore[arg-type]
        elif _is_scalar(out):
            out["w"] = xr.apply_ufunc(func, out["w"])  # type: ignore[arg-type]
        else:
            raise ValueError("operf: expected a vector (u,v) or scalar (w) Dataset")
        return _with_history(out, f"operf('{op}', ans)")

    def _apply_binary(ds: xr.Dataset, rhs: xr.Dataset | float | int | np.ndarray) -> xr.Dataset:
        op_s = str(op)
        op_l = op_s.lower()

        # Normalize operator aliases
        if op_s in (".*",):
            op_s = "*"
        if op_s in ("./",):
            op_s = "/"
        if op_s == "=":
            op_s = "=="

        # Field-field
        if isinstance(rhs, xr.Dataset):
            if _is_vector(ds) != _is_vector(rhs) or _is_scalar(ds) != _is_scalar(rhs):
                raise ValueError("operf: f1 and f2 must be of the same type")
            if op_s not in ("+", "-", "*", "/"):
                raise ValueError("operf: invalid binary operation for fields")
            a, b = xr.align(ds, rhs, join="exact")
            out = a.copy(deep=True)
            if _is_vector(out):
                if op_s == "+":
                    out["u"] = out["u"] + b["u"]
                    out["v"] = out["v"] + b["v"]
                elif op_s == "-":
                    out["u"] = out["u"] - b["u"]
                    out["v"] = out["v"] - b["v"]
                elif op_s == "*":
                    out["u"] = out["u"] * b["u"]
                    out["v"] = out["v"] * b["v"]
                else:
                    out["u"] = out["u"] / b["u"]
                    out["v"] = out["v"] / b["v"]
            else:
                if op_s == "+":
                    out["w"] = out["w"] + b["w"]
                elif op_s == "-":
                    out["w"] = out["w"] - b["w"]
                elif op_s == "*":
                    out["w"] = out["w"] * b["w"]
                else:
                    out["w"] = out["w"] / b["w"]
            return _with_history(out, f"operf('{op}', ans1, ans2)")

        out = ds.copy(deep=True)
        rhs_arr = np.asarray(rhs)

        binarize = op_l.startswith("b")
        cmp_op = op_l[1:] if binarize else op_l
        if cmp_op == "=":
            cmp_op = "=="

        def _cmp(a: xr.DataArray, thr: float) -> xr.DataArray:
            if cmp_op == ">":
                return a > thr
            if cmp_op == "<":
                return a < thr
            if cmp_op == ">=":
                return a >= thr
            if cmp_op == "<=":
                return a <= thr
            if cmp_op == "==":
                return a == thr
            raise ValueError("operf: invalid operation")

        # Field-number (vector)
        if _is_vector(out):
            if op_s in ("+", "-"):
                if rhs_arr.size == 1:
                    ru, rv = float(rhs_arr), float(rhs_arr)
                elif rhs_arr.size >= 2:
                    ru, rv = float(rhs_arr.flat[0]), float(rhs_arr.flat[1])
                else:
                    raise ValueError("operf: invalid numeric operand")
                if op_s == "+":
                    out["u"] = out["u"] + ru
                    out["v"] = out["v"] + rv
                else:
                    out["u"] = out["u"] - ru
                    out["v"] = out["v"] - rv
                return _with_history(out, f"operf('{op}', ans, {rhs_arr})")
            if op_s in ("*", "/"):
                r = float(rhs_arr.flat[0])
                out["u"] = (out["u"] * r) if op_s == "*" else (out["u"] / r)
                out["v"] = (out["v"] * r) if op_s == "*" else (out["v"] / r)
                return _with_history(out, f"operf('{op}', ans, {r})")
            if op_s in (".^", "^"):
                r = float(rhs_arr.flat[0])
                out["u"] = out["u"] ** r
                out["v"] = out["v"] ** r
                return _with_history(out, f"operf('{op}', ans, {r})")

            thr = float(rhs_arr.flat[0])
            m_u = _cmp(out["u"], thr)
            m_v = _cmp(out["v"], thr)
            if binarize:
                out["u"] = m_u.astype(float)
                out["v"] = m_v.astype(float)
            else:
                out["u"] = m_u.astype(float) * out["u"]
                out["v"] = m_v.astype(float) * out["v"]
            return _with_history(out, f"operf('{op}', ans, {thr})")

        # Field-number (scalar)
        if not _is_scalar(out):
            raise ValueError("operf: expected a vector (u,v) or scalar (w) Dataset")

        if op_s in ("+", "-", "*", "/", ".^", "^"):
            r = float(rhs_arr.flat[0])
            if op_s == "+":
                out["w"] = out["w"] + r
            elif op_s == "-":
                out["w"] = out["w"] - r
            elif op_s == "*":
                out["w"] = out["w"] * r
            elif op_s == "/":
                out["w"] = out["w"] / r
            else:
                out["w"] = out["w"] ** r
            return _with_history(out, f"operf('{op}', ans, {r})")

        thr = float(rhs_arr.flat[0])
        m = _cmp(out["w"], thr)
        if binarize:
            out["w"] = m.astype(float)
        else:
            out["w"] = m.astype(float) * out["w"]
        return _with_history(out, f"operf('{op}', ans, {thr})")

    f1_list = f1 if isinstance(f1, list) else [f1]
    if f2 is None:
        out_list = [_apply_unary(ds) for ds in f1_list]
        return out_list if isinstance(f1, list) else out_list[0]

    if isinstance(f2, list):
        out_list: list[xr.Dataset] = []
        for i, ds in enumerate(f1_list):
            rhs = f2[min(i, len(f2) - 1)]
            out_list.append(_apply_binary(ds, rhs))
        return out_list if isinstance(f1, list) else out_list[0]

    out_list = [_apply_binary(ds, f2) for ds in f1_list]
    return out_list if isinstance(f1, list) else out_list[0]


def _as_field_list(f: xr.Dataset | list[xr.Dataset]) -> list[xr.Dataset]:
    return f if isinstance(f, list) else [f]


def _with_history(ds: xr.Dataset, entry: str) -> xr.Dataset:
    out = ds.copy(deep=True)
    hist = list(out.attrs.get("history", []))
    hist.append(entry)
    out.attrs["history"] = hist
    return out


def _is_vector(ds: xr.Dataset) -> bool:
    return "u" in ds.data_vars and "v" in ds.data_vars


def _is_scalar(ds: xr.Dataset) -> bool:
    return "w" in ds.data_vars and not _is_vector(ds)


def setoriginf(f: xr.Dataset | list[xr.Dataset], P0: ArrayLike) -> xr.Dataset | list[xr.Dataset]:
    """Set the origin (0,0) of a vector/scalar field (PIVMAT-compatible).

    Port of PIVMAT's ``setoriginf.m``.

    Parameters
    ----------
    f:
        Vector/scalar dataset (or list of datasets).
    P0:
        New origin as ``[x0, y0]`` in the same units as the coords.
    """

    p = np.asarray(P0, dtype=float).ravel()
    if p.size < 2:
        raise ValueError("P0 must be a 2-element sequence [x0, y0]")
    x0, y0 = float(p[0]), float(p[1])

    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        out = ds.assign_coords(x=ds["x"] - x0, y=ds["y"] - y0)
        out_list.append(_with_history(out, f"setoriginf(ans, [{x0}, {y0}])"))
    return out_list if isinstance(f, list) else out_list[0]


def shiftf(f: xr.Dataset | list[xr.Dataset], opt: str = "bottomleft") -> xr.Dataset | list[xr.Dataset]:
    """Shift the axis of a vector/scalar field (PIVMAT-compatible).

    Port of PIVMAT's ``shiftf.m``.

    Parameters
    ----------
    opt:
        One of: 'bottomleft'/'bl' (default), 'bottomright'/'br',
        'topleft'/'tl', 'topright'/'tr', 'center'/'c'/'middle'.
    """

    opt_l = str(opt).lower()
    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        if x.size == 0 or y.size == 0:
            out_list.append(ds)
            continue

        if opt_l in {"center", "c", "middle"}:
            sx = 0.5 * (float(x[0]) + float(x[-1]))
            sy = 0.5 * (float(y[0]) + float(y[-1]))
        elif opt_l in {"bottomleft", "bl"}:
            sx = float(x[0])
            sy = float(y[0])
        elif opt_l in {"bottomright", "br"}:
            sx = float(x[-1])
            sy = float(y[0])
        elif opt_l in {"topleft", "tl"}:
            sx = float(x[0])
            sy = float(y[-1])
        elif opt_l in {"topright", "tr"}:
            sx = float(x[-1])
            sy = float(y[-1])
        else:
            raise ValueError("opt must be one of: center/c/middle, bottomleft/bl, bottomright/br, topleft/tl, topright/tr")

        out = ds.assign_coords(x=ds["x"] - sx, y=ds["y"] - sy)
        out_list.append(_with_history(out, f"shiftf(ans, '{opt}')"))
    return out_list if isinstance(f, list) else out_list[0]


def smoothf(f: xr.Dataset | list[xr.Dataset], n: int = 3, opt: str = "") -> xr.Dataset | list[xr.Dataset]:
    r"""Temporal running-average smoothing (PIVMAT-compatible).

    Port of PIVMAT's ``smoothf.m``.

    Notes
    -----
    For a time series of length $L$ and window length $n$, the output length is
    $L-2\lfloor n/2 \rfloor$ (PIVMAT behavior).
    """

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        if "t" not in ds.dims:
            raise ValueError("smoothf requires a time dimension 't'")
        nt = int(ds.sizes["t"])
        cn = n // 2
        if nt < n:
            raise ValueError("smoothf requires len(t) >= n")

        frames: list[xr.Dataset] = []
        t_out = np.asarray(ds["t"].values, dtype=float)[cn : nt - cn]
        # Smoothing should not treat 0 as missing by default.
        opt_eff = str(opt)
        if "0" not in opt_eff:
            opt_eff = opt_eff + "0"

        for i in range(0, nt - n + 1):
            sub = ds.isel(t=slice(i, i + n))
            avg = sub.piv.averf(opt_eff)  # type: ignore[attr-defined]
            # Ensure each window-average has a unique time coordinate so concat stacks,
            # rather than aligning on identical t=0 values.
            avg = avg.assign_coords(t=np.asarray([t_out[i]], dtype=float))
            frames.append(avg)

        out = xr.concat(frames, dim="t")
        out.attrs = dict(ds.attrs)
        out_list.append(_with_history(out, f"smoothf(ans, {n}, '{opt}')"))
    return out_list if isinstance(f, list) else out_list[0]


def timederivativef(f: xr.Dataset | list[xr.Dataset], order: int = 2) -> xr.Dataset | list[xr.Dataset]:
    r"""Time derivative by finite differences (PIVMAT-compatible).

    Port of PIVMAT's ``timederivativef.m``.

    The time unit is not applied here (matches PIVMAT). Divide by $\Delta t$
    externally if needed.
    """

    order = int(order)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    def _diff_arr(a: np.ndarray) -> np.ndarray:
        if a.ndim != 3:
            raise ValueError("Expected arrays with dims (y,x,t)")
        if order == 1:
            return np.diff(a, axis=2)
        # order 2
        out = np.empty_like(a)
        if a.shape[2] == 1:
            out[...] = 0.0
            return out
        out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2]) / 2.0
        out[:, :, 0] = a[:, :, 1] - a[:, :, 0]
        out[:, :, -1] = a[:, :, -1] - a[:, :, -2]
        return out

    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        if "t" not in ds.dims:
            raise ValueError("timederivativef requires a time dimension 't'")

        if order == 1:
            out = ds.isel(t=slice(0, -1)).copy(deep=True)
            t_out = np.asarray(ds["t"].values, dtype=float)[:-1]
            out = out.assign_coords(t=("t", t_out))
        else:
            out = ds.copy(deep=True)

        if _is_vector(out):
            out["u"] = xr.DataArray(_diff_arr(np.asarray(ds["u"].values, dtype=float)), dims=("y", "x", "t"), attrs=ds["u"].attrs)
            out["v"] = xr.DataArray(_diff_arr(np.asarray(ds["v"].values, dtype=float)), dims=("y", "x", "t"), attrs=ds["v"].attrs)
        elif _is_scalar(out):
            out["w"] = xr.DataArray(_diff_arr(np.asarray(ds["w"].values, dtype=float)), dims=("y", "x", "t"), attrs=ds["w"].attrs)
        else:
            raise ValueError("timederivativef: expected a vector (u,v) or scalar (w) Dataset")

        out_list.append(_with_history(out, f"timederivativef(ans,{order})"))
    return out_list if isinstance(f, list) else out_list[0]


def zerotonanfield(f: xr.Dataset | list[xr.Dataset]) -> xr.Dataset | list[xr.Dataset]:
    """Convert 0 elements to NaNs in fields (PIVMAT-compatible)."""

    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        out = ds.copy(deep=True)
        if _is_vector(out):
            out["u"] = out["u"].where(out["u"] != 0)
            out["v"] = out["v"].where(out["v"] != 0)
        elif _is_scalar(out):
            out["w"] = out["w"].where(out["w"] != 0)
        else:
            raise ValueError("zerotonanfield: expected a vector (u,v) or scalar (w) Dataset")
        out_list.append(_with_history(out, "zerotonanfield(ans)"))
    return out_list if isinstance(f, list) else out_list[0]


def zeropadf(f: xr.Dataset | list[xr.Dataset]) -> xr.Dataset | list[xr.Dataset]:
    """Zero-pad a rectangular field to a square (PIVMAT-compatible)."""

    def _pad(ds: xr.Dataset) -> xr.Dataset:
        ny = int(ds.sizes.get("y", 0))
        nx = int(ds.sizes.get("x", 0))
        if nx == ny:
            return ds
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        dx = float(x[1] - x[0]) if x.size >= 2 else 1.0
        dy = float(y[1] - y[0]) if y.size >= 2 else 1.0

        out = ds.copy(deep=True)
        if nx > ny:
            pad = nx - ny
            y_new = y[0] + dy * np.arange(nx, dtype=float)
            out = out.reindex(y=y_new, fill_value=0.0)
        else:
            pad = ny - nx
            x_new = x[0] + dx * np.arange(ny, dtype=float)
            out = out.reindex(x=x_new, fill_value=0.0)
        return out

    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        out_list.append(_with_history(_pad(ds), "zeropadf(ans)"))
    return out_list if isinstance(f, list) else out_list[0]


def truncf(
    f: xr.Dataset | list[xr.Dataset],
    cut: float | str = 0.0,
    *opts: str,
) -> xr.Dataset | list[xr.Dataset]:
    """Truncate a field to the largest centered square (PIVMAT-compatible).

    Supports:
    - ``truncf(f)``: centered square
    - ``truncf(f, cut, 'phys')``: cut specified in physical units
    - ``truncf(f, 'nonzero')``: smallest rectangle excluding zeros
    """

    # PIVMAT allows calling truncf(f,'nonzero') with the option in place of `cut`.
    if isinstance(cut, str):
        opts = (cut,) + opts
        cut = 0.0

    opts_l = {str(o).lower() for o in opts}

    def _to_mesh_cut(ds: xr.Dataset, c: float) -> int:
        if "phys" not in opts_l:
            return int(round(float(c)))
        x = np.asarray(ds["x"].values, dtype=float)
        dx = abs(float(x[1] - x[0])) if x.size >= 2 else 1.0
        if dx == 0:
            dx = 1.0
        return int(round(float(c) / dx))

    def _nonzero_crop(ds: xr.Dataset) -> xr.Dataset:
        if _is_vector(ds):
            a = np.asarray(ds["u"].values, dtype=float)
            b = np.asarray(ds["v"].values, dtype=float)
            # any nonzero over time
            m = (a != 0) | (b != 0)
        elif _is_scalar(ds):
            w = np.asarray(ds["w"].values, dtype=float)
            m = w != 0
        else:
            raise ValueError("truncf: expected a vector (u,v) or scalar (w) Dataset")
        if m.ndim == 3:
            m2 = np.any(m, axis=2)
        else:
            m2 = m
        rows = np.where(np.any(m2, axis=1))[0]
        cols = np.where(np.any(m2, axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return ds.isel(y=slice(0, 0), x=slice(0, 0))
        return ds.isel(y=slice(int(rows[0]), int(rows[-1]) + 1), x=slice(int(cols[0]), int(cols[-1]) + 1))

    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        if "nonzero" in opts_l and float(cut) == 0.0:
            out = _nonzero_crop(ds)
            out_list.append(_with_history(out, "truncf(ans, 'nonzero')"))
            continue

        ny = int(ds.sizes.get("y", 0))
        nx = int(ds.sizes.get("x", 0))
        if ny == 0 or nx == 0:
            out_list.append(ds)
            continue

        cut_m = _to_mesh_cut(ds, cut)
        side = min(nx, ny)
        x0 = (nx - side) // 2
        y0 = (ny - side) // 2
        x1 = x0 + side
        y1 = y0 + side

        x0 += cut_m
        y0 += cut_m
        x1 -= cut_m
        y1 -= cut_m
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(nx, x1)
        y1 = min(ny, y1)
        if x1 < x0:
            x1 = x0
        if y1 < y0:
            y1 = y0

        out = ds.isel(x=slice(x0, x1), y=slice(y0, y1))
        out_list.append(_with_history(out, f"truncf(ans, {cut}, {opts})"))
    return out_list if isinstance(f, list) else out_list[0]


def spatiotempcorrf(f: xr.Dataset, *opts: str) -> xr.Dataset:
    """Spatio-temporal correlation function for a scalar time series (PIVMAT-compatible).

    Port of PIVMAT's ``spatiotempcorrf.m``.

    Input must be a scalar dataset with variable ``w`` and dims ``(y,x,t)``.

    Options
    -------
    - 'full': use all possible X and T (noisy for large lags)
    - 'verbose': print progress
    """

    opt_l = {str(o).lower() for o in opts}
    verbose = any(o.startswith("verb") for o in opt_l)
    full = any(o.startswith("full") for o in opt_l)

    if not _is_scalar(f):
        raise ValueError("spatiotempcorrf requires a scalar dataset with variable 'w'")
    if "t" not in f.dims:
        raise ValueError("spatiotempcorrf requires a time dimension 't'")

    w = np.asarray(f["w"].values, dtype=float)
    ny, nx, nt = w.shape
    x = np.asarray(f["x"].values, dtype=float)
    dx = float(x[1] - x[0]) if x.size >= 2 else 1.0

    if full:
        T = np.arange(nt, dtype=int)
        X = np.arange(nx, dtype=int)
    else:
        T = np.arange(nt // 2 + 1, dtype=int)
        X = np.arange(nx // 2 + 1, dtype=int)

    corpos = np.zeros((X.size, T.size), dtype=float)
    corneg = np.zeros((X.size, T.size), dtype=float)

    for it, lagT in enumerate(T):
        if verbose:
            print(f"{(it + 1) / max(1, T.size) * 100:.1f}%", end=", ")
        for ix, lagX in enumerate(X):
            acc_p = 0.0
            acc_n = 0.0
            for j in range(0, nt - lagT):
                a = w[:, : nx - lagX, j]
                b = w[:, lagX:, j + lagT]
                acc_p += float(np.mean(a * b))

                a2 = w[:, lagX:, j]
                b2 = w[:, : nx - lagX, j + lagT]
                acc_n += float(np.mean(a2 * b2))
            corpos[ix, it] = acc_p
            corneg[ix, it] = acc_n
    if verbose:
        print("\n")

    cor = np.vstack([corneg[:0:-1, :], corpos])
    # Normalize by C(0,0)
    cor = cor / float(cor[X.size - 1, 0])
    Xlags = dx * np.concatenate([-X[:0:-1], X]).astype(float)

    return xr.Dataset(
        data_vars={"cor": (("X", "T"), cor)},
        coords={"X": ("X", Xlags), "T": ("T", T.astype(float))},
        attrs={"unitX": f["x"].attrs.get("units", ""), "unitcor": f"({f['w'].attrs.get('units','')})^2"},
    )


def statf(s: xr.Dataset | list[xr.Dataset], maxorder: int = 6):
    """Statistics of a vector/scalar field (PIVMAT-compatible).

    Port of PIVMAT's ``statf.m``.

    - Zeros are treated as invalid and excluded.
    - For vector datasets, returns one dict per component (u, v).
    """

    maxorder = int(maxorder)
    if maxorder <= 0:
        raise ValueError("maxorder must be positive")

    fields = _as_field_list(s)
    if len(fields) == 0:
        raise ValueError("Empty input")

    ds0 = fields[0]
    if _is_vector(ds0):
        su = statf([d[["u"]].rename({"u": "w"}) for d in fields], maxorder)
        sv = statf([d[["v"]].rename({"v": "w"}) for d in fields], maxorder)
        return su, sv

    if not _is_scalar(ds0):
        raise ValueError("statf expects a vector (u,v) or scalar (w) Dataset")

    # Stack all samples across space and time.
    vecs: list[np.ndarray] = []
    zeros = 0
    for ds in fields:
        w = np.asarray(ds["w"].values, dtype=float)
        vec = w.ravel()
        zeros += int(np.sum(vec == 0))
        vecs.append(vec)
    f_vect = np.concatenate(vecs)
    nz = f_vect != 0
    f_vect = f_vect[nz]
    if f_vect.size == 0:
        f_vect = np.asarray([0.0], dtype=float)

    mean = float(np.mean(f_vect))
    std = float(np.std(f_vect, ddof=0))
    rms = float(np.sqrt(np.mean(f_vect**2)))
    stat: dict[str, object] = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "min": float(np.min(f_vect)),
        "max": float(np.max(f_vect)),
        "nfields": len(fields),
        "n": int(f_vect.size),
        "zeros": int(zeros),
        "mom": np.zeros(maxorder, dtype=float),
        "momabs": np.zeros(maxorder, dtype=float),
        "cmom": np.zeros(maxorder, dtype=float),
        "cmomabs": np.zeros(maxorder, dtype=float),
    }

    for order in range(1, maxorder + 1):
        stat["cmom"][order - 1] = float(np.mean((f_vect - mean) ** order))
        stat["cmomabs"][order - 1] = float(np.mean(np.abs(f_vect - mean) ** order))
        stat["mom"][order - 1] = float(np.mean(f_vect**order))
        stat["momabs"][order - 1] = float(np.mean(np.abs(f_vect) ** order))

    if maxorder >= 3 and float(stat["mom"][1]) != 0.0:
        stat["skewness"] = float(stat["mom"][2] / (stat["mom"][1] ** 1.5))
        stat["flatness"] = float(stat["mom"][3] / (stat["mom"][1] ** 2))
        stat["skewnessc"] = float(stat["cmom"][2] / (stat["cmom"][1] ** 1.5))
        stat["flatnessc"] = float(stat["cmom"][3] / (stat["cmom"][1] ** 2))

    stat["history"] = ["statf(ans)"]
    return stat


def stresstensor(v: xr.Dataset | list[xr.Dataset]):
    """Reynolds stress tensor (PIVMAT-compatible).

    Port of PIVMAT's ``stresstensor.m`` for 2-component vector datasets.

    Returns
    -------
    tuple
        ``(t, b)`` where ``t`` is the stress tensor and ``b`` the deviatoric tensor.
    """

    fields = _as_field_list(v)
    if len(fields) == 0:
        raise ValueError("Empty input")
    ds0 = fields[0]
    if not _is_vector(ds0):
        raise ValueError("stresstensor requires a vector dataset with variables 'u' and 'v'")

    u = np.concatenate([np.asarray(d["u"].values, dtype=float).ravel() for d in fields])
    w = np.concatenate([np.asarray(d["v"].values, dtype=float).ravel() for d in fields])
    valid = (u != 0) & (w != 0) & np.isfinite(u) & np.isfinite(w)
    if not np.any(valid):
        t = np.zeros((2, 2), dtype=float)
        b = np.zeros((2, 2), dtype=float)
        return t, b
    u = u[valid]
    w = w[valid]
    t = np.zeros((2, 2), dtype=float)
    t[0, 0] = float(np.mean(u * u))
    t[1, 1] = float(np.mean(w * w))
    t[0, 1] = t[1, 0] = float(np.mean(u * w))
    tr = float(np.trace(t))
    if tr == 0.0:
        b = np.zeros_like(t)
    else:
        b = t / tr - np.eye(2) / 2.0
    return t, b


def subsbr(f: xr.Dataset | list[xr.Dataset], r0: ArrayLike | None = None) -> xr.Dataset | list[xr.Dataset]:
    """Subtract the mean solid-body rotation (PIVMAT-compatible)."""

    out_list: list[xr.Dataset] = []
    for ds in _as_field_list(f):
        if not _is_vector(ds):
            raise ValueError("subsbr requires a vector dataset with variables 'u' and 'v'")

        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        if r0 is None:
            r0x = 0.5 * (float(x[0]) + float(x[-1]))
            r0y = 0.5 * (float(y[0]) + float(y[-1]))
        else:
            rr = np.asarray(r0, dtype=float).ravel()
            r0x, r0y = float(rr[0]), float(rr[1])

        # Mean vorticity over space/time
        u = np.asarray(ds["u"].values, dtype=float)
        v = np.asarray(ds["v"].values, dtype=float)
        dx = float(x[1] - x[0]) if x.size >= 2 else 1.0
        dy = float(y[1] - y[0]) if y.size >= 2 else 1.0
        x_units = str(ds["x"].attrs.get("units", "")).lower()
        scale = 1000.0 if "mm" in x_units else 1.0
        dx_m = dx / scale
        dy_m = dy / scale
        dvdx = np.gradient(v, dx_m, axis=1, edge_order=1)
        dudy = np.gradient(u, dy_m, axis=0, edge_order=1)
        rot = dvdx - dudy
        meanrot = float(np.nanmean(rot))

        ycol = ((y[:, None] - r0y) / scale).astype(float)  # (ny,1)
        xrow = ((x[None, :] - r0x) / scale).astype(float)  # (1,nx)
        sbr_u = np.broadcast_to(-ycol * meanrot / 2.0, (y.size, x.size))
        sbr_v = np.broadcast_to(+xrow * meanrot / 2.0, (y.size, x.size))

        out = ds.copy(deep=True)
        out["u"] = out["u"] - xr.DataArray(sbr_u[:, :, None], dims=("y", "x", "t"))
        out["v"] = out["v"] - xr.DataArray(sbr_v[:, :, None], dims=("y", "x", "t"))
        out_list.append(_with_history(out, f"subsbr(ans, [{r0x}, {r0y}])"))
    return out_list if isinstance(f, list) else out_list[0]


def _rotate_about(ds: xr.Dataset, theta_rad: float, x0: float, y0: float) -> xr.Dataset:
    if _nd_map_coordinates is None:
        raise ImportError("subsbr2 requires SciPy (scipy.ndimage.map_coordinates)")

    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return ds

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    # Target grid points
    xx, yy = np.meshgrid(x, y)
    # Antecedent (inverse rotation)
    ct = float(np.cos(theta_rad))
    st = float(np.sin(theta_rad))
    xa = x0 + (xx - x0) * ct - (yy - y0) * st
    ya = y0 + (xx - x0) * st + (yy - y0) * ct

    # Convert physical -> fractional indices
    ix = (xa - x[0]) / dx
    iy = (ya - y[0]) / dy
    coords = np.vstack([iy.ravel(), ix.ravel()])

    out = ds.copy(deep=True)
    for name in ("u", "v"):
        arr = np.asarray(ds[name].values, dtype=float)
        out_arr = np.zeros_like(arr)
        for ti in range(arr.shape[2]):
            samp = _nd_map_coordinates(arr[:, :, ti], coords, order=1, mode="constant", cval=0.0).reshape(ny, nx)
            out_arr[:, :, ti] = samp
        out[name] = xr.DataArray(out_arr, dims=("y", "x", "t"), attrs=ds[name].attrs)

    # Rotate vector components
    u = out["u"].values
    v = out["v"].values
    out["u"].values[...] = u * ct - v * st
    out["v"].values[...] = u * st + v * ct
    return out


def subsbr2(
    f: xr.Dataset | list[xr.Dataset],
    dt: float = 1.0,
    r0: ArrayLike | None = None,
) -> xr.Dataset | list[xr.Dataset]:
    """Subtract mean rotation and compensate integrated camera rotation (PIVMAT-compatible)."""

    dt = float(dt)
    out_list: list[xr.Dataset] = []
    theta = 0.0

    for ds in _as_field_list(f):
        out = subsbr(ds, r0=r0)
        assert isinstance(out, xr.Dataset)

        # Recompute mean vorticity after subtraction (to match PIVMAT's integrated omega estimate).
        x = np.asarray(out["x"].values, dtype=float)
        y = np.asarray(out["y"].values, dtype=float)
        u = np.asarray(out["u"].values, dtype=float)
        v = np.asarray(out["v"].values, dtype=float)
        dx = float(x[1] - x[0]) if x.size >= 2 else 1.0
        dy = float(y[1] - y[0]) if y.size >= 2 else 1.0
        x_units = str(out["x"].attrs.get("units", "")).lower()
        scale = 1000.0 if "mm" in x_units else 1.0
        dx_m = dx / scale
        dy_m = dy / scale
        dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx_m)
        dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy_m)
        meanrot = float(np.nanmean(dvdx - dudy))

        theta += (meanrot / 2.0) * dt

        if r0 is None:
            r0x = 0.5 * (float(x[0]) + float(x[-1]))
            r0y = 0.5 * (float(y[0]) + float(y[-1]))
        else:
            rr = np.asarray(r0, dtype=float).ravel()
            r0x, r0y = float(rr[0]), float(rr[1])

        out = _rotate_about(out, theta_rad=theta, x0=r0x, y0=r0y)
        out_list.append(_with_history(out, f"subsbr2(ans, {dt}, [{r0x}, {r0y}])"))

    return out_list if isinstance(f, list) else out_list[0]


def tempfilterf(v: xr.Dataset, indexpos: ArrayLike, *opts: str) -> xr.Dataset:
    """Fourier temporal filter of a vector/scalar time series (PIVMAT-compatible).

    Port of PIVMAT's ``tempfilterf.m``.

    Parameters
    ----------
    indexpos:
        Integer frequency index/indices (PIVMAT/Matlab 1-based indexing into FFT bins).

    Options
    -------
    - 'remove': remove specified indices instead of keeping them
    - 'complex': keep only positive frequencies (output may be complex)
    - 'phaseaverf': for single frequency index, phase-average one period
    """

    if "t" not in v.dims:
        raise ValueError("tempfilterf requires a time dimension 't'")

    nt = int(v.sizes["t"])
    if nt <= 0:
        raise ValueError("Empty time dimension")

    idx = np.asarray(indexpos, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("indexpos must be non-empty")
    if np.any(idx < 1) or np.any(idx > nt):
        raise ValueError("indexpos values must be in [1, len(t)]")

    opts_l = {str(o).lower() for o in opts}
    complex_mode = any(o.startswith("comp") for o in opts_l)
    remove = any(o.startswith("rem") for o in opts_l)

    idx0 = (idx - 1).astype(int)
    mask = np.zeros(nt, dtype=bool)
    if complex_mode:
        mask[idx0] = True
    else:
        neg = (-idx0) % nt
        mask[idx0] = True
        mask[neg] = True
    if remove:
        mask = ~mask

    def _filt(a: np.ndarray) -> np.ndarray:
        A = np.fft.fft(a, axis=2)
        A *= mask[None, None, :]
        out = np.fft.ifft(A, axis=2)
        return out if complex_mode else out.real

    out = v.copy(deep=True)
    if _is_vector(out):
        out["u"] = xr.DataArray(_filt(np.asarray(v["u"].values)), dims=("y", "x", "t"), attrs=v["u"].attrs)
        out["v"] = xr.DataArray(_filt(np.asarray(v["v"].values)), dims=("y", "x", "t"), attrs=v["v"].attrs)
    elif _is_scalar(out):
        out["w"] = xr.DataArray(_filt(np.asarray(v["w"].values)), dims=("y", "x", "t"), attrs=v["w"].attrs)
    else:
        raise ValueError("tempfilterf expects a vector (u,v) or scalar (w) Dataset")

    # Phase average option for a single index.
    if any(o.startswith("phase") for o in opts_l):
        if idx.size != 1:
            raise ValueError("Option 'phaseaverf' works only with a scalar frequency index")
        period = float(nt) / float(idx[0] - 1) if idx[0] > 1 else float(nt)
        out = out.piv.phaseaverf(period)  # type: ignore[attr-defined]

        if complex_mode:
            omega = 2.0 * np.pi / period
            t = np.arange(out.sizes["t"], dtype=float)
            ph = np.exp(-1j * omega * t)
            if _is_vector(out):
                out["u"] = out["u"] * xr.DataArray(ph, dims=("t",))
                out["v"] = out["v"] * xr.DataArray(ph, dims=("t",))
                out = out.isel(t=[0]).copy(deep=True)
                out["u"] = out["u"].mean(dim="t")
                out["v"] = out["v"].mean(dim="t")
            else:
                out["w"] = out["w"] * xr.DataArray(ph, dims=("t",))
                out = out.isel(t=[0]).copy(deep=True)
                out["w"] = out["w"].mean(dim="t")

    return _with_history(out, f"tempfilterf(ans, {idx.tolist()})")


def surfheight(
    dr: xr.Dataset | list[xr.Dataset],
    h0: float,
    H: float = np.inf,
    n: float = 1.33,
    ctr: ArrayLike | None = None,
    *opts: str,
) -> xr.Dataset | list[xr.Dataset]:
    """Surface height reconstruction for FS-SS (PIVMAT-compatible, simplified).

    Port of PIVMAT's ``surfheight.m``.

    This implementation reconstructs height from gradients using a Fourier Poisson solver
    (periodic boundary assumption). It supports the main options:
    - 'submean'
    - 'nosetzero'
    - 'remap' (requires SciPy)
    """

    h0 = float(h0)
    H = float(H)
    n = float(n)
    opts_l = {str(o).lower() for o in opts}
    submean = any(o.startswith("subm") for o in opts_l)
    nosetzero = any(o.startswith("nose") for o in opts_l)
    remap = any(o.startswith("rema") for o in opts_l)

    def _integrate_grad_fft(dhdx: np.ndarray, dhdy: np.ndarray, dx: float, dy: float) -> np.ndarray:
        # Solve Laplacian(h) = d/dx dhdx + d/dy dhdy in Fourier space.
        ny, nx = dhdx.shape
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        kx2d, ky2d = np.meshgrid(kx, ky)
        denom = kx2d * kx2d + ky2d * ky2d

        F = np.fft.fft2(dhdx)
        G = np.fft.fft2(dhdy)
        rhs = 1j * kx2d * F + 1j * ky2d * G
        Hhat = np.zeros_like(rhs)
        mask = denom != 0
        Hhat[mask] = -rhs[mask] / denom[mask]
        h = np.fft.ifft2(Hhat).real
        return h

    out_fields: list[xr.Dataset] = []
    for ds in _as_field_list(dr):
        if not _is_vector(ds):
            raise ValueError("surfheight expects a vector dataset with variables 'u' and 'v' (displacements)")

        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        dx = abs(float(x[1] - x[0])) if x.size >= 2 else 1.0
        dy = abs(float(y[1] - y[0])) if y.size >= 2 else 1.0

        if ctr is None:
            ctrx = float(np.mean(x))
            ctry = float(np.mean(y))
        else:
            cc = np.asarray(ctr, dtype=float).ravel()
            ctrx, ctry = float(cc[0]), float(cc[1])

        alpha = 1.0 - 1.0 / n
        factor = 1.0 / H - 1.0 / (alpha * h0)

        out = ds.isel(t=[0]).copy(deep=True)
        out = out.drop_vars([v for v in out.data_vars if v not in {"u", "v", "chc"}], errors="ignore")

        # Subtract mean displacement if requested.
        u = np.asarray(ds["u"].isel(t=0).values, dtype=float)
        v = np.asarray(ds["v"].isel(t=0).values, dtype=float)
        if submean:
            u = u - float(np.mean(u))
            v = v - float(np.mean(v))

        dhdx = u * factor
        dhdy = v * factor

        if remap:
            if _sp_griddata is None:
                raise ImportError("surfheight(...,'remap') requires SciPy (scipy.interpolate.griddata)")
            yy, xx = np.meshgrid(y, x, indexing="ij")
            xxmes = (1.0 - h0 / H) * (xx + u - ctrx) + ctrx
            yymes = (1.0 - h0 / H) * (yy + v - ctry) + ctry
            pts = np.column_stack([xxmes.ravel(), yymes.ravel()])
            grid = (xx.ravel(), yy.ravel())
            dhdx = _sp_griddata(pts, dhdx.ravel(), grid, method="cubic").reshape(dhdx.shape)
            dhdy = _sp_griddata(pts, dhdy.ravel(), grid, method="cubic").reshape(dhdy.shape)
            dhdx = np.nan_to_num(dhdx, nan=0.0)
            dhdy = np.nan_to_num(dhdy, nan=0.0)

        h = _integrate_grad_fft(dhdx, dhdy, dx=dx, dy=dy)
        if not nosetzero:
            h = h - float(np.mean(h)) + h0

        # Return scalar dataset with 'w'
        out = out.drop_vars(["u", "v"], errors="ignore")
        out["w"] = xr.DataArray(h[:, :, None], dims=("y", "x", "t"), attrs={"units": ds["x"].attrs.get("units", ""), "standard_name": "height"})
        out.attrs = dict(ds.attrs)
        out_fields.append(_with_history(out, f"surfheight(ans,{h0},{H},{n})"))

    return out_fields if isinstance(dr, list) else out_fields[0]


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
        fWin (xarray.Dataset): A moving-window view of the dataset.
        n (int): Rolling window radius. The window size is ``(2*n+1) x (2*n+1)``.

    Returns:
        xarray.DataArray: Γ1 value for the given rolling window.
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
        fWin (xarray.Dataset): A moving-window view of the dataset.
        n (int): Rolling window radius. The window size is ``(2*n+1) x (2*n+1)``.

    Returns:
        xarray.DataArray: Γ2 value for the given rolling window.
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